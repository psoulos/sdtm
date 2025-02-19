import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from node import Node, left_child, right_child, root_node_index


class TPR(nn.Module):
    def __init__(
        self,
        args,
        num_input_fillers,
        num_output_fillers,
        num_roles,
        d_filler=32,
        d_role=32,
        filler_emb_gain=1,
        learn_empty_filler=None,
        tied_io_languages=True,
        empty_filler_initialization='zero',
        device=None,
        sparse=False,
        nt_token_index=None,
    ) -> None:
        super().__init__()
        self.sparse = sparse
        self.learn_empty_filler = learn_empty_filler
        filler_padding_idx = None if learn_empty_filler else 0
        self.d_filler = d_filler
        self.num_output_fillers = num_output_fillers
        self.filler_emb = nn.Embedding(num_input_fillers, d_filler, padding_idx=filler_padding_idx, device=device)
        self.out = self.filler_emb

        if nt_token_index:
            with torch.no_grad():
                self.out.weight[nt_token_index] = self.out.weight[nt_token_index] / 1.5


        self.d_role = d_role
        self.num_roles = num_roles
        if self.sparse:
            role_matrix = None
        else:
            role_matrix = torch.empty((num_roles, d_role), device=device)
        self.register_buffer('role_matrix', role_matrix)
        if empty_filler_initialization == 'random':
            # If the empty filler is not 0, we need to keep track of the nodes at the bottom of the tree so that we can
            # bind the empty filler with these nodes and add this to the result of car/cdr since they now have the 0
            # vector at the bottom of their tree instead of the correct empty filler.
            self.bottom_node_indices = torch.arange(2 ** (math.log2(num_roles + 1) - 1) - 1,
                                                     num_roles, dtype=torch.long, device=device)
        self.max_interior_index = 2 ** (math.log2(num_roles) - 1)

        self._empty_leaves_tpr = None
        self.learn_filler_embed = args.learn_filler_embed
        self.empty_filler_initialization = empty_filler_initialization
        if not args.learn_filler_embed:
            self.filler_emb.requires_grad = False
            self.filler_emb.weight.requires_grad = False
        # Attributes
        self.proj_filler_to_unit_ball = args.proj_filler_to_unit_ball
        self.filler_emb_gain = filler_emb_gain
        self.reset_parameters(device=device)
    
    def reset_parameters(self, device=None):
        if not self.learn_filler_embed:
            nn.init.orthogonal_(
                self.filler_emb.weight,
                gain=math.sqrt(self.filler_emb.weight.shape[1])
            )#self.filler_emb_gain)
            if self.empty_filler_initialization == 'zero':
                self.filler_emb.weight.data[0, :] = 0
        else:
            pass
            # TODO: at some point, I need to investigate how to initialize the filler embeddings. This is tied up with
            # the fact that our filler embedding magnitude shrinks throughout processing, which I think messes with
            # the norm of the gradients, but this should be looked into.
            #nn.init.normal_(self.filler_emb.weight, mean=0, std=1/math.sqrt(self.filler_emb.weight.shape[1]))
            #if self.empty_filler_initialization == 'zero':
            #    self.filler_emb.weight.data[0].zero_()

        if not self.sparse:
            nn.init.orthogonal_(self.role_matrix, gain=1)
        # If the empty filler is not learned and random, we can precompute and store the TPR for the empty leaves
        if not self.learn_empty_filler and self.empty_filler_initialization == 'random':
            self._empty_leaves_tpr = self.empty_leaves_tpr(device=device)

    def empty_leaves_tpr(self, device=None):
        if self._empty_leaves_tpr is not None:
            return self._empty_leaves_tpr
        return torch.einsum('f,nr->fr', self.filler_emb(torch.tensor(0, device=device)), F.embedding(self.bottom_node_indices, self.role_matrix))

    def empty_tpr(self, device=None):
        return torch.einsum('f,nr->fr', self.filler_emb(torch.tensor(0, device=device)), self.role_matrix)

    def forward(self, tree_tensor):
        '''
        Given a binary tree represented by a tensor, construct the TPR
        '''
        if self.proj_filler_to_unit_ball:
            self.filler_emb.weight.data = self.filler_emb.weight.data / self.filler_emb.weight.data.norm(p=2, dim=-1).unsqueeze(1)
        if self.sparse:
            sparse_embeddings = self.filler_emb(tree_tensor.values())
            return SparseTPR(tree_tensor.indices(), sparse_embeddings) if tree_tensor.dim() == 2 \
                else SparseTPRBlock(tree_tensor.indices(), sparse_embeddings)
        else:
            x = self.filler_emb(tree_tensor)
            return torch.einsum('brm,rn->bmn', x, self.role_matrix) if tree_tensor.dim() == 2 \
                else torch.einsum('blrm,rn->blmn', x, self.role_matrix)

    # TODO: type_ as Data.Lang probably makes more sense
    def unbind(self, tpr_tensor, type_='output', decode=True):
        if self.sparse:
            if type_ == 'input':
                w = self.filler_emb.weight
            else:
                w = self.out.weight
            return SparseTPR(tpr_tensor.indices(), tpr_tensor.values() @ w.T)
        '''
        Given a TPR, unbind it
        '''
        unbinded = torch.einsum('bmn,rn->brm', tpr_tensor, self.role_matrix)
        if not decode:
            return unbinded
        if type == 'input':
            w = self.filler_emb.weight
        else:
            w = self.out.weight
        return torch.einsum('brm,fm->brf', unbinded, w)

@torch.no_grad()
def build_E(role_matrix, sparse=False):
    # If sparse, D and E are simulated by changing the indices in the sparse matrix
    if sparse:
        return None, None
    '''
    Build E matrices given the role embeddings (binary trees-only)
    '''
    d_role = role_matrix.size(1)
    E_l = role_matrix.new_zeros(d_role, d_role)
    E_r = role_matrix.new_zeros(d_role, d_role)
    def _add_to(mat, ind_from, ind_to):
        if ind_to >= role_matrix.size(0):
            return
        mat += torch.einsum('a,b->ab', role_matrix[ind_to], role_matrix[ind_from])
        _add_to(mat, ind_from*2+1, ind_to*2+1)
        _add_to(mat, ind_from*2+2, ind_to*2+2)
    _add_to(E_l, 0, 1)
    _add_to(E_r, 0, 2)
    E_l.requires_grad = False
    E_r.requires_grad = False
    if sparse:
        E_l = E_l.to_sparse()
        E_r = E_r.to_sparse()
    return E_l, E_r

@torch.no_grad()
def build_D(role_matrix, sparse=False):
    # If sparse, D and E are simulated by changing the indices in the sparse matrix
    if sparse:
        return None, None

    '''
    Build D matrices given the role embeddings (binary trees-only)
    '''
    d_role = role_matrix.size(1)
    D_l = role_matrix.new_zeros(d_role, d_role)
    D_r = role_matrix.new_zeros(d_role, d_role)
    def _add_to(mat, ind_from, ind_to):
        if ind_from >= role_matrix.size(0):
            return
        mat += torch.einsum('a,b->ab', role_matrix[ind_to], role_matrix[ind_from])
        _add_to(mat, ind_from*2+1, ind_to*2+1)
        _add_to(mat, ind_from*2+2, ind_to*2+2)
    _add_to(D_l, 1, 0)
    _add_to(D_r, 2, 0)
    D_l.requires_grad = False
    D_r.requires_grad = False
    if sparse:
        D_l = D_l.T.to_sparse()
        D_r = D_r.T.to_sparse()
    return D_l, D_r


def decoded_tpr_to_tree(decoded_tpr, loss_type='filler_xent', eps=.5, sparse=False, output_indices_mask=None):
    if loss_type == 'tpr_mse':
        contain_symbols = decoded_tpr.norm(p=2, dim=-1) > eps
        return torch.where(contain_symbols, decoded_tpr.argmax(dim=-1), 0)
    elif loss_type == 'filler_xent':
        if sparse:
            masked_values = decoded_tpr.values().detach().clone()
            # Don't decode tokens that are not in the output vocabulary
            masked_values[:, output_indices_mask] = -float('inf')
            return SparseTPR(decoded_tpr.indices(), masked_values.argmax(-1))
        else:
            return decoded_tpr.argmax(dim=-1)
    else:
        raise NotImplementedError


def decoded_tpr_to_tree_fn(loss_type='filler_xent', eps=.5, sparse=False, output_indices_mask=None):
    return lambda decoded_tpr: decoded_tpr_to_tree(decoded_tpr, loss_type, eps, sparse, output_indices_mask)


# works for binary trees only
def symbols_to_node_tree(index_tree, i2v, terminal_vocab=(), unary_vocab=()):
    def _traverse_and_detensorify(par, ind):
        cur = Node(i2v[index_tree[ind].item()])
        is_empty = index_tree[ind] == 0 or index_tree[ind] == 1
        has_non_empty_right_child = False
        has_non_empty_left_child = False
        if len(index_tree) > left_child(ind) and cur.label not in terminal_vocab:
            # work on the left child
            _, has_non_empty_left_child = _traverse_and_detensorify(cur, left_child(ind))
        if len(index_tree) > right_child(ind) and cur.label not in terminal_vocab and cur.label not in unary_vocab:
            # work on the right child
            _, has_non_empty_right_child = _traverse_and_detensorify(cur, right_child(ind))
        if is_empty and not has_non_empty_left_child and not has_non_empty_right_child:
            return par, False
        if par:
            par.children.append(cur)
        return cur, True
    node_tree, _ = _traverse_and_detensorify(None, root_node_index())
    return node_tree if node_tree else Node('')

# example usage in main.py: batch_symbols_to_node_tree(fully_decoded, train_data.ind2vocab)
def batch_symbols_to_node_tree(decoded_tpr_batch, i2v, terminal_vocab=(), unary_vocab=(), sparse=False):
    def s2nt(index_tree):
        return symbols_to_node_tree(index_tree, i2v, terminal_vocab, unary_vocab)
    if sparse:
        return list(map(s2nt, torch.sparse_coo_tensor(indices=decoded_tpr_batch.indices(),
                                                      values=decoded_tpr_batch.values())))
    else:
        return list(map(s2nt, decoded_tpr_batch))


def index_up_one_level_map(max_index, device=None):
    index_map = torch.empty(max_index + 1, dtype=torch.long, device=device)
    index_map[0] = -1
    index_map[1] = 0
    index_map[2] = 0
    for index in range(3, max_index+1):
        parent = (index - 1) // 2
        new_parent = index_map[parent]
        if index % 2 == 1:
            index_map[index] = new_parent * 2 + 1
        else:
            index_map[index] = new_parent * 2 + 2
    return index_map


def index_down_one_level_left_map(max_index, device=None):
    index_map = torch.empty(max_index + 1, dtype=torch.long, device=device)
    index_map[0] = 1
    for index in range(1, max_index+1):
        parent = (index - 1) // 2
        new_parent = index_map[parent]
        if index % 2 == 1:
            index_map[index] = new_parent * 2 + 1
        else:
            index_map[index] = new_parent * 2 + 2
    return index_map


def index_down_one_level_right_map(max_index, device=None):
    index_map = torch.empty(max_index + 1, dtype=torch.long, device=device)
    index_map[0] = 2
    for index in range(1, max_index+1):
        parent = (index - 1) // 2
        new_parent = index_map[parent]
        if index % 2 == 1:
            index_map[index] = new_parent * 2 + 1
        else:
            index_map[index] = new_parent * 2 + 2
    return index_map


class SparseTPR:
    def __init__(self, indices, values):
        self._indices = indices
        self._values = values
        self.device = self._values.device

    def indices(self):
        return self._indices

    def values(self):
        return self._values

    def batch_indices(self):
        return self._indices[0]

    def role_indices(self):
        return self._indices[1]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'SparseTPR(indices={self._indices}\nvalues={self._values})'


class SparseTPRBlock(SparseTPR):
    """This class accommodates the memory dimension."""
    def memory_slot_indices(self):
        return self._indices[1]

    def role_indices(self):
        return self._indices[2]

    def __repr__(self):
        return f'SparseTPRBlock(indices={self._indices}\nvalues={self._values})'
