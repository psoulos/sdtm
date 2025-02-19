import numpy as np
import time
import torch
import torch.nn as nn


class MemorySet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, x, index):
        #memory.data[:, index].copy_(x)
        memory.data[:, index] = x
        ctx.index = index
        return memory

    @staticmethod
    def backward(ctx, grad_out):
        index = ctx.index
        return grad_out, grad_out[:,index], None


memory_set = MemorySet.apply
n_samples = 50

batch_size = 128
steps = 50
memory_dim = 512
repeat_i = 10
f = nn.RNN(memory_dim, memory_dim, batch_first=True).to(device='cuda')
input_ = torch.randn((batch_size, memory_dim), dtype=torch.float32, device='cuda')

forward_times = []
backward_times = []
print('Testing custom function')
torch.cuda.empty_cache()
print('Memory allocated at the start: ', torch.cuda.memory_allocated())
for sample in range(n_samples):
    # Pre allocate the entire memory
    memory = torch.zeros((batch_size, steps + 1, memory_dim*repeat_i), device='cuda', dtype=torch.float32)
    # The input is placed in the first memory slot
    memory = memory_set(memory, input_.repeat(1, repeat_i), 0)
    for i in range(steps):
        # TODO just write a random tensor, no need to run the RNN
        #output = f(memory[:, :i + 1])
        torch.cuda.synchronize()
        t0 = time.time()
        memory = memory_set(memory, output[0][:, -1].repeat(1, repeat_i), i + 1)
        torch.cuda.synchronize()
        t1 = time.time()
        forward_times.append(t1 - t0)

    loss = (1 - memory[:, -1]).mean()
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t1 = time.time()
    backward_times.append(t1 - t0)

print('Forward times: {}'.format(np.mean(forward_times)))
print('Backward times: {}'.format(np.mean(backward_times)))
print('Memory allocated at the end: ', torch.cuda.memory_allocated())

forward_times = []
backward_times = []
print('Testing cat')
torch.cuda.empty_cache()
print('Memory allocated at the start: ', torch.cuda.memory_allocated())
for sample in range(n_samples):
    # Pre allocate the entire memory
    memory = input_.repeat(1, repeat_i).unsqueeze(1)
    for i in range(steps):
        output = f(memory[:, :i + 1])
        torch.cuda.synchronize()
        t0 = time.time()
        memory = torch.cat([memory, output[0][:, -1].repeat(1, repeat_i).unsqueeze(1)], dim=1)
        torch.cuda.synchronize()
        t1 = time.time()
        forward_times.append(t1 - t0)

    loss = (1 - memory[:, -1]).mean()
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    t1 = time.time()
    backward_times.append(t1 - t0)

print('Forward times: {}'.format(np.mean(forward_times)))
print('Backward times: {}'.format(np.mean(backward_times)))
print('Memory allocated at the end: ', torch.cuda.memory_allocated())