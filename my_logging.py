import logging


def verbose(self, message, *args, **kws):
    if self.isEnabledFor(VERBOSE):
        self._log(VERBOSE, message, args, **kws)


VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")
logging.Logger.verbose = verbose
logging.VERBOSE = VERBOSE
