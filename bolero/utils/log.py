# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Alexander Fabisch <afabisch@informatik.uni-bremen.de>

import os
import sys
import logging
from cStringIO import StringIO


class HideExtern(object):
    """Hide one of the standard streams for external components.

    Based on http://stackoverflow.com/a/14797594/915743

    Parameters
    ----------
    stream : string, optional (default: 'stdout')
        Either 'stdout' or 'stderr'.
    """
    def __init__(self, stream="stdout"):
        streams = ["stdout", "stderr"]
        if stream not in streams:
            raise ValueError("Stream '%s' not in %r" % (stream, streams))
        self.stream = stream

        self._target = os.open(os.devnull, os.O_WRONLY)
        self._fno = streams.index(stream) + 1
        self._origstream = eval("sys." + self.stream)

    def __enter__(self):
        self._origstream.flush()
        try:
            self._oldstdout_fno = os.dup(eval("sys." + self.stream).fileno())
        except:
            self._oldstdout_fno = None

        os.dup2(self._target, self._fno)
        os.close(self._target)

        if self.stream == "stdout":
            sys.stdout = StringIO()
        elif self.stream == "stderr":
            sys.stderr = StringIO()

    def __exit__(self, *_):
        if self.stream == "stdout":
            sys.stdout = self._origstream
        elif self.stream == "stderr":
            sys.stderr = self._origstream

        eval("sys." + self.stream).flush()
        if self._oldstdout_fno:
            os.dup2(self._oldstdout_fno, self._fno)
            os.close(self._oldstdout_fno)


def get_logger(obj, log_to_file, log_to_stdout):
    """Get logger for given object.

    Removes all previously assigned handlers from the logger.

    Parameters
    ----------
    obj : Unknown
        Some object

    log_to_file: optional, boolean or string (default: False)
        Log results to given file, it will be located in the $BL_LOG_PATH

    log_to_stdout: optional, boolean (default: False)
        Log to standard output

    Returns
    -------
    logger : Logger
        Logger object
    """
    logger = logging.getLogger(type(obj).__name__)
    logger.handlers = []  # Remove all handlers
    logger.setLevel(logging.DEBUG)
    if log_to_file:
        bl_log_path = os.environ.get("BL_LOG_PATH", ".")
        log_file_name = "%s/%s" % (bl_log_path, log_to_file)
        handler = logging.FileHandler(log_file_name)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    if log_to_stdout:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    if (not (log_to_file or log_to_stdout) and
            "NullHandler" in logging.__dict__):
        handler = logging.NullHandler()
        logger.addHandler(handler)
    return logger
