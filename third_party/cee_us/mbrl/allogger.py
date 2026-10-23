"""Logging-only adapter: the backend returns metrics directly."""
class _Logger:
    logdir = None
    def log(self, *args, **kwargs):
        pass

def get_logger(*args, **kwargs):
    return _Logger()
