import logging
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s %(filename)s] %(levelname)s : %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "level": "WARNING",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "simulator": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "./sim_outputs/simulator.log",
            "maxBytes": 10000,
            "backupCount": 3,
            "delay": True,
        },
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "siminfo": {
            "handlers": ["default", "simulator"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

class NonRepetitiveLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name=name, level=level)
        self._message_cache = []

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
        msg_hash = hash(msg)
        if msg_hash in self._message_cache:
            return
        self._message_cache.append(msg_hash)
        super()._log(level, msg, args, exc_info, extra, stack_info)

logging.config.dictConfig(LOGGING_CONFIG)

rootlogger = NonRepetitiveLogger("test")