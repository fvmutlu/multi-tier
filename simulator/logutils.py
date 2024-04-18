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
logging.config.dictConfig(LOGGING_CONFIG)
rootlogger = logging.getLogger()