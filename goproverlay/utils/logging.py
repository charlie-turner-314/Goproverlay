import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = os.environ.get("GOPROVERLAY_LOG", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        ch = logging.StreamHandler()
        ch.setLevel(logger.level)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        logger.propagate = False
    return logger

