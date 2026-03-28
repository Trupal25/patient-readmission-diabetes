"""
Structured logging for the patient readmission pipeline.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Loading data...")
"""
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger that writes to stdout with a clean format.

    Args:
        name:  Usually ``__name__`` of the calling module.
        level: Logging level (default INFO).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this module is imported multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

    return logger
