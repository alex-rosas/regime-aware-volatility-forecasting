import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with the given name, inheriting the root configuration.
    Usage in any module:
        from src.logger import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configures the root logger once for the entire project.
    Call this at the entry point of your pipeline (e.g. main script or DVC stage).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
