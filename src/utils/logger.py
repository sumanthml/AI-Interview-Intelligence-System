import logging


def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return logging.getLogger(name)