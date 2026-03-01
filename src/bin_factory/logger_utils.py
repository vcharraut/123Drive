import logging
import os


def get_logger(name: str = "123Drive") -> logging.Logger:
    return logging.getLogger(name)


def setup_logger(level: str | int | None = None):
    if level is None:
        level = os.getenv("123DRIVE_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    for h in root.handlers[:]:
        root.removeHandler(h)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%d-%m %H:%M:%S"))
    root.addHandler(handler)

    return root
