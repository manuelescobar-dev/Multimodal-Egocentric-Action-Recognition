import logging
import coloredlogs
import sys


def setup_logger(name, logfile=None):
    logger_instance = logging.getLogger(name)
    coloredlogs.install(
        level="DEBUG",
        logger=logger_instance,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return logger_instance


def get_handler(logfile):
    i_handler = logging.FileHandler(logfile)
    i_handler.setLevel(logging.INFO)
    return i_handler


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception
logger = setup_logger("LOG")
