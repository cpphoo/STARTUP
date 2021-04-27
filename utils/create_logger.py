import logging

__all__ = ['create_logger']

def create_logger(fname, logger_name):
    # Get a logger with name logger_name
    logger = logging.getLogger(logger_name)

    # File handler for log
    hdlr = logging.FileHandler(fname)
    # Format of the logging information 
    formatter = logging.Formatter('%(levelname)s %(message)s')
    
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)

    # Set the level to logging info, meaning anything information
    # with information level above info will be logged
    logger.setLevel(logging.INFO)

    return logger