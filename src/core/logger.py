import logging
import sys
from logging.handlers import RotatingFileHandler

def get_custom_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """Creates a custom logger with console and rotating file handlers."""
    
    # 1. Instantiate the logger object
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if the function is called multiple times
    if logger.hasHandlers():
        return logger

    # 2. Set the global minimum threshold level
    logger.setLevel(logging.DEBUG)

    # 3. Create standardized formatters
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 4. Create and configure the Console Handler (Streams to Stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Show only INFO and worse in console
    console_handler.setFormatter(console_format)

    # 5. Create and configure the Rotating File Handler (Saves to Disk)
    # Automatically rotates when the file hits 5MB, keeping up to 3 backups
    file_handler = RotatingFileHandler(
        filename=log_file, 
        maxBytes=5_000_000, 
        backupCount=3, 
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)  # Capture everything down to DEBUG in file
    file_handler.setFormatter(file_format)

    # 6. Attach handlers to the logger instance
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
