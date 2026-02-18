# In logger.py
import logging
import sys
from config import load_config


def setup_logger():
    # Get the ROOT logger
    root_logger = logging.getLogger()

    try:
        config = load_config()
        # Default to True if key is missing
        is_logging_enabled = config.get('logging', False)
    except Exception:
        is_logging_enabled = False
    
    # Remove existing handlers to avoid duplication
    if is_logging_enabled and root_logger.hasHandlers():
        root_logger.handlers.clear()

    if is_logging_enabled:
        target_level = logging.INFO
    else:
        target_level = logging.ERROR

    root_logger.setLevel(target_level)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(target_level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

# Call this once when the app starts
setup_logger()

def get_logger(name: str = None):
    """Get logger instance."""
    if name:
        return logging.getLogger(name)
    return logging.getLogger()