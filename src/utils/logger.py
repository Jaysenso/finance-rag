# In logger.py
import logging
import sys

def setup_logger():
    # Get the ROOT logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
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