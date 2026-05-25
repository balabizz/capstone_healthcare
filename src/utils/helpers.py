"""Helper utilities."""

import logging
from src.config import LOG_LEVEL


def setup_logging(name: str) -> logging.Logger:
    """
    Setup logging for a module.
    
    Args:
        name: Module name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
