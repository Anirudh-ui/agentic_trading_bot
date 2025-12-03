"""
Enhanced Logging System
Provides structured logging with multiple handlers and log levels
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys

# Create logs directory
LOG_DIR = Path(os.getcwd()) / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = LOG_DIR / LOG_FILE

# Log format with additional context
DETAILED_FORMAT = (
    "[%(asctime)s] %(levelname)-8s | "
    "%(name)s | %(filename)s:%(lineno)d | "
    "%(funcName)s() | %(message)s"
)

SIMPLE_FORMAT = (
    "[%(asctime)s] %(levelname)-8s | %(message)s"
)

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = "trading_bot_rag",
    level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup and configure logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # File Handler with rotation
    if log_to_file:
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console Handler with colors
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if sys.stdout.isatty():
            console_formatter = ColoredFormatter(SIMPLE_FORMAT, datefmt=DATE_FORMAT)
        else:
            console_formatter = logging.Formatter(SIMPLE_FORMAT, datefmt=DATE_FORMAT)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Error Handler - separate file for errors only
    error_log_path = LOG_DIR / f"errors_{datetime.now().strftime('%m_%d_%Y')}.log"
    error_handler = TimedRotatingFileHandler(
        error_log_path,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(DETAILED_FORMAT, datefmt=DATE_FORMAT)
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)
    
    return logger


# Create default logger instance
logger = setup_logger(
    name="trading_bot_rag",
    level=os.getenv("LOG_LEVEL", "INFO"),
    log_to_file=True,
    log_to_console=True
)


class LoggerContext:
    """Context manager for logging with additional context"""
    
    def __init__(self, logger, context: dict):
        self.logger = logger
        self.context = context
        self.original_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.original_factory)


def log_execution_time(func):
    """Decorator to log function execution time"""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting execution: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Completed {func.__name__} in {execution_time:.2f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Failed {func.__name__} after {execution_time:.2f}s: {str(e)}"
            )
            raise
    
    return wrapper


def log_async_execution_time(func):
    """Async decorator to log function execution time"""
    import time
    from functools import wraps
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.debug(f"Starting async execution: {func.__name__}")
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Completed async {func.__name__} in {execution_time:.2f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Failed async {func.__name__} after {execution_time:.2f}s: {str(e)}"
            )
            raise
    
    return wrapper


# Export logger and utilities
__all__ = [
    'logger',
    'setup_logger',
    'LoggerContext',
    'log_execution_time',
    'log_async_execution_time'
]