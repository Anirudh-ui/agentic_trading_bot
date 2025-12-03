"""
Enhanced Exception Handling Module
Provides custom exceptions with detailed logging and context
"""

import sys
from typing import Optional, Dict, Any
from custom_logging.my_logger import logger


class TradingBotException(Exception):
    """Base exception class for Trading Bot application"""
    
    def __init__(
        self, 
        error_message: Any, 
        error_detail: Optional[sys] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(error_message)
        self.error_message = str(error_message)
        self.context = context or {}
        
        if error_detail:
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb:
                self.file_name = exc_tb.tb_frame.f_code.co_filename
                self.line_number = exc_tb.tb_lineno
                self.function_name = exc_tb.tb_frame.f_code.co_name
            else:
                self.file_name = "unknown"
                self.line_number = 0
                self.function_name = "unknown"
        else:
            self.file_name = "unknown"
            self.line_number = 0
            self.function_name = "unknown"
        
        # Log the exception
        self._log_exception()
    
    def _log_exception(self):
        """Log exception details"""
        log_message = (
            f"Exception in [{self.file_name}] "
            f"function [{self.function_name}] "
            f"line [{self.line_number}]: {self.error_message}"
        )
        if self.context:
            log_message += f" | Context: {self.context}"
        
        logger.error(log_message)
    
    def __str__(self):
        return (
            f"Error in {self.file_name}:{self.line_number} "
            f"[{self.function_name}] - {self.error_message}"
        )


class DocumentProcessingException(TradingBotException):
    """Exception raised during document processing"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "document_processing",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


class MemoryManagerException(TradingBotException):
    """Exception raised during memory operations"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "memory_manager",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


class WorkflowException(TradingBotException):
    """Exception raised during workflow execution"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "workflow",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


class APIException(TradingBotException):
    """Exception raised during API calls"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "api",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


class VectorStoreException(TradingBotException):
    """Exception raised during vector store operations"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "vector_store",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


class CacheException(TradingBotException):
    """Exception raised during caching operations"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "cache",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


class SessionException(TradingBotException):
    """Exception raised during session management"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "session",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


class ValidationException(TradingBotException):
    """Exception raised during input validation"""
    
    def __init__(self, error_message: Any, error_detail: Optional[sys] = None, **kwargs):
        context = {
            "type": "validation",
            **kwargs
        }
        super().__init__(error_message, error_detail, context)


def handle_exception(func):
    """Decorator for exception handling with logging"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TradingBotException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise TradingBotException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                sys,
                context={"function": func.__name__, "args": str(args)[:100]}
            )
    return wrapper


async def async_handle_exception(func):
    """Async decorator for exception handling with logging"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except TradingBotException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise TradingBotException(
                f"Unexpected error in {func.__name__}: {str(e)}",
                sys,
                context={"function": func.__name__, "args": str(args)[:100]}
            )
    return wrapper