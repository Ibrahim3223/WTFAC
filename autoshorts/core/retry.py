"""
Retry utilities for resilient operations.

Provides decorators and functions for automatic retry with exponential backoff.
"""

import time
import logging
from typing import TypeVar, Callable, Optional, Type, Tuple
from functools import wraps

from .result import Result

T = TypeVar('T')
logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_name: Optional[str] = None
) -> Callable:
    """
    Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        backoff_factor: Multiplier for delay after each attempt (default: 2.0)
        max_delay: Maximum delay between attempts (default: 60.0)
        exceptions: Tuple of exception types to catch (default: all exceptions)
        logger_name: Logger name for retry messages (default: None)

    Example:
        @retry(max_attempts=5, initial_delay=2.0)
        def fetch_data():
            response = requests.get("https://api.example.com")
            response.raise_for_status()
            return response.json()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            _logger = logging.getLogger(logger_name or func.__module__)
            delay = initial_delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        _logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    _logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            # Should never reach here, but for type safety
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed without exception")

        return wrapper
    return decorator


def retry_with_result(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    logger_name: Optional[str] = None
) -> Callable:
    """
    Retry decorator that returns Result instead of raising.

    Example:
        @retry_with_result(max_attempts=3)
        def fetch_data() -> str:
            response = requests.get("https://api.example.com")
            response.raise_for_status()
            return response.json()

        # Returns Result[str, str]
        result = fetch_data()
        if result.is_ok():
            data = result.unwrap()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Result[T, str]]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Result[T, str]:
            _logger = logging.getLogger(logger_name or func.__module__)
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    value = func(*args, **kwargs)
                    return Result.ok(value)
                except Exception as e:
                    if attempt == max_attempts:
                        error_msg = f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        _logger.error(error_msg)
                        return Result.err(error_msg)

                    _logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            # Should never reach here
            return Result.err(f"{func.__name__} failed without exception")

        return wrapper
    return decorator


class RetryContext:
    """
    Context manager for retry operations.

    Example:
        with RetryContext(max_attempts=3, delay=1.0) as retry:
            for attempt in retry:
                try:
                    result = risky_operation()
                    retry.success()
                    break
                except Exception as e:
                    if not retry.should_retry(e):
                        raise
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 60.0,
        logger_name: Optional[str] = None
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.logger = logging.getLogger(logger_name or __name__)
        self._attempt = 0
        self._delay = initial_delay
        self._succeeded = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __iter__(self):
        self._attempt = 0
        self._succeeded = False
        return self

    def __next__(self):
        if self._succeeded or self._attempt >= self.max_attempts:
            raise StopIteration

        self._attempt += 1
        if self._attempt > 1:
            self.logger.info(f"Retry attempt {self._attempt}/{self.max_attempts}")
            time.sleep(self._delay)
            self._delay = min(self._delay * self.backoff_factor, self.max_delay)

        return self._attempt

    def success(self):
        """Mark the operation as successful."""
        self._succeeded = True

    def should_retry(self, exception: Exception) -> bool:
        """Check if should retry after an exception."""
        if self._attempt >= self.max_attempts:
            self.logger.error(
                f"Max attempts ({self.max_attempts}) reached: {exception}"
            )
            return False
        self.logger.warning(
            f"Attempt {self._attempt}/{self.max_attempts} failed: {exception}"
        )
        return True
