"""
Result Pattern - Railway-Oriented Programming for robust error handling.

Instead of returning None or raising exceptions directly, use Result[T, E]:
- Result.ok(value) for success
- Result.err(error) for failure

Benefits:
- Explicit error handling
- Type-safe (with type hints)
- No hidden None checks
- Clear success/failure paths
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Callable, Union

T = TypeVar('T')
E = TypeVar('E')
U = TypeVar('U')


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """
    Result type for explicit error handling.

    Examples:
        >>> result = Result.ok(42)
        >>> result.is_ok()
        True
        >>> result.unwrap()
        42

        >>> result = Result.err("something failed")
        >>> result.is_err()
        True
        >>> result.unwrap_or(0)
        0
    """

    _value: Optional[T] = None
    _error: Optional[E] = None
    _is_ok: bool = False

    @staticmethod
    def ok(value: T) -> Result[T, E]:
        """Create a successful Result."""
        return Result(_value=value, _is_ok=True)

    @staticmethod
    def err(error: E) -> Result[T, E]:
        """Create a failed Result."""
        return Result(_error=error, _is_ok=False)

    def is_ok(self) -> bool:
        """Check if Result is successful."""
        return self._is_ok

    def is_err(self) -> bool:
        """Check if Result is failed."""
        return not self._is_ok

    def unwrap(self) -> T:
        """
        Get the success value or raise ValueError.
        Use only when you're certain the Result is Ok.
        """
        if not self._is_ok:
            raise ValueError(f"Called unwrap() on error Result: {self._error}")
        return self._value  # type: ignore

    def unwrap_or(self, default: T) -> T:
        """Get the success value or return default."""
        return self._value if self._is_ok else default  # type: ignore

    def unwrap_err(self) -> E:
        """
        Get the error value or raise ValueError.
        Use only when you're certain the Result is Err.
        """
        if self._is_ok:
            raise ValueError(f"Called unwrap_err() on ok Result: {self._value}")
        return self._error  # type: ignore

    def map(self, func: Callable[[T], U]) -> Result[U, E]:
        """
        Map a function over the success value.
        If Result is Err, returns the error unchanged.

        Example:
            >>> Result.ok(2).map(lambda x: x * 2)
            Result.ok(4)
            >>> Result.err("fail").map(lambda x: x * 2)
            Result.err("fail")
        """
        if self._is_ok:
            try:
                return Result.ok(func(self._value))  # type: ignore
            except Exception as e:
                return Result.err(e)  # type: ignore
        return Result.err(self._error)  # type: ignore

    def and_then(self, func: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """
        Chain operations that return Results (flatMap/bind).

        Example:
            >>> def divide(x: int) -> Result[float, str]:
            ...     if x == 0:
            ...         return Result.err("division by zero")
            ...     return Result.ok(10.0 / x)
            >>> Result.ok(2).and_then(divide)
            Result.ok(5.0)
            >>> Result.ok(0).and_then(divide)
            Result.err("division by zero")
        """
        if self._is_ok:
            try:
                return func(self._value)  # type: ignore
            except Exception as e:
                return Result.err(e)  # type: ignore
        return Result.err(self._error)  # type: ignore

    def or_else(self, func: Callable[[E], Result[T, E]]) -> Result[T, E]:
        """
        Provide fallback logic for errors.

        Example:
            >>> Result.err("failed").or_else(lambda e: Result.ok("fallback"))
            Result.ok("fallback")
        """
        if self._is_err:
            try:
                return func(self._error)  # type: ignore
            except Exception as e:
                return Result.err(e)  # type: ignore
        return Result.ok(self._value)  # type: ignore

    def __repr__(self) -> str:
        if self._is_ok:
            return f"Result.ok({self._value!r})"
        return f"Result.err({self._error!r})"

    def __bool__(self) -> bool:
        """Allow using Result in boolean context (True if Ok)."""
        return self._is_ok


# Convenience type aliases
Success = Result.ok
Failure = Result.err
