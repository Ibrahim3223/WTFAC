"""
Unit tests for Result pattern.
"""

import pytest
from autoshorts.core import Result


class TestResult:
    """Test Result pattern."""

    def test_ok_result(self):
        """Test successful Result."""
        result = Result.ok(42)

        assert result.is_ok()
        assert not result.is_err()
        assert result.unwrap() == 42
        assert result.unwrap_or(0) == 42

    def test_err_result(self):
        """Test error Result."""
        result = Result.err("something failed")

        assert result.is_err()
        assert not result.is_ok()
        assert result.unwrap_err() == "something failed"
        assert result.unwrap_or(0) == 0

    def test_unwrap_err_raises(self):
        """Test unwrap() raises on error Result."""
        result = Result.err("failed")

        with pytest.raises(ValueError, match="error Result"):
            result.unwrap()

    def test_unwrap_err_on_ok_raises(self):
        """Test unwrap_err() raises on ok Result."""
        result = Result.ok(42)

        with pytest.raises(ValueError, match="ok Result"):
            result.unwrap_err()

    def test_map_on_ok(self):
        """Test map() on successful Result."""
        result = Result.ok(2).map(lambda x: x * 2)

        assert result.is_ok()
        assert result.unwrap() == 4

    def test_map_on_err(self):
        """Test map() on error Result (propagates error)."""
        result = Result.err("failed").map(lambda x: x * 2)

        assert result.is_err()
        assert result.unwrap_err() == "failed"

    def test_and_then_chain(self):
        """Test and_then() for chaining."""
        def divide(x: int) -> Result[float, str]:
            if x == 0:
                return Result.err("division by zero")
            return Result.ok(10.0 / x)

        result = Result.ok(2).and_then(divide)
        assert result.is_ok()
        assert result.unwrap() == 5.0

        result = Result.ok(0).and_then(divide)
        assert result.is_err()
        assert result.unwrap_err() == "division by zero"

    def test_or_else_fallback(self):
        """Test or_else() for error handling."""
        def fallback(e: str) -> Result[int, str]:
            return Result.ok(0)

        result = Result.err("failed").or_else(fallback)
        assert result.is_ok()
        assert result.unwrap() == 0

    def test_bool_conversion(self):
        """Test Result in boolean context."""
        assert Result.ok(42)
        assert not Result.err("failed")
