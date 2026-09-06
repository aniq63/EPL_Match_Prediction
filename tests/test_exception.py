"""Tests for the custom exception utility."""

import sys

import pytest

from src.utils.exception import MyException, error_message_detail


class TestErrorMessageDetail:
    def test_formats_file_and_line(self):
        try:
            raise ValueError("boom")
        except ValueError as e:
            msg = error_message_detail(e, sys)
        assert "boom" in msg
        assert "Error occurred in python script" in msg
        assert "line number" in msg


class TestMyExceptionStr:
    def test_string_contains_original_message(self):
        try:
            raise ValueError("custom failure")
        except ValueError as e:
            exc = MyException("wrapper", sys)
        assert "wrapper" in str(exc)

    def test_is_exception_subclass(self):
        assert issubclass(MyException, Exception)

    def test_raised_and_caught(self):
        with pytest.raises(MyException):
            try:
                raise RuntimeError("inner")
            except RuntimeError:
                raise MyException("outer", sys)
