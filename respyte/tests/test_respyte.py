"""
Unit and regression test for the respyte package.
"""

# Import package, test suite, and other packages as needed
import respyte
import pytest
import sys

def test_respyte_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "respyte" in sys.modules
