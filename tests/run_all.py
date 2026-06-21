"""
Run all FlyLLM tests.
Usage: python tests/run_all.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

loader = unittest.TestLoader()
suite  = loader.discover(os.path.dirname(__file__), pattern="test_*.py")

runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)

sys.exit(0 if result.wasSuccessful() else 1)
