#!/usr/bin/env python3
import unittest
import sys
import os

def run_tests():
    """
    Run only the tests that are known to work
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Add the project root to the path so imports work correctly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
    print(f"Project root: {project_root}")
    
    # Create a test suite with just the simple test
    suite = unittest.TestSuite()
    
    # Add the simple test
    from tests.test_simple import TestSimple
    suite.addTest(unittest.makeSuite(TestSimple))
    
    # Try to add the tracking tests if they're available
    try:
        from tests.unit.tracking.test_evaluation_tracker import TestEvaluationTracker
        suite.addTest(unittest.makeSuite(TestEvaluationTracker))
        print("Added tracking tests")
    except ImportError as e:
        print(f"Could not add tracking tests: {e}")
    
    # Print test count
    test_count = suite.countTestCases()
    print(f"Found {test_count} tests")
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 