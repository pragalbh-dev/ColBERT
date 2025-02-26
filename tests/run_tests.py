#!/usr/bin/env python3
import unittest
import sys
import os

def run_tests(test_type=None):
    """
    Run the specified tests or all tests if no type is specified.
    
    Args:
        test_type (str, optional): Type of tests to run ('unit', 'integration', or None for all)
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Add the project root to the path so imports work correctly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    
    print(f"Project root: {project_root}")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    
    if test_type == 'unit':
        print("Running unit tests...")
        start_dir = os.path.join(os.path.dirname(__file__), 'unit')
    elif test_type == 'integration':
        print("Running integration tests...")
        start_dir = os.path.join(os.path.dirname(__file__), 'integration')
    else:
        print("Running all tests...")
        start_dir = os.path.dirname(__file__)
    
    print(f"Looking for tests in: {start_dir}")
    print(f"Directory exists: {os.path.exists(start_dir)}")
    if os.path.exists(start_dir):
        print(f"Directory contents: {os.listdir(start_dir)}")
    
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    # Print test count
    test_count = suite.countTestCases()
    print(f"Found {test_count} tests")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # Parse command line arguments
    test_type = None
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type not in ['unit', 'integration']:
            print(f"Unknown test type: {test_type}")
            print("Usage: python run_tests.py [unit|integration]")
            sys.exit(1)
    
    # Run tests
    success = run_tests(test_type)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 