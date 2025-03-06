"""
Test configuration for the experimentation framework.
This file defines which tests should be run based on the available components.
"""

import importlib.util
import os
import sys

# Add the project root to the path so imports work correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Check which components are available
COMPONENTS = {
    "tracking": importlib.util.find_spec("tracking") is not None,
    "config": False,  # We'll check this differently
    "artifacts": False,  # We'll check this differently
    "evaluation_job": False,  # We'll check this differently
    "seaborn": importlib.util.find_spec("seaborn") is not None,
}

# Check for training_workflow components if the module exists
try:
    import training_workflow
    COMPONENTS["config"] = importlib.util.find_spec("training_workflow.config") is not None
    COMPONENTS["artifacts"] = importlib.util.find_spec("training_workflow.storage") is not None
    COMPONENTS["evaluation_job"] = importlib.util.find_spec("training_workflow.jobs.evaluation_job") is not None
except ImportError:
    # training_workflow module doesn't exist, so these components aren't available
    pass

# Print available components
print("Available components for testing:")
for component, available in COMPONENTS.items():
    print(f"  {component}: {'✓' if available else '✗'}")

# Define test suites based on available components
def get_test_suite():
    """Get a test suite based on available components"""
    import unittest
    
    # Always include the simple test
    test_suite = unittest.TestLoader().discover('tests', pattern='test_simple.py')
    
    # Add component-specific tests if available
    if COMPONENTS["tracking"]:
        try:
            tracking_tests = unittest.TestLoader().discover('tests/unit/tracking', pattern='test_*.py')
            test_suite.addTests(tracking_tests)
            print("Added tracking tests")
        except ImportError as e:
            print(f"Could not add tracking tests: {e}")
    
    if COMPONENTS["config"]:
        try:
            config_tests = unittest.TestLoader().discover('tests/unit/config', pattern='test_*.py')
            test_suite.addTests(config_tests)
            print("Added config tests")
        except ImportError as e:
            print(f"Could not add config tests: {e}")
    
    if all([COMPONENTS["tracking"], COMPONENTS["config"], COMPONENTS["artifacts"], 
            COMPONENTS["evaluation_job"], COMPONENTS["seaborn"]]):
        try:
            integration_tests = unittest.TestLoader().discover('tests/integration', pattern='test_*.py')
            test_suite.addTests(integration_tests)
            print("Added integration tests")
        except ImportError as e:
            print(f"Could not add integration tests: {e}")
    
    return test_suite

if __name__ == "__main__":
    import unittest
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(get_test_suite()) 