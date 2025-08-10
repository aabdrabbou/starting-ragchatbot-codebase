#!/usr/bin/env python3
"""
Test runner for RAG system components

This script runs all tests and provides detailed output about failures
to help identify which components are broken.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_test_suite():
    """Run all tests and return results"""
    # Discover and load tests
    test_loader = unittest.TestLoader()
    test_dir = os.path.dirname(__file__)
    
    # Load all test modules
    test_suite = test_loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 70)
    print("RUNNING RAG SYSTEM COMPONENT TESTS")
    print("=" * 70)
    print()
    
    result = runner.run(test_suite)
    
    print()
    print("=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"FAILED: {test}")
            print(f"Reason: {traceback}")
            print("-" * 40)
    
    if result.errors:
        print("\nERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"ERROR: {test}")
            print(f"Reason: {traceback}")
            print("-" * 40)
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {len(result.failures + result.errors)} TEST(S) FAILED")
    
    return result

def main():
    """Main test runner"""
    try:
        result = run_test_suite()
        # Exit with error code if tests failed
        sys.exit(0 if result.wasSuccessful() else 1)
    except Exception as e:
        print(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()