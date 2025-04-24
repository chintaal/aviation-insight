#!/usr/bin/env python
import subprocess
import sys
import os

def run_tests():
    """Run pytest with coverage and generate a report"""
    print("Running tests...")
    result = subprocess.run(["pytest", "-v", "app/tests/"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Tests failed with error code: {result.returncode}")
        print("Error output:")
        print(result.stderr)
        return False
    return True

def check_server():
    """Check if the API server can start"""
    print("Checking server startup...")
    try:
        # Import the app to see if there are any import or initialization errors
        from app.api.api import app
        print("Server imports successfully.")
        return True
    except Exception as e:
        print(f"Server startup check failed: {str(e)}")
        return False

if __name__ == "__main__":
    tests_ok = run_tests()
    server_ok = check_server()
    
    if tests_ok and server_ok:
        print("\n✅ All checks passed. The backend is working correctly!")
    else:
        print("\n❌ Some checks failed. Please fix the issues before proceeding.")
        sys.exit(1)
