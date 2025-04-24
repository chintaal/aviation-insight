#!/usr/bin/env python
"""
Master Test Runner for Aviation Weather API Hub

This script runs all tests for the backend and generates a coverage report.
It provides a unified way to run test suites by service or by test type.
"""
import argparse
import os
import subprocess
import sys
from typing import List, Optional


def run_command(cmd: List[str]) -> int:
    """Run a command and print its output in real-time"""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Print output in real-time
    for line in iter(process.stdout.readline, ""):
        print(line, end="")
    
    process.stdout.close()
    return process.wait()


def run_tests(
    test_path: Optional[str] = None,
    coverage: bool = False,
    verbose: bool = False,
    service: Optional[str] = None
) -> bool:
    """Run the tests with specified options"""
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add specific test path or service
    if test_path:
        cmd.append(test_path)
    elif service:
        service_map = {
            "metar": "app/tests/test_api_metar.py",
            "taf": "app/tests/test_api_taf.py",
            "pirep": "app/tests/test_api_pirep.py",
            "sigmet": "app/tests/test_api_sigmet.py",
            "all": "app/tests/",
        }
        if service not in service_map:
            print(f"Error: Unknown service '{service}'. Available services: {', '.join(service_map.keys())}")
            return False
        cmd.append(service_map[service])
    else:
        cmd.append("app/tests/")
    
    # Add coverage if requested
    if coverage:
        cmd = ["coverage", "run", "-m"] + cmd[1:]
    
    # Run the tests
    print(f"Running: {' '.join(cmd)}")
    result = run_command(cmd)
    
    # Generate coverage report if requested
    if coverage and result == 0:
        print("\nGenerating coverage report...")
        run_command(["coverage", "report", "-m"])
        run_command(["coverage", "html"])
        print("\nHTML coverage report generated in htmlcov/index.html")
    
    return result == 0


def check_imports() -> bool:
    """Check if all modules can be imported correctly"""
    print("Checking imports...")
    try:
        # Test importing the main app
        from app.api.api import app
        
        # Try importing all service modules
        from app.services.metar_service import AWCMetarService, AVWXMetarService
        from app.services.taf_service import AWCTafService, AVWXTafService
        from app.services.pirep_service import AWCPirepService, AVWXPirepService
        from app.services.sigmet_service import (
            AWCSigmetService, AWCAirmetService, AVWXSigmetService, AVWXAirmetService
        )
        
        # Try importing all schemas
        from app.schemas.weather import (
            MetarResponse, TafResponse, PirepResponse, SigmetResponse, AirmetResponse
        )
        
        print("All imports successful!")
        return True
    except Exception as e:
        print(f"Import error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master test runner for Aviation Weather API Hub"
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "-p", "--path", 
        help="Specific test path to run"
    )
    test_group.add_argument(
        "-s", "--service",
        choices=["metar", "taf", "pirep", "sigmet", "all"],
        help="Run tests for a specific service"
    )
    
    # Test options
    parser.add_argument(
        "-c", "--coverage", 
        action="store_true",
        help="Generate code coverage report"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Increase verbosity"
    )
    parser.add_argument(
        "--check-imports", 
        action="store_true",
        help="Only check if all modules can be imported correctly"
    )
    
    args = parser.parse_args()
    
    # Make sure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n" + "=" * 60)
    print(" Aviation Weather API Hub - Test Runner ")
    print("=" * 60 + "\n")
    
    if args.check_imports:
        if check_imports():
            print("\n✅ All modules can be imported correctly.")
            sys.exit(0)
        else:
            print("\n❌ Import checks failed. Please fix the issues before proceeding.")
            sys.exit(1)
    
    # Run the tests
    success = run_tests(
        test_path=args.path,
        coverage=args.coverage,
        verbose=args.verbose,
        service=args.service
    )
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)