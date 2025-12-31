#!/usr/bin/env python
"""
Master Runner for Aviation Weather API Hub

This script provides a unified interface for starting and managing the API server with various
configuration options, environment management, and utility functions.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
import webbrowser
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("master-runner")


def load_env_file(env_file: str = ".env") -> Dict[str, str]:
    """Load environment variables from .env file"""
    env_vars = {}
    if os.path.exists(env_file):
        logger.info(f"Loading environment from {env_file}")
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip('"\'')
    else:
        logger.warning(f"Environment file {env_file} not found")
    return env_vars


def check_api_keys() -> Dict[str, bool]:
    """Check if API keys are configured"""
    from app.core.config import settings
    
    return {
        "AWC": bool(settings.AWC_API_KEY),
        "AVWX": bool(settings.AVWX_API_KEY),
        "CHECKWX": bool(settings.CHECKWX_API_KEY)
    }


def print_api_key_status():
    """Print status of API keys"""
    api_keys = check_api_keys()
    print("\n=== API Key Status ===")
    for key, status in api_keys.items():
        status_str = "✅ Configured" if status else "❌ Not Configured"
        print(f"{key}: {status_str}")
    print()
    
    if not any(api_keys.values()):
        print("⚠️  WARNING: No API keys configured. Some endpoints may not work correctly.")
        print("   Add API keys to your .env file to enable all features.\n")


def run_server(host: str = "0.0.0.0", port: int = 8001, reload: bool = True, 
               log_level: str = "info", open_browser: bool = False, 
               env_file: str = ".env", debug: bool = False):
    """Run the FastAPI server"""
    # Set the environment file
    os.environ["ENV_FILE"] = env_file
    
    # Update environment variables from the env file
    env_vars = load_env_file(env_file)
    for key, value in env_vars.items():
        os.environ[key] = value

    # Print API key status
    print_api_key_status()
    
    # Prepare the uvicorn command
    cmd = [
        "uvicorn", "app.api.api:app",
        "--host", host,
        "--port", str(port),
        "--log-level", log_level
    ]
    
    if reload:
        cmd.append("--reload")
    
    if debug:
        print(f"Running command: {' '.join(cmd)}")
    
    # Print server URLs
    server_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    print(f"\n🚀 Starting Aviation Weather API Hub at {server_url}")
    print(f"📚 API Documentation:")
    print(f"   - Swagger UI: {server_url}/docs")
    print(f"   - ReDoc: {server_url}/redoc")
    print(f"📋 API Catalog: {server_url}/api/v1/catalog")
    print(f"🔍 Health Check: {server_url}/api/v1/health\n")
    
    # Open browser if requested
    if open_browser:
        webbrowser.open(f"{server_url}/docs")
        
    # Run the server
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⛔ Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)


def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import aiohttp
        import pytest
        import dotenv
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {str(e)}")
        print("Please install all required dependencies:")
        print("   pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master runner for Aviation Weather API Hub"
    )
    
    # Server configuration
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host IP to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8001,
        help="Port to run the server on (default: 8001)"
    )
    parser.add_argument(
        "--no-reload", 
        action="store_true",
        help="Disable auto-reload on code changes"
    )
    parser.add_argument(
        "--log-level", 
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the log level (default: info)"
    )
    parser.add_argument(
        "--open", 
        action="store_true",
        help="Open the API documentation in a browser"
    )
    
    # Environment configuration
    parser.add_argument(
        "--env-file", 
        default=".env",
        help="Path to the environment file (default: .env)"
    )
    
    # Debug options
    parser.add_argument(
        "--check-keys", 
        action="store_true",
        help="Check API key status and exit"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    # Make sure we're in the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("\n" + "=" * 60)
    print(" Aviation Weather API Hub - Server Runner ")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    if args.check_keys:
        print_api_key_status()
        sys.exit(0)
    
    run_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level,
        open_browser=args.open,
        env_file=args.env_file,
        debug=args.debug
    )