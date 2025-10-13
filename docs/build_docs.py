#!/usr/bin/env python3
"""
Build script for Sphinx documentation.

This script automates the process of building the documentation,
including generating API documentation and building the HTML output.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(f"Error output: {result.stderr}")
        return False
    return True

def main():
    """Main build function."""
    # Get the docs directory
    docs_dir = Path(__file__).parent
    project_root = docs_dir.parent
    
    print("Building JAX Measure Transport Documentation")
    print("=" * 50)
    
    # Clean previous builds
    print("\n1. Cleaning previous builds...")
    if (docs_dir / "_build").exists():
        shutil.rmtree(docs_dir / "_build")
    if (docs_dir / "api").exists():
        shutil.rmtree(docs_dir / "api")
    
    # Generate API documentation
    print("\n2. Generating API documentation...")
    api_cmd = [
        "sphinx-apidoc",
        "-o", "api/",
        "../interfaces",
        "../networks", 
        "../samplers",
        "../trainers",
        "../eval",
        "../utils",
        "../data",
        "--separate"
    ]
    
    if not run_command(api_cmd, cwd=docs_dir):
        print("Failed to generate API documentation")
        return 1
    
    # Build HTML documentation
    print("\n3. Building HTML documentation...")
    html_cmd = ["sphinx-build", "-b", "html", ".", "_build/html"]
    
    if not run_command(html_cmd, cwd=docs_dir):
        print("Failed to build HTML documentation")
        return 1
    
    print("\n4. Documentation built successfully!")
    print(f"HTML documentation available at: {docs_dir / '_build' / 'html' / 'index.html'}")
    
    # Optional: Open in browser
    if len(sys.argv) > 1 and sys.argv[1] == "--open":
        import webbrowser
        webbrowser.open(f"file://{docs_dir / '_build' / 'html' / 'index.html'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
