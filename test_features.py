#!/usr/bin/env python3
"""
Test script for new thinking and searching features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work"""
    try:
        from src.inference.text_generator import TextGenerator, GenerationType
        print("✓ TextGenerator import successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_methods():
    """Test that new methods exist"""
    try:
        from src.inference.text_generator import TextGenerator
        methods = [method for method in dir(TextGenerator) if not method.startswith('_')]
        required_methods = ['generate', 'think_and_generate', 'search_and_generate']
        
        for method in required_methods:
            if method in methods:
                print(f"✓ Method {method} found")
            else:
                print(f"✗ Method {method} not found")
                return False
        return True
    except Exception as e:
        print(f"✗ Method check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing new thinking and searching features...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_methods()
    
    print("=" * 50)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")