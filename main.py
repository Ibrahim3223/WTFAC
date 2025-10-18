#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for autoshorts.
Run this file to generate a YouTube Short.
"""
import sys
import os
import shutil
import logging

# CRITICAL: Clear Python cache before starting
def clear_cache():
    """Clear all Python cache files"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    autoshorts_path = os.path.join(project_root, 'autoshorts')
    
    if os.path.exists(autoshorts_path):
        for root, dirs, files in os.walk(autoshorts_path):
            # Remove __pycache__ directories
            if '__pycache__' in dirs:
                cache_dir = os.path.join(root, '__pycache__')
                try:
                    shutil.rmtree(cache_dir)
                    print(f"[CACHE] Cleared: {cache_dir}")
                except Exception as e:
                    print(f"[CACHE] Warning: Could not clear {cache_dir}: {e}")
            
            # Remove .pyc files
            for file in files:
                if file.endswith('.pyc'):
                    pyc_file = os.path.join(root, file)
                    try:
                        os.remove(pyc_file)
                        print(f"[CACHE] Removed: {pyc_file}")
                    except Exception as e:
                        print(f"[CACHE] Warning: Could not remove {pyc_file}: {e}")

# Clear cache first
print("[CACHE] Clearing Python cache...")
clear_cache()
print("[CACHE] Cache cleared successfully\n")

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"[DEBUG] Python path: {project_root}")
print(f"[DEBUG] Checking autoshorts module...")

# Verify autoshorts exists
autoshorts_path = os.path.join(project_root, 'autoshorts')
if not os.path.exists(autoshorts_path):
    print(f"❌ ERROR: autoshorts directory not found at {autoshorts_path}")
    sys.exit(1)

init_file = os.path.join(autoshorts_path, '__init__.py')
if not os.path.exists(init_file):
    print(f"❌ ERROR: autoshorts/__init__.py not found")
    sys.exit(1)

print(f"✅ autoshorts module found at {autoshorts_path}")

# Now safe to import
try:
    from autoshorts.orchestrator import ShortsOrchestrator
    print("✅ Successfully imported ShortsOrchestrator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\n[DEBUG] Directory structure:")
    for root, dirs, files in os.walk(autoshorts_path):
        level = root.replace(autoshorts_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f'{subindent}{file}')
    sys.exit(1)


def main():
    """Main entry point."""
    print("=" * 60)
    print("  YouTube Shorts Generator v2.0")
    print("=" * 60)
    
    # ✅ Set up logging properly
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    try:
        print("\n🔧 Creating orchestrator...")
        orchestrator = ShortsOrchestrator()
        
        print("\n🎬 Starting video generation...\n")
        video_id = orchestrator.run()
        
        if video_id:
            print("\n" + "=" * 60)
            print(f"✅ SUCCESS! Video ID: {video_id}")
            print(f"   Watch: https://youtube.com/watch?v={video_id}")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("❌ Video generation failed")
            print("=" * 60)
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 130
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        
        # Always print full traceback for debugging
        import traceback
        print("\n[DEBUG] Full traceback:")
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
