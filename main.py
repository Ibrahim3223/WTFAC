#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for autoshorts.
Run this file to generate a YouTube Short.
"""
import sys
import os

# CRITICAL: Add project root to Python path
# This ensures autoshorts module can be imported
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
    
    try:
        orchestrator = ShortsOrchestrator()
        video_id = orchestrator.run()
        
        if video_id:
            print("\n" + "=" * 60)
            print(f"✅ SUCCESS!")
            print(f"🔗 https://youtube.com/watch?v={video_id}")
            print("=" * 60)
            return 0
        else:
            print("\n" + "=" * 60)
            print("⏭️ Video created but not uploaded")
            print("=" * 60)
            return 0
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 1
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ ERROR: {e}")
        print("=" * 60)
        
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
