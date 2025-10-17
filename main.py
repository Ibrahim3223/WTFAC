#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry point for autoshorts.
Run this file to generate a YouTube Short.
"""
import sys
import os

# Ensure autoshorts is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autoshorts.orchestrator import ShortsOrchestrator


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
