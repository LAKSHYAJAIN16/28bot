#!/usr/bin/env python3
"""
Simple bot to keep laptop active by periodically pressing Scroll Lock key.
This prevents sleep mode and screen saver activation.
"""

import time
import threading
from datetime import datetime
import sys

try:
    import pyautogui
except ImportError:
    print("pyautogui not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyautogui"])
    import pyautogui

def press_scroll_lock():
    """Press the Scroll Lock key to keep system active."""
    try:
        # Press Scroll Lock (harmless key that doesn't affect most applications)
        pyautogui.press('scrolllock')
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Pressed Scroll Lock")
    except Exception as e:
        print(f"Error pressing key: {e}")

def keep_active(interval_seconds=60):
    """
    Keep the system active by pressing Scroll Lock every interval_seconds.
    
    Args:
        interval_seconds (int): Time between key presses in seconds (default: 60)
    """
    print(f"Starting keep-active bot...")
    print(f"Will press Scroll Lock every {interval_seconds} seconds")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        while True:
            press_scroll_lock()
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Main function with user input for interval."""
    print("=== Laptop Keep-Active Bot ===")
    print("This bot will periodically press Scroll Lock to keep your laptop active.")
    print("Useful for preventing sleep mode during long tasks.")
    print()
    
    # Get interval from user
    while True:
        try:
            interval_input = input("Enter interval in seconds (default 60): ").strip()
            if not interval_input:
                interval_seconds = 60
                break
            interval_seconds = int(interval_input)
            if interval_seconds <= 0:
                print("Interval must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nStarting bot with {interval_seconds}-second interval...")
    keep_active(interval_seconds)

if __name__ == "__main__":
    main()
