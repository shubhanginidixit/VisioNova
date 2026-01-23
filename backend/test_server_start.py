#!/usr/bin/env python
"""
Minimal test server to verify backend is working
"""
import sys
import threading
import time
sys.path.insert(0, '.')

def start_server():
    try:
        from app import app
        print("Attempting to start Flask server...", flush=True)
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"ERROR starting server: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Try to connect
    try:
        import requests
        r = requests.get('http://127.0.0.1:5000/', timeout=2)
        print(f"SUCCESS! Server is running. Status: {r.status_code}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)
    
    # Keep server alive
    while True:
        time.sleep(1)
