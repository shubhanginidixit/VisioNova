#!/usr/bin/env python
"""
Simple Flask app launcher without debug reloader
"""
import os
import sys

# Flush output immediately
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

from app import app

if __name__ == '__main__':
    print("\n" + "="*70, flush=True)
    print("VisioNova Backend API Server", flush=True)
    print("="*70, flush=True)
    print(f"\n✓ Starting API server on http://localhost:5000", flush=True)
    print(f"✓ Endpoints available:", flush=True)
    print(f"  - POST /api/detect-ai", flush=True)
    print(f"  - POST /api/detect-ai/upload", flush=True)
    print(f"  - POST /api/fact-check", flush=True)
    print(f"\nPress CTRL+C to stop\n", flush=True)
    print("="*70 + "\n", flush=True)
    
    try:
        # Run without debug and without reloader
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
