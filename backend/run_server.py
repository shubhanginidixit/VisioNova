#!/usr/bin/env python
"""
Simple Flask app launcher without debug reloader
"""
import os

from app import app

if __name__ == '__main__':
    print("\n" + "="*70)
    print("VisioNova Backend API Server")
    print("="*70)
    print(f"\n✓ Starting API server on http://localhost:5000")
    print(f"✓ Endpoints available:")
    print(f"  - POST /api/detect-ai")
    print(f"  - POST /api/detect-ai/upload")
    print(f"  - POST /api/fact-check")
    print(f"\nPress CTRL+C to stop\n")
    print("="*70 + "\n")
    
    # Run without debug and without reloader
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
