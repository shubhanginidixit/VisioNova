import os
import sys

# Add project root to path so imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from app import app
except ImportError as e:
    print(f"[ERR] Failed to import app: {e}")
    sys.exit(1)

if __name__ == '__main__':
    print("======================================================================")
    print("VisioNova Backend API Server (Windows Optimized)")
    print("======================================================================")
    print("[OK] Starting API server on http://localhost:5000")
    print("[OK] Debug mode: ON (Reloader: OFF for Windows stability)")
    print("[OK] Performance: Threaded mode enabled")
    print("\nEndpoints:")
    print("  - POST /api/fact-check      - Fact checking")
    print("  - POST /api/detect-ai       - AI text detection")
    print("  - POST /api/detect-image    - AI image detection")
    print("======================================================================")
    
    try:
        # use_reloader=False prevents crashes on some Windows configurations
        # threaded=True handles concurrent requests from the frontend
        is_debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
        host = os.getenv('FLASK_HOST', '127.0.0.1')
        port = int(os.getenv('FLASK_PORT', '5000'))
        app.run(debug=is_debug, host=host, port=port, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"[ERR] Server failed: {e}")
