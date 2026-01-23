# 🚨 Backend Server Not Running - SOLUTION

## Problem
The frontend shows "Analyzing..." but never completes because **the backend API server is not running**.

## Solution: Start the Backend Server

### Simple Method (Recommended)

Open a terminal/command prompt and run:

```bash
cd "c:\Users\adm\OneDrive\Desktop\VisioNova\backend"
python run_server.py
```

You should see:

```
===================================================================
VisioNova Backend API Server
===================================================================

✓ Starting API server on http://localhost:5000
✓ Endpoints available:
  - POST /api/detect-ai
  - POST /api/detect-ai/upload
  - POST /api/fact-check

Press CTRL+C to stop

===================================================================

 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment...
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.29.76:5000
Press CTRL+C to quit
```

### Keep Terminal Open
**IMPORTANT:** Keep this terminal window open and running. The server must stay running for the frontend to work.

### Test the Backend

In a **separate** terminal window, run:

```bash
python -c "import requests; r = requests.post('http://127.0.0.1:5000/api/detect-ai', json={'text': 'hey'}, timeout=5); print(r.json())"
```

You should get a JSON response with prediction and confidence.

## Using the Frontend

Once the backend is running:

1. Open the frontend in your browser:
   ```
   file:///c:/Users/adm/OneDrive/Desktop/VisioNova/frontend/html/AnalysisDashboard.html
   ```

2. Paste text in the analysis box

3. Click "Analyze"

4. Wait for results (should complete in < 1 second for text detection)

## Troubleshooting

### "Still showing Analyzing..."
- Check if terminal with `run_server.py` is still open and running
- If terminal closed, restart it with the command above

### "Connection Refused"
- Make sure the backend terminal is still running
- Try accessing `http://localhost:5000` in your browser to test

### "ModuleNotFoundError"
- Make sure you're in the correct directory: `c:\Users\adm\OneDrive\Desktop\VisioNova\backend`
- Install dependencies: `pip install flask flask-cors flask-limiter requests beautifulsoup4 ddgs`

## Keeping Backend Running

To keep the backend running even if you close the terminal, use:

```bash
# On Windows, start it in a new window that stays open
start python run_server.py
```

Or install a process manager like:

```bash
pip install python-daemon
# Then configure it to run the app as a service
```

## Summary

✅ **Backend must be running for analysis to work**  
✅ **Use `python run_server.py` to start it**  
✅ **Keep the terminal open**  
✅ **Then use the frontend**  

The frontend cannot work without the backend API server running!
