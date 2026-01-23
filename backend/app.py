"""
VisioNova Backend API Server
Flask application providing fact-checking API endpoints.
"""
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from fact_check import FactChecker
from text_detector import AIContentDetector, TextExplainer, DocumentParser


app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="memory://"
)

# Initialize the fact checker and AI content detector
fact_checker = FactChecker()
ai_detector = AIContentDetector(use_ml_model=True)  # Enable ML text detection
text_explainer = TextExplainer()
doc_parser = DocumentParser()

# File upload limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.doc'}

# Input validation constants
MAX_CLAIM_LENGTH = 1000
ALLOWED_URL_SCHEMES = ['http', 'https']


def validate_input(user_input: str) -> dict:
    """
    Validate user input for security and constraints.
    
    Returns:
        dict with 'valid' (bool) and 'error' (str or None)
    """
    if not user_input or not user_input.strip():
        return {'valid': False, 'error': 'Input cannot be empty'}
    
    user_input = user_input.strip()
    
    # Check length
    if len(user_input) > MAX_CLAIM_LENGTH:
        return {
            'valid': False, 
            'error': f'Input too long (max {MAX_CLAIM_LENGTH} characters)'
        }
    
    # Check for HTML/script injection attempts
    dangerous_patterns = [
        r'<script[\s\S]*?>[\s\S]*?</script>',
        r'javascript:',
        r'onerror\s*=',
        r'onclick\s*='
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return {
                'valid': False,
                'error': 'Invalid input: potential security risk detected'
            }
    
    return {'valid': True, 'error': None}


@app.route('/')
@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint with system status."""
    cache_info = fact_checker.cache_info()
    return jsonify({
        'status': 'ok',
        'service': 'VisioNova Fact-Check API',
        'version': '1.0.0',
        'cache': cache_info,
        'endpoints': {
            'fact_check': '/api/fact-check (POST)',
            'deep_check': '/api/fact-check/deep (POST)',
            'feedback': '/api/fact-check/feedback (POST)'
        }
    })


@app.route('/api/fact-check', methods=['POST'])
@limiter.limit("5 per minute")
def check_fact():
    """
    Fact-check endpoint with rate limiting and input validation.
    
    Request body:
        {
            "input": "The claim or URL to check"
        }
    
    Response:
        {
            "success": true,
            "input": "original input",
            "input_type": "claim|question|url",
            "claim": "extracted claim",
            "verdict": "TRUE|FALSE|PARTIALLY TRUE|MISLEADING|UNVERIFIABLE",
            "confidence": 0-100,
            "confidence_breakdown": {...},
            "sources": [...],
            "source_count": number,
            "explanation": "..."
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "input" field in request body',
                'error_code': 'MISSING_INPUT'
            }), 400
        
        user_input = data['input']
        
        # Validate input
        validation = validate_input(user_input)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': validation['error'],
                'error_code': 'INVALID_INPUT'
            }), 400
        
        # Run fact-check
        result = fact_checker.check(user_input.strip())
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/fact-check', methods=['GET'])
def check_fact_get():
    """GET version of fact-check for simple testing."""
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({
            'success': False,
            'error': 'Missing "q" query parameter'
        }), 400
    
    result = fact_checker.check(query)
    return jsonify(result)


@app.route('/api/fact-check/feedback', methods=['POST'])
@limiter.limit("10 per minute")
def submit_feedback():
    """
    User feedback endpoint for reporting incorrect verdicts.
    
    Request body:
        {
            "claim": "The original claim",
            "original_verdict": "TRUE|FALSE|...",
            "user_verdict": "User's opinion on correct verdict",
            "reason": "Explanation (optional)",
            "additional_sources": ["url1", "url2"] (optional)
        }
    """
    try:
        from fact_check.feedback_handler import FeedbackHandler
        
        data = request.get_json()
        
        required_fields = ['claim', 'original_verdict', 'user_verdict']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}',
                    'error_code': 'MISSING_FIELD'
                }), 400
        
        handler = FeedbackHandler()
        result = handler.submit_feedback(
            claim=data['claim'],
            original_verdict=data['original_verdict'],
            user_verdict=data['user_verdict'],
            reason=data.get('reason'),
            additional_sources=data.get('additional_sources', [])
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-ai', methods=['POST'])
@limiter.limit("10 per minute")
def detect_ai_content():
    """
    Detect if the given text is AI-generated or human-written.
    
    Request body:
        {
            "text": "The text to analyze",
            "explain": true/false (optional, default false)
        }
    
    Response:
        {
            "success": true,
            "prediction": "ai_generated|human",
            "confidence": 0-100,
            "scores": {...},
            "metrics": {...},
            "detected_patterns": {...},
            "sentence_analysis": [...],
            "explanation": {...}  // if explain=true
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "text" field in request body',
                'error_code': 'MISSING_TEXT'
            }), 400
        
        text = data['text']
        explain = data.get('explain', False)
        
        # Validate input (increase limit for text detection)
        if not text or not text.strip():
            return jsonify({
                'success': False,
                'error': 'Input cannot be empty',
                'error_code': 'INVALID_INPUT'
            }), 400
        
        if len(text) > 10000:  # 10k char limit for text detection
            return jsonify({
                'success': False,
                'error': 'Input too long (max 10,000 characters)',
                'error_code': 'INVALID_INPUT'
            }), 400
        
        # Run detection
        result = ai_detector.predict(text.strip(), detailed=True)
        
        if "error" in result:
            return jsonify({
                'success': False,
                'error': result['error'],
                'error_code': 'DETECTION_ERROR'
            }), 400
        
        # Add Groq explanation if requested
        if explain:
            explanation = text_explainer.explain(result, text[:500])
            result['explanation'] = explanation
        
        result['success'] = True
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-ai/upload', methods=['POST'])
@limiter.limit("5 per minute")
def detect_ai_file_upload():
    """
    Detect AI-generated content in uploaded files (PDF, DOCX, TXT).
    Supports large documents through chunked processing.
    
    Request:
        multipart/form-data with 'file' field
        Optional: 'explain' (true/false)
    
    Response:
        Detection results with chunk analysis for large documents
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded',
                'error_code': 'NO_FILE'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'error_code': 'NO_FILE'
            }), 400
        
        # Check file extension
        import os
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                'success': False,
                'error': f'Unsupported file format: {ext}. Allowed: PDF, DOCX, TXT',
                'error_code': 'INVALID_FORMAT'
            }), 400
        
        # Read file content
        file_bytes = file.read()
        
        # Check file size
        if len(file_bytes) > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB',
                'error_code': 'FILE_TOO_LARGE'
            }), 400
        
        # Parse document
        parse_result = doc_parser.parse_bytes(file_bytes, file.filename)
        
        if parse_result.get('error'):
            return jsonify({
                'success': False,
                'error': parse_result['error'],
                'error_code': 'PARSE_ERROR'
            }), 400
        
        text = parse_result['text']
        chunks = parse_result['chunks']
        metadata = parse_result['metadata']
        
        # Decide processing method based on text length
        explain = request.form.get('explain', 'false').lower() == 'true'
        
        if len(text) <= 5000:
            # Small document - analyze as single text
            result = ai_detector.predict(text, detailed=True)
        else:
            # Large document - use chunked analysis
            result = ai_detector.analyze_chunks(chunks, include_per_chunk=True)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error'],
                'error_code': 'DETECTION_ERROR'
            }), 400
        
        # Add file metadata
        result['file_info'] = {
            'filename': file.filename,
            'format': metadata.get('format'),
            'char_count': metadata.get('char_count'),
            'pages': metadata.get('pages'),
            'paragraphs': metadata.get('paragraphs')
        }
        
        # Add Groq explanation if requested
        if explain:
            explanation = text_explainer.explain(result, text[:500])
            result['explanation'] = explanation
        
        result['success'] = True
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/fact-check/deep', methods=['POST'])
@limiter.limit("3 per minute")
def deep_check_fact():
    """
    Deep fact-check endpoint - performs enhanced analysis with multiple searches.
    
    Request body:
        {
            "input": "The claim or URL to check"
        }
    
    Response:
        Same as regular fact-check but with additional fields:
        - deep_scan: true
        - queries_used: number of search queries used
        - total_sources_found: total sources before deduplication
        - unique_sources: unique sources after deduplication
    """
    try:
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "input" field in request body',
                'error_code': 'MISSING_INPUT'
            }), 400
        
        user_input = data['input']
        
        # Validate input
        validation = validate_input(user_input)
        if not validation['valid']:
            return jsonify({
                'success': False,
                'error': validation['error'],
                'error_code': 'INVALID_INPUT'
            }), 400
        
        # Run deep fact-check
        result = fact_checker.deep_check(user_input.strip())
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


if __name__ == '__main__':
    print("Starting VisioNova Fact-Check API Server...")
    print("API available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /                    - Health check")
    print("  POST /api/fact-check      - Check a claim/URL")
    print("  GET  /api/fact-check?q=   - Check a claim (simple)")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
