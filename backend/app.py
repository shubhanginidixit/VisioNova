"""
VisioNova Backend API Server
Flask application providing fact-checking API endpoints.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from fact_check import FactChecker
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"],
    storage_uri="memory://"
)

# Initialize the fact checker
fact_checker = FactChecker()

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
