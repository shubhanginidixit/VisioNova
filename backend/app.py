"""
VisioNova Backend API Server
Flask application providing fact-checking API endpoints.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from fact_check import FactChecker

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize the fact checker
fact_checker = FactChecker()


@app.route('/')
def home():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'VisioNova Fact-Check API',
        'version': '1.0.0'
    })


@app.route('/api/fact-check', methods=['POST'])
def check_fact():
    """
    Fact-check endpoint.
    
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
                'error': 'Missing "input" field in request body'
            }), 400
        
        user_input = data['input'].strip()
        
        if not user_input:
            return jsonify({
                'success': False,
                'error': 'Input cannot be empty'
            }), 400
        
        # Run fact-check
        result = fact_checker.check(user_input)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
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


@app.route('/api/fact-check/deep', methods=['POST'])
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
                'error': 'Missing "input" field in request body'
            }), 400
        
        user_input = data['input'].strip()
        
        if not user_input:
            return jsonify({
                'success': False,
                'error': 'Input cannot be empty'
            }), 400
        
        # Run deep fact-check
        result = fact_checker.deep_check(user_input)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
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
