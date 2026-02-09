"""
VisioNova Backend API Server
Flask application providing fact-checking and AI detection API endpoints.
"""
import re
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from fact_check import FactChecker
from fact_check.feedback_handler import FeedbackHandler
from text_detector import AIContentDetector, TextExplainer, DocumentParser
from image_detector import (
    ImageDetector, MetadataAnalyzer, ELAAnalyzer, 
    WatermarkDetector, ContentCredentialsDetector, ImageExplainer,
    NoiseAnalyzer, EnsembleDetector, FastCascadeDetector, ML_DETECTORS_AVAILABLE
)


app = Flask(__name__, static_folder='../frontend', static_url_path='')
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
feedback_handler = FeedbackHandler()  # Initialize feedback handler
ai_detector = AIContentDetector(use_ml_model=True)  # Enable hybrid detection (ML + Statistical)
text_explainer = TextExplainer()
doc_parser = DocumentParser()

# Initialize image detector and AI explainer
# Basic detector (always available, fast)
image_detector = ImageDetector(use_gpu=False)
metadata_analyzer = MetadataAnalyzer()
ela_analyzer = ELAAnalyzer()
watermark_detector = WatermarkDetector()
content_credentials_detector = ContentCredentialsDetector()
noise_analyzer = NoiseAnalyzer()
image_explainer = ImageExplainer()  # Groq Vision API for AI-powered image analysis

# Ensemble detector (advanced, loads ML models on demand)
# Set load_ml_models=False initially for faster startup, models load on first use
ensemble_detector = None
try:
    ensemble_detector = EnsembleDetector(use_gpu=False, load_ml_models=False)
    print(f"✓ Ensemble detector initialized (ML models available: {ML_DETECTORS_AVAILABLE})")
except Exception as e:
    print(f"⚠ Ensemble detector not available: {e}")

# Fast cascade detector (speed-optimized, 3-5x faster for clear cases)
fast_detector = None
try:
    fast_detector = FastCascadeDetector(use_gpu=False, enable_fp16=False)
    print("✓ Fast cascade detector initialized (3-5x speedup for clear cases)")
except Exception as e:
    print(f"⚠ Fast cascade detector not available: {e}")

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


# ===========================================
# STATIC FILE ROUTES
# ===========================================

@app.route('/')
def serve_home():
    """Serve the homepage."""
    return app.send_static_file('html/homepage.html')

@app.route('/home')
def serve_home_alt():
    """Alternative route for homepage."""
    return app.send_static_file('html/homepage.html')

@app.route('/html/<path:filename>')
def serve_html(filename):
    """Serve HTML pages."""
    return app.send_static_file(f'html/{filename}')

@app.route('/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files."""
    return app.send_static_file(f'css/{filename}')

@app.route('/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files."""
    return app.send_static_file(f'js/{filename}')

# Friendly URL routes for each page
@app.route('/dashboard')
def serve_dashboard():
    """Serve the analysis dashboard."""
    return app.send_static_file('html/AnalysisDashboard.html')

@app.route('/fact-check')
def serve_fact_check_page():
    """Serve the fact check page."""
    return app.send_static_file('html/FactCheckPage.html')

@app.route('/image-result')
def serve_image_result():
    """Serve the image result page."""
    return app.send_static_file('html/ImageResultPage.html')

@app.route('/text-result')
def serve_text_result():
    """Serve the text result page."""
    return app.send_static_file('html/TextResultPage.html')

@app.route('/audio-result')
def serve_audio_result():
    """Serve the audio result page."""
    return app.send_static_file('html/AudioResultPage.html')

@app.route('/video-result')
def serve_video_result():
    """Serve the video result page."""
    return app.send_static_file('html/VideoResultPage.html')

@app.route('/report')
def serve_report():
    """Serve the report page."""
    return app.send_static_file('html/ReportPage.html')

@app.route('/api-test')
def serve_api_test():
    """Serve the API test page."""
    return app.send_static_file('html/APITest.html')

@app.route('/backend-test')
def serve_backend_test():
    """Serve the backend test page."""
    return app.send_static_file('html/BackendTest.html')


# Routes for direct HTML file access (without /html/ prefix)
# These are needed because the homepage uses relative URLs like "AnalysisDashboard.html"
@app.route('/AnalysisDashboard.html')
def serve_analysis_dashboard_direct():
    """Serve AnalysisDashboard via direct file access."""
    return app.send_static_file('html/AnalysisDashboard.html')

@app.route('/ImageResultPage.html')
def serve_image_result_direct():
    """Serve ImageResultPage via direct file access."""
    return app.send_static_file('html/ImageResultPage.html')

@app.route('/TextResultPage.html')
def serve_text_result_direct():
    """Serve TextResultPage via direct file access."""
    return app.send_static_file('html/TextResultPage.html')

@app.route('/AudioResultPage.html')
def serve_audio_result_direct():
    """Serve AudioResultPage via direct file access."""
    return app.send_static_file('html/AudioResultPage.html')

@app.route('/VideoResultPage.html')
def serve_video_result_direct():
    """Serve VideoResultPage via direct file access."""
    return app.send_static_file('html/VideoResultPage.html')

@app.route('/FactCheckPage.html')
def serve_factcheck_direct():
    """Serve FactCheckPage via direct file access."""
    return app.send_static_file('html/FactCheckPage.html')

@app.route('/ReportPage.html')
def serve_report_direct():
    """Serve ReportPage via direct file access."""
    return app.send_static_file('html/ReportPage.html')

@app.route('/homepage.html')
def serve_homepage_direct():
    """Serve homepage via direct file access."""
    return app.send_static_file('html/homepage.html')


# ===========================================
# API ENDPOINTS
# ===========================================

@app.route('/api/health')
def health_check():
    """Enhanced health check endpoint with system status."""
    cache_info = fact_checker.cache_info()
    return jsonify({
        'status': 'ok',
        'service': 'VisioNova API',
        'version': '2.0.0',
        'cache': cache_info,
        'features': {
            'watermark_detection': watermark_detector.watermark_lib_available,
            'c2pa_detection': content_credentials_detector.c2pa_available,
            'ensemble_detection': ensemble_detector is not None,
            'fast_detection': fast_detector is not None,
            'ml_models_available': ML_DETECTORS_AVAILABLE,
            'stable_signature_detection': watermark_detector.stable_signature_available if hasattr(watermark_detector, 'stable_signature_available') else False,
        },
        'endpoints': {
            'fact_check': '/api/fact-check (POST)',
            'deep_check': '/api/fact-check/deep (POST)',
            'detect_text': '/api/detect-ai (POST)',
            'detect_image': '/api/detect-image (POST)',
            'detect_image_fast': '/api/detect-image/fast (POST)',
            'detect_image_ensemble': '/api/detect-image/ensemble (POST)',
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


# ============================================
# IMAGE DETECTION ENDPOINTS
# ============================================

@app.route('/api/detect-image', methods=['POST'])
@limiter.limit("10 per minute")
def detect_ai_image():
    """
    Detect if an image is AI-generated.
    
    Uses a combination of:
    - Statistical analysis (frequency domain, noise patterns)
    - Metadata forensics
    - Watermark detection
    - C2PA Content Credentials
    - Groq Vision AI analysis (Llama 4 Scout) for visual artifact detection
    
    Request body:
        {
            "image": "base64_encoded_image_data",
            "filename": "optional_filename.jpg",
            "include_ela": true,  // optional, default false
            "include_metadata": true,  // optional, default true
            "include_watermark": true,  // optional, default true
            "include_c2pa": true,  // optional, default true
            "include_ai_analysis": true  // optional, default true - Groq Vision analysis
        }
    
    Response:
        {
            "success": true,
            "ai_probability": 75.5,
            "verdict": "LIKELY_AI",
            "verdict_description": "...",
            "analysis_scores": {...},
            "metadata": {...},
            "ela": {...},  // if include_ela=true
            "watermark": {...},  // if include_watermark=true
            "content_credentials": {...},  // if include_c2pa=true
            "ai_analysis": {  // if include_ai_analysis=true
                "visual_analysis": {...},
                "explanation": {...},
                "combined_verdict": {...}
            }
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image" field in request body',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        image_data = data['image']
        filename = data.get('filename', 'uploaded_image')
        include_ela = data.get('include_ela', True)
        include_metadata = data.get('include_metadata', True)
        include_watermark = data.get('include_watermark', True)
        include_c2pa = data.get('include_c2pa', True)
        include_ai_analysis = data.get('include_ai_analysis', True)  # Groq Vision AI analysis
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Check file size (max 50MB for images)
        max_image_size = 50 * 1024 * 1024
        if len(image_bytes) > max_image_size:
            return jsonify({
                'success': False,
                'error': f'Image too large (max {max_image_size // 1024 // 1024}MB)',
                'error_code': 'IMAGE_TOO_LARGE'
            }), 400
        
        # Run AI detection
        detection_result = image_detector.detect(image_bytes, filename)
        
        if not detection_result.get('success', False):
            return jsonify(detection_result), 400
        
        # Add metadata analysis if requested
        if include_metadata:
            metadata_result = metadata_analyzer.analyze(image_bytes)
            detection_result['metadata'] = metadata_result
            
            # Adjust AI probability based on metadata
            modifier = metadata_result.get('ai_probability_modifier', 0)
            original_prob = detection_result['ai_probability']
            adjusted_prob = max(0, min(100, original_prob + modifier))
            detection_result['ai_probability'] = round(adjusted_prob, 2)
            detection_result['probability_adjustment'] = modifier
        
        # Add ELA analysis if requested
        if include_ela:
            ela_result = ela_analyzer.analyze(image_bytes)
            detection_result['ela'] = ela_result
        
        # Add watermark detection if requested
        if include_watermark:
            watermark_result = watermark_detector.analyze(image_bytes)
            detection_result['watermark'] = watermark_result
            
            # Adjust AI probability if watermark found
            if watermark_result.get('watermark_detected'):
                ai_prob = detection_result['ai_probability']
                wm_confidence = watermark_result.get('confidence', 0)
                # Watermarks strongly indicate AI generation
                adjustment = min(30, wm_confidence * 0.4)
                detection_result['ai_probability'] = min(100, round(ai_prob + adjustment, 2))
                detection_result['watermark_adjustment'] = round(adjustment, 2)
        
        # Add advanced noise analysis (always run for forensics)
        noise_result = noise_analyzer.analyze(image_bytes)
        if noise_result.get('success'):
            # Add noise map to top-level response for UI
            detection_result['noise_map'] = noise_result.get('noise_map')
            # Update analysis_scores with noise consistency from advanced analyzer
            if 'analysis_scores' in detection_result:
                detection_result['analysis_scores']['noise_consistency'] = noise_result.get('noise_consistency', 
                    detection_result['analysis_scores'].get('noise_consistency', 50))
            # Store full noise details for forensics tab (frontend expects 'noise_analysis')
            detection_result['noise_analysis'] = {
                'consistency': noise_result.get('noise_consistency', 50),
                'low_freq': noise_result.get('low_freq', 'N/A'),
                'mid_freq': noise_result.get('mid_freq', 'N/A'),
                'high_freq': noise_result.get('high_freq', 'N/A'),
                'pattern_analysis': noise_result.get('pattern_analysis', {})
            }
        
        # Add C2PA Content Credentials detection if requested
        if include_c2pa:
            c2pa_result = content_credentials_detector.analyze(image_bytes, filename)
            detection_result['content_credentials'] = c2pa_result
            
            # If C2PA confirms AI generation, adjust probability
            if c2pa_result.get('is_ai_generated'):
                detection_result['ai_probability'] = 95.0  # C2PA is authoritative
                detection_result['ai_generator'] = c2pa_result.get('ai_generator')
                detection_result['verdict'] = 'AI_GENERATED'
                detection_result['verdict_description'] = f"Confirmed AI-generated by {c2pa_result.get('ai_generator')} (verified via Content Credentials)"
        
        # Add AI-powered visual analysis and explanation (Groq Vision - Llama 4 Scout)
        if include_ai_analysis:
            ai_analysis = image_explainer.analyze_image(image_bytes, detection_result)
            detection_result['ai_analysis'] = ai_analysis
            
            # Update verdict with combined analysis if available
            if ai_analysis.get('success') and ai_analysis.get('combined_verdict'):
                combined = ai_analysis['combined_verdict']
                # Only override if not already confirmed by C2PA
                if not detection_result.get('content_credentials', {}).get('is_ai_generated'):
                    detection_result['ai_probability'] = combined['combined_probability']
                    detection_result['verdict'] = combined['verdict']
                    detection_result['verdict_description'] = combined['verdict_description']
        
        return jsonify(detection_result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-image/ensemble', methods=['POST'])
@limiter.limit("5 per minute")
def detect_ai_image_ensemble():
    """
    Advanced AI image detection using multi-model ensemble.
    
    Uses multiple detection methods with weighted score fusion:
    - Statistical analysis (frequency, noise, texture)
    - NYUAD ViT Detector (Vision Transformer - 97.36% accuracy)
    - UniversalFakeDetect (CLIP-based - generalizes across generators)
    - Frequency domain analysis (GAN fingerprints)
    - Watermark detection (10+ methods including Meta Stable Signature)
    - C2PA Content Credentials
    - Deepfake detection (for images with faces)
    
    Request body:
        {
            "image": "base64_encoded_image_data",
            "filename": "optional_filename.jpg",
            "load_ml_models": true  // optional, loads heavy ML models
        }
    
    Response:
        {
            "success": true,
            "ensemble_verdict": "AI_GENERATED|LIKELY_AI|POSSIBLY_AI|UNCERTAIN|POSSIBLY_REAL|LIKELY_REAL|REAL",
            "ai_probability": 75.5,
            "confidence": 85,
            "verdict_description": "...",
            "individual_results": {
                "statistical": {...},
                "nyuad": {...},
                "clip": {...},
                "frequency": {...},
                "watermark": {...},
                "c2pa": {...}
            },
            "score_breakdown": {...},
            "detection_agreement": {...},
            "overrides_applied": [...],
            "recommendations": [...]
        }
    """
    try:
        global ensemble_detector
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image" field in request body',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        image_data = data['image']
        filename = data.get('filename', 'uploaded_image')
        load_ml_models = data.get('load_ml_models', True)
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Check file size
        max_image_size = 50 * 1024 * 1024
        if len(image_bytes) > max_image_size:
            return jsonify({
                'success': False,
                'error': f'Image too large (max {max_image_size // 1024 // 1024}MB)',
                'error_code': 'IMAGE_TOO_LARGE'
            }), 400
        
        # Initialize ensemble detector with ML models if requested
        if ensemble_detector is None or (load_ml_models and not hasattr(ensemble_detector, '_ml_loaded')):
            try:
                from image_detector import EnsembleDetector
                ensemble_detector = EnsembleDetector(use_gpu=False, load_ml_models=load_ml_models)
                if load_ml_models:
                    ensemble_detector._ml_loaded = True
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Could not initialize ensemble detector: {str(e)}',
                    'error_code': 'DETECTOR_INIT_ERROR'
                }), 500
        
        # Run ensemble detection
        result = ensemble_detector.detect(image_bytes, filename)
        
        # Add AI-powered visual analysis for explanation
        if image_explainer.client:
            ai_analysis = image_explainer.analyze_image(image_bytes, result)
            result['ai_analysis'] = ai_analysis
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-image/fast', methods=['POST'])
@limiter.limit("20 per minute")
def detect_ai_image_fast():
    """
    Fast AI image detection using cascading 3-stage detection.
    
    3-5x faster than full ensemble for clear cases:
    - Stage 1: Quick statistical/frequency analysis (~20ms)
    - Stage 2: Single fast ML model (~100ms)  
    - Stage 3: Full ensemble (only for uncertain cases)
    
    Request body:
        {
            "image": "base64_encoded_image_data",
            "filename": "optional_filename.jpg"
        }
    
    Response:
        {
            "success": true,
            "ai_probability": 75.5,
            "verdict": "LIKELY_AI",
            "verdict_description": "...",
            "cascade_stage": 2,
            "stages_run": ["statistical", "fast_ml"],
            "timing_ms": {"stage1": 25.3, "stage2": 98.1, "total": 123.4},
            "performance": {"early_exit": true, "speedup_factor": 3.0}
        }
    """
    try:
        global fast_detector
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image" field in request body',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        image_data = data['image']
        filename = data.get('filename', 'uploaded_image')
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Check file size
        max_image_size = 50 * 1024 * 1024
        if len(image_bytes) > max_image_size:
            return jsonify({
                'success': False,
                'error': f'Image too large (max {max_image_size // 1024 // 1024}MB)',
                'error_code': 'IMAGE_TOO_LARGE'
            }), 400
        
        # Initialize fast detector if needed
        if fast_detector is None:
            from image_detector import FastCascadeDetector
            fast_detector = FastCascadeDetector(use_gpu=False, enable_fp16=False)
        
        # Run fast cascading detection
        result = fast_detector.detect(image_bytes, filename)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-image/ela', methods=['POST'])
@limiter.limit("10 per minute")
def get_ela_heatmap():
    """
    Generate Error Level Analysis heatmap for an image.
    
    Request body:
        {
            "image": "base64_encoded_image_data",
            "colormap": "hot"  // optional: "hot", "jet", "viridis"
        }
    
    Response:
        {
            "success": true,
            "ela_heatmap": "base64_encoded_heatmap",
            "manipulation_likelihood": 45.5,
            "analysis": {...}
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image" field',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        image_data = data['image']
        colormap = data.get('colormap', 'hot')
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Generate ELA analysis
        ela_result = ela_analyzer.analyze(image_bytes)
        
        # Generate colored heatmap
        heatmap = ela_analyzer.generate_heatmap(image_bytes, colormap)
        
        return jsonify({
            'success': True,
            'ela_heatmap': heatmap,
            'ela_raw': ela_result.get('ela_image'),
            'manipulation_likelihood': ela_result.get('manipulation_likelihood', 0),
            'suspicious_regions': ela_result.get('suspicious_regions', []),
            'analysis': ela_result.get('analysis', {})
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-image/metadata', methods=['POST'])
@limiter.limit("20 per minute")
def get_image_metadata():
    """
    Extract and analyze image metadata (EXIF).
    
    Request body:
        {
            "image": "base64_encoded_image_data"
        }
    
    Response:
        {
            "success": true,
            "has_exif": true,
            "has_camera_info": true,
            "camera_make": "Canon",
            "camera_model": "EOS 5D Mark IV",
            "anomalies": [...],
            "metadata": {...}
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image" field',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        image_data = data['image']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Analyze metadata
        result = metadata_analyzer.analyze(image_bytes)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-image/watermark', methods=['POST'])
@limiter.limit("10 per minute")
def detect_image_watermark():
    """
    Detect invisible watermarks in an image.
    
    Detects watermarks from:
    - Stable Diffusion (DWT-DCT method)
    - Custom invisible-watermark implementations
    - Spectral domain watermarks
    - LSB steganographic watermarks
    
    Request body:
        {
            "image": "base64_encoded_image_data"
        }
    
    Response:
        {
            "success": true,
            "watermark_detected": true,
            "watermark_type": "DWT-DCT",
            "ai_generator_signature": "stable_diffusion",
            "confidence": 75,
            "detection_methods": {...},
            "details": [...]
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image" field',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        image_data = data['image']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Detect watermarks
        result = watermark_detector.analyze(image_bytes)
        result['success'] = True
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR'
        }), 500


@app.route('/api/detect-image/c2pa', methods=['POST'])
@limiter.limit("10 per minute")
def detect_content_credentials():
    """
    Detect C2PA Content Credentials in an image.
    
    C2PA (Coalition for Content Provenance and Authenticity) is used by:
    - OpenAI DALL-E 3
    - Adobe Firefly
    - Microsoft Designer
    - Google Imagen
    
    Request body:
        {
            "image": "base64_encoded_image_data",
            "filename": "optional_filename.jpg"
        }
    
    Response:
        {
            "success": true,
            "c2pa_found": true,
            "is_ai_generated": true,
            "ai_generator": "DALL-E 3",
            "signature_valid": true,
            "provenance_chain": [...],
            "trust_indicators": {...}
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "image" field',
                'error_code': 'MISSING_IMAGE'
            }), 400
        
        image_data = data['image']
        filename = data.get('filename', 'image')
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid base64 image data: {str(e)}',
                'error_code': 'INVALID_IMAGE'
            }), 400
        
        # Detect Content Credentials
        result = content_credentials_detector.analyze(image_bytes, filename)
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


@app.route('/api/fact-check/feedback/stats', methods=['GET'])
def feedback_stats():
    """Get feedback statistics."""
    try:
        stats = feedback_handler.get_feedback_stats()
        return jsonify({
            'success': True,
            **stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error retrieving feedback stats: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("Starting VisioNova Fact-Check API Server...")
    print("API available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /                    - Health check")
    print("  POST /api/fact-check      - Check a claim/URL")
    print("  GET  /api/fact-check?q=   - Check a claim (simple)")
    print("  POST /api/fact-check/feedback - Submit user feedback")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
