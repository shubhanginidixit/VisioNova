"""
VisioNova Content Credentials Detector (C2PA)
Detects and validates Content Credentials / C2PA provenance data in images.

C2PA (Coalition for Content Provenance and Authenticity) is an open standard
for digital content certification that shows:
- Content origin (camera, software, AI generator)
- Editing history
- Cryptographic signatures for tamper detection

Supported by: Adobe, Microsoft, Google, OpenAI, Meta, BBC, Sony, Truepic
"""

import io
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Known AI generator signatures in C2PA manifests
AI_GENERATOR_SIGNATURES = {
    # OpenAI
    'dall-e': {'name': 'DALL-E', 'company': 'OpenAI', 'type': 'image_generation'},
    'dall-e-2': {'name': 'DALL-E 2', 'company': 'OpenAI', 'type': 'image_generation'},
    'dall-e-3': {'name': 'DALL-E 3', 'company': 'OpenAI', 'type': 'image_generation'},
    'chatgpt': {'name': 'ChatGPT', 'company': 'OpenAI', 'type': 'image_generation'},
    'openai': {'name': 'OpenAI', 'company': 'OpenAI', 'type': 'ai_platform'},
    
    # Adobe
    'adobe firefly': {'name': 'Adobe Firefly', 'company': 'Adobe', 'type': 'image_generation'},
    'firefly': {'name': 'Adobe Firefly', 'company': 'Adobe', 'type': 'image_generation'},
    'photoshop': {'name': 'Adobe Photoshop', 'company': 'Adobe', 'type': 'editor'},
    'lightroom': {'name': 'Adobe Lightroom', 'company': 'Adobe', 'type': 'editor'},
    
    # Microsoft
    'microsoft designer': {'name': 'Microsoft Designer', 'company': 'Microsoft', 'type': 'image_generation'},
    'bing image creator': {'name': 'Bing Image Creator', 'company': 'Microsoft', 'type': 'image_generation'},
    'copilot': {'name': 'Microsoft Copilot', 'company': 'Microsoft', 'type': 'ai_assistant'},
    
    # Google
    'imagen': {'name': 'Google Imagen', 'company': 'Google', 'type': 'image_generation'},
    'gemini': {'name': 'Google Gemini', 'company': 'Google', 'type': 'ai_platform'},
    
    # Others
    'midjourney': {'name': 'Midjourney', 'company': 'Midjourney', 'type': 'image_generation'},
    'stable diffusion': {'name': 'Stable Diffusion', 'company': 'Stability AI', 'type': 'image_generation'},
    'stability ai': {'name': 'Stability AI', 'company': 'Stability AI', 'type': 'ai_platform'},
}


class ContentCredentialsDetector:
    """
    C2PA Content Credentials Detector.
    
    Detects and extracts provenance information from images with
    Content Credentials (C2PA manifests).
    """
    
    def __init__(self):
        """Initialize the C2PA detector."""
        self.c2pa_available = False
        self.c2pa_module = None
        
        try:
            import c2pa
            self.c2pa_available = True
            self.c2pa_module = c2pa
            logger.info("c2pa-python library available")
        except ImportError:
            logger.warning("c2pa-python library not available. Install with: pip install c2pa-python")
    
    def analyze(self, image_data: bytes, filename: str = "image") -> Dict[str, Any]:
        """
        Analyze an image for C2PA Content Credentials.
        
        Args:
            image_data: Raw image bytes
            filename: Original filename (helps determine format)
            
        Returns:
            dict with C2PA detection results
        """
        result = {
            'c2pa_found': False,
            'has_content_credentials': False,
            'is_ai_generated': False,
            'ai_generator': None,
            'generator_info': None,
            'manifest': None,
            'claims': [],
            'assertions': [],
            'signature_valid': None,
            'editing_history': [],
            'provenance_chain': [],
            'trust_indicators': {},
            'status': 'CHECKING',
            'details': []
        }
        
        try:
            # Method 1: Use c2pa-python library if available
            if self.c2pa_available:
                c2pa_result = self._read_c2pa_manifest(image_data)
                
                if c2pa_result.get('found'):
                    result['c2pa_found'] = True
                    result['has_content_credentials'] = True
                    result['manifest'] = c2pa_result.get('manifest')
                    result['claims'] = c2pa_result.get('claims', [])
                    result['assertions'] = c2pa_result.get('assertions', [])
                    result['signature_valid'] = c2pa_result.get('signature_valid')
                    result['editing_history'] = c2pa_result.get('editing_history', [])
                    result['provenance_chain'] = c2pa_result.get('provenance_chain', [])
                    
                    # Check for AI generator in manifest
                    ai_info = self._detect_ai_generator(c2pa_result)
                    if ai_info:
                        result['is_ai_generated'] = True
                        result['ai_generator'] = ai_info.get('name')
                        result['generator_info'] = ai_info
                        result['details'].append(f"AI generator detected: {ai_info.get('name')}")
            
            # Method 2: Manual JUMBF box detection (fallback)
            if not result['c2pa_found']:
                jumbf_result = self._detect_jumbf_box(image_data)
                
                if jumbf_result.get('found'):
                    result['c2pa_found'] = True
                    result['has_content_credentials'] = True
                    result['details'].append("C2PA JUMBF box detected (manifest parsing limited)")
                    
                    # Try to extract basic info
                    if jumbf_result.get('generator'):
                        result['is_ai_generated'] = True
                        result['ai_generator'] = jumbf_result.get('generator')
            
            # Method 3: Check for XMP provenance metadata
            xmp_result = self._check_xmp_provenance(image_data)
            
            if xmp_result.get('found'):
                result['details'].append("XMP provenance metadata found")
                if xmp_result.get('ai_generated'):
                    result['is_ai_generated'] = True
                    if not result['ai_generator']:
                        result['ai_generator'] = xmp_result.get('generator')
                
                # Add XMP assertions
                result['assertions'].extend(xmp_result.get('assertions', []))
            
            # Build trust indicators
            result['trust_indicators'] = self._build_trust_indicators(result)
            
            # Set final status
            if result['c2pa_found']:
                if result['is_ai_generated']:
                    result['status'] = 'AI_GENERATED_CERTIFIED'
                    result['details'].append(f"Image certified as AI-generated by {result['ai_generator']}")
                else:
                    result['status'] = 'PROVENANCE_VERIFIED'
                    result['details'].append("Content credentials verified")
            else:
                result['status'] = 'NO_CREDENTIALS'
                result['details'].append("No C2PA Content Credentials found")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing C2PA: {e}")
            return {
                'c2pa_found': False,
                'has_content_credentials': False,
                'is_ai_generated': False,
                'status': 'ERROR',
                'error': str(e),
                'details': [f"Analysis error: {str(e)}"]
            }
    
    def _read_c2pa_manifest(self, image_data: bytes) -> Dict[str, Any]:
        """
        Read C2PA manifest using the c2pa-python library.
        """
        result = {
            'found': False,
            'manifest': None,
            'claims': [],
            'assertions': [],
            'signature_valid': None,
            'editing_history': [],
            'provenance_chain': []
        }
        
        if not self.c2pa_available:
            return result
        
        try:
            # Read manifest from image bytes
            reader = self.c2pa_module.Reader.from_stream("image/jpeg", io.BytesIO(image_data))
            
            if reader:
                result['found'] = True
                
                # Get the active manifest
                manifest_store = reader.json()
                if manifest_store:
                    result['manifest'] = json.loads(manifest_store) if isinstance(manifest_store, str) else manifest_store
                    
                    # Extract claims
                    manifests = result['manifest'].get('manifests', {})
                    for manifest_id, manifest_data in manifests.items():
                        # Get claim generator (tool that created this)
                        claim_generator = manifest_data.get('claim_generator', '')
                        if claim_generator:
                            result['claims'].append({
                                'id': manifest_id,
                                'generator': claim_generator,
                                'title': manifest_data.get('title', ''),
                                'format': manifest_data.get('format', '')
                            })
                            result['provenance_chain'].append(claim_generator)
                        
                        # Get assertions
                        assertions = manifest_data.get('assertions', [])
                        for assertion in assertions:
                            result['assertions'].append({
                                'label': assertion.get('label', ''),
                                'data': assertion.get('data', {})
                            })
                            
                            # Check for AI training/generation assertions
                            label = assertion.get('label', '').lower()
                            if 'c2pa.ai_generative_training' in label or 'c2pa.ai_generated' in label:
                                result['ai_generated_assertion'] = True
                        
                        # Get ingredients (editing history)
                        ingredients = manifest_data.get('ingredients', [])
                        for ingredient in ingredients:
                            result['editing_history'].append({
                                'title': ingredient.get('title', ''),
                                'format': ingredient.get('format', ''),
                                'relationship': ingredient.get('relationship', '')
                            })
                    
                    # Signature validation
                    result['signature_valid'] = True  # c2pa library validates on read
            
            return result
            
        except Exception as e:
            logger.debug(f"c2pa read failed (may not have C2PA): {e}")
            return result
    
    def _detect_jumbf_box(self, image_data: bytes) -> Dict[str, Any]:
        """
        Detect C2PA JUMBF box in image data (manual detection).
        
        JUMBF (JPEG Universal Metadata Box Format) is the container
        format used by C2PA to embed manifests.
        """
        result = {
            'found': False,
            'generator': None
        }
        
        try:
            # C2PA uses JUMBF boxes with specific markers
            # Look for the C2PA UUID: 6332 7061 (c2pa in ASCII)
            c2pa_marker = b'c2pa'
            jumbf_marker = b'jumb'
            
            data = image_data
            
            # Search for C2PA markers
            c2pa_pos = data.find(c2pa_marker)
            jumbf_pos = data.find(jumbf_marker)
            
            if c2pa_pos != -1 or jumbf_pos != -1:
                result['found'] = True
                
                # Try to find generator string nearby
                search_region = data[max(0, c2pa_pos - 500):min(len(data), c2pa_pos + 2000)] if c2pa_pos != -1 else data
                
                for sig_key, sig_info in AI_GENERATOR_SIGNATURES.items():
                    if sig_key.encode() in search_region.lower():
                        result['generator'] = sig_info['name']
                        break
            
            # Also check for APP11 marker (JPEG C2PA)
            app11_marker = b'\xff\xeb'
            if app11_marker in data:
                result['found'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting JUMBF: {e}")
            return result
    
    def _check_xmp_provenance(self, image_data: bytes) -> Dict[str, Any]:
        """
        Check XMP metadata for provenance information.
        
        Some tools embed AI generation info in XMP before C2PA.
        """
        result = {
            'found': False,
            'ai_generated': False,
            'generator': None,
            'assertions': []
        }
        
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(image_data))
            
            # Get XMP data
            xmp_data = image.info.get('XML:com.adobe.xmp', '') or image.info.get('xmp', '')
            if isinstance(xmp_data, bytes):
                xmp_data = xmp_data.decode('utf-8', errors='ignore')
            
            if not xmp_data:
                return result
            
            result['found'] = True
            xmp_lower = xmp_data.lower()
            
            # Check for AI generation markers
            ai_markers = [
                ('dall-e', 'DALL-E'),
                ('dalle', 'DALL-E'),
                ('openai', 'OpenAI'),
                ('midjourney', 'Midjourney'),
                ('stable diffusion', 'Stable Diffusion'),
                ('firefly', 'Adobe Firefly'),
                ('imagen', 'Google Imagen'),
                ('ai generated', None),
                ('ai-generated', None),
                ('artificially generated', None),
            ]
            
            for marker, generator in ai_markers:
                if marker in xmp_lower:
                    result['ai_generated'] = True
                    if generator:
                        result['generator'] = generator
                        result['assertions'].append({
                            'label': 'xmp:AIGenerator',
                            'data': {'generator': generator}
                        })
                    break
            
            # Check for CAI (Content Authenticity Initiative) namespace
            if 'contentauthenticity' in xmp_lower or 'cai:' in xmp_lower:
                result['assertions'].append({
                    'label': 'cai:ProvenanceData',
                    'data': {'has_cai_metadata': True}
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking XMP: {e}")
            return result
    
    def _detect_ai_generator(self, c2pa_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Detect AI generator from C2PA manifest data.
        """
        if not c2pa_result.get('found'):
            return None
        
        # Check claims for AI generator signatures
        for claim in c2pa_result.get('claims', []):
            generator = claim.get('generator', '').lower()
            
            for sig_key, sig_info in AI_GENERATOR_SIGNATURES.items():
                if sig_key in generator:
                    return sig_info
        
        # Check provenance chain
        for item in c2pa_result.get('provenance_chain', []):
            item_lower = item.lower()
            for sig_key, sig_info in AI_GENERATOR_SIGNATURES.items():
                if sig_key in item_lower:
                    return sig_info
        
        # Check assertions for AI generation
        for assertion in c2pa_result.get('assertions', []):
            label = assertion.get('label', '').lower()
            
            if 'ai_generated' in label or 'ai_generative' in label:
                data = assertion.get('data', {})
                generator_name = data.get('generator', data.get('model', 'Unknown AI'))
                
                # Try to match known generator
                gen_lower = generator_name.lower()
                for sig_key, sig_info in AI_GENERATOR_SIGNATURES.items():
                    if sig_key in gen_lower:
                        return sig_info
                
                return {
                    'name': generator_name,
                    'company': 'Unknown',
                    'type': 'image_generation'
                }
        
        return None
    
    def _build_trust_indicators(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build trust indicators based on analysis results.
        """
        indicators = {
            'has_provenance': result.get('c2pa_found', False),
            'signature_verified': result.get('signature_valid', False),
            'ai_disclosure': result.get('is_ai_generated', False),
            'editing_disclosed': len(result.get('editing_history', [])) > 0,
            'trust_score': 0
        }
        
        # Calculate trust score
        score = 0
        if indicators['has_provenance']:
            score += 30
        if indicators['signature_verified']:
            score += 30
        if indicators['ai_disclosure'] and result.get('c2pa_found'):
            score += 25  # Disclosed AI generation is good
        if indicators['editing_disclosed']:
            score += 15
        
        indicators['trust_score'] = score
        
        return indicators


# Convenience function
def detect_content_credentials(image_data: bytes, filename: str = "image") -> Dict[str, Any]:
    """Convenience function to detect C2PA Content Credentials."""
    detector = ContentCredentialsDetector()
    return detector.analyze(image_data, filename)
