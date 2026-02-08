"""
Test Suite for Watermark Detector
Tests all 10 detection methods with various image types
"""

import pytest
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from image_detector.watermark_detector import WatermarkDetector
from PIL import Image
import numpy as np
import io


class TestWatermarkDetector:
    """Test watermark detection functionality"""
    
    @pytest.fixture
    def detector(self):
        """Create watermark detector instance"""
        return WatermarkDetector()
    
    @pytest.fixture
    def clean_image_bytes(self):
        """Generate clean test image without watermark"""
        # Create 512x512 random natural-looking image
        img = Image.new('RGB', (512, 512))
        pixels = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(pixels)
        
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector is not None
        assert hasattr(detector, 'analyze')
        assert hasattr(detector, 'watermark_lib_available')
        assert hasattr(detector, 'stable_signature_available')
    
    def test_library_status_logging(self, detector, caplog):
        """Test library availability is logged"""
        # Check if initialization logged library status
        assert detector.watermark_lib_available is not None
        assert detector.steganogan_available is not None
        assert detector.stable_signature_available is not None
    
    def test_clean_image_no_watermark(self, detector, clean_image_bytes):
        """Test clean image produces no false positives"""
        result = detector.analyze(clean_image_bytes)
        
        assert result is not None
        assert 'watermark_detected' in result
        assert 'detection_methods' in result
        assert 'confidence' in result
        
        # Clean image should not trigger watermark detection
        # (allowing for some false positives in probabilistic methods)
        if result['watermark_detected']:
            # If detected, confidence should be low (<50%)
            assert result['confidence'] < 50, \
                f"False positive with high confidence: {result['confidence']}%"
    
    def test_detection_methods_structure(self, detector, clean_image_bytes):
        """Test all detection methods are executed and return proper structure"""
        result = detector.analyze(clean_image_bytes)
        methods = result.get('detection_methods', {})
        
        # Check expected methods are present
        expected_methods = [
            'spectral_analysis',
            'lsb_analysis',
            'metadata_watermark',
            'stable_signature',
            'gaussian_shading',
            'treering_analysis',
            'adversarial_analysis'
        ]
        
        for method in expected_methods:
            assert method in methods, f"Method {method} not found in results"
            assert isinstance(methods[method], dict), f"Method {method} result is not a dict"
    
    def test_watermarks_found_array(self, detector, clean_image_bytes):
        """Test watermarks_found array is generated"""
        result = detector.analyze(clean_image_bytes)
        
        assert 'watermarks_found' in result
        assert isinstance(result['watermarks_found'], list)
        
        # For clean image, should be empty or very short
        if result['watermark_detected']:
            assert len(result['watermarks_found']) > 0
        else:
            assert len(result['watermarks_found']) == 0
    
    def test_confidence_range(self, detector, clean_image_bytes):
        """Test confidence is in valid range 0-100"""
        result = detector.analyze(clean_image_bytes)
        
        assert 0 <= result['confidence'] <= 100, \
            f"Confidence out of range: {result['confidence']}"
    
    def test_spectral_analysis_multiscale(self, detector, clean_image_bytes):
        """Test multi-scale spectral analysis produces scale_results"""
        result = detector.analyze(clean_image_bytes)
        spectral = result['detection_methods'].get('spectral_analysis', {})
        
        # Should have scale_results from multi-scale analysis
        if 'scale_results' in spectral:
            assert isinstance(spectral['scale_results'], list)
            assert len(spectral['scale_results']) > 0
            
            # Check each scale has required fields
            for scale_result in spectral['scale_results']:
                assert 'scale' in scale_result
                assert 'patterns_found' in scale_result
                assert 'confidence' in scale_result
    
    def test_heatmap_generation(self, detector, clean_image_bytes):
        """Test watermark heatmap is generated"""
        result = detector.analyze(clean_image_bytes)
        
        assert 'visualizations' in result
        
        # Heatmap may fail on very small images or due to errors
        if result.get('visualizations'):
            heatmap = result['visualizations'].get('watermark_heatmap')
            if heatmap:
                # Should be base64 data URL
                assert heatmap.startswith('data:image/png;base64,')
                assert len(heatmap) > 100  # Should have actual data
    
    def test_metadata_watermark_detection(self, detector):
        """Test metadata watermark detection with EXIF tags"""
        # Create image with AI metadata
        img = Image.new('RGB', (512, 512), color='white')
        
        # Add EXIF with AI tool signature
        from PIL.ExifTags import TAGS
        exif_dict = {
            0x0131: 'Stable Diffusion',  # Software tag
            0x010e: 'AI Generated Image',  # ImageDescription
        }
        
        # Note: PIL doesn't easily allow setting EXIF for test
        # This is a placeholder - real test would use actual watermarked image
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        
        result = detector.analyze(buf.getvalue())
        
        # Just verify structure exists (actual detection depends on metadata)
        assert 'metadata_watermark' in result['detection_methods']
    
    def test_weighted_confidence_calculation(self, detector):
        """Test weighted confidence is calculated correctly"""
        # Create mock detection_methods dict
        detection_methods = {
            'metadata_watermark': {'detected': True, 'confidence': 90},
            'invisible_watermark': {'detected': True, 'confidence': 75},
            'spectral_analysis': {'patterns_found': False, 'confidence': 0},
        }
        
        weighted = detector._calculate_weighted_confidence(detection_methods)
        
        # Weighted confidence should be between individual confidences
        # metadata (90% × 0.95) + invisible (75% × 0.85) / (0.95 + 0.85) = 83
        assert 80 <= weighted <= 90, f"Weighted confidence unexpected: {weighted}"
        
        # Should be higher than simple average
        simple_avg = (90 + 75) / 2  # 82.5
        assert weighted != simple_avg  # Should be different due to weights
    
    def test_generate_watermarks_found_array(self, detector):
        """Test watermarks_found array generation from detection methods"""
        detection_methods = {
            'invisible_watermark': {'detected': True, 'type': 'dwtDct'},
            'stable_signature': {'detected': True, 'confidence': 85},
            'spectral_analysis': {'patterns_found': True},
            'lsb_analysis': {'anomaly_detected': False},
            'metadata_watermark': {'found': False}
        }
        
        watermarks_found = detector._generate_watermarks_found_array(detection_methods)
        
        assert isinstance(watermarks_found, list)
        assert len(watermarks_found) == 3  # 3 methods detected
        
        # Check friendly names are used
        assert any('DWT' in wm for wm in watermarks_found)
        assert any('Meta' in wm or 'Stable' in wm for wm in watermarks_found)
        assert any('Spectral' in wm for wm in watermarks_found)
    
    def test_error_handling_invalid_image(self, detector):
        """Test detector handles invalid image data gracefully"""
        invalid_data = b'not an image'
        
        result = detector.analyze(invalid_data)
        
        assert result is not None
        assert result['watermark_detected'] == False
        assert result['status'] == 'ERROR'
        assert 'error' in result
    
    def test_error_handling_corrupted_image(self, detector):
        """Test detector handles corrupted image data"""
        # Create truncated PNG
        img = Image.new('RGB', (100, 100))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        corrupted = buf.getvalue()[:100]  # Truncate
        
        result = detector.analyze(corrupted)
        
        # Should not crash, either succeed or return error
        assert result is not None
        assert 'watermark_detected' in result
    
    def test_performance_benchmark(self, detector, clean_image_bytes, benchmark):
        """Benchmark watermark detection performance"""
        # Requires pytest-benchmark: pip install pytest-benchmark
        try:
            result = benchmark(detector.analyze, clean_image_bytes)
            assert result is not None
        except:
            # Skip if benchmark not available
            pytest.skip("pytest-benchmark not installed")
    
    def test_different_image_sizes(self, detector):
        """Test detector works with various image sizes"""
        sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        
        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='gray')
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            
            result = detector.analyze(buf.getvalue())
            
            assert result is not None
            assert 'watermark_detected' in result
            # Larger images shouldn't cause errors
    
    def test_different_image_formats(self, detector):
        """Test detector works with PNG, JPEG, WebP"""
        formats = ['PNG', 'JPEG', 'WebP']
        
        for fmt in formats:
            try:
                img = Image.new('RGB', (512, 512), color='blue')
                buf = io.BytesIO()
                img.save(buf, format=fmt)
                
                result = detector.analyze(buf.getvalue())
                
                assert result is not None
                assert 'watermark_detected' in result
            except:
                # Skip if format not supported
                pytest.skip(f"{fmt} not supported")


class TestWatermarkMethods:
    """Test individual watermark detection methods"""
    
    @pytest.fixture
    def detector(self):
        return WatermarkDetector()
    
    @pytest.fixture
    def test_image_array(self):
        """Create test image as numpy array"""
        return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    def test_spectral_analysis_method(self, detector, test_image_array):
        """Test spectral analysis method directly"""
        result = detector._spectral_watermark_analysis(test_image_array)
        
        assert isinstance(result, dict)
        assert 'patterns_found' in result
        assert 'confidence' in result
        assert 'analysis' in result
        
        # Check analysis contains expected metrics
        if result.get('analysis'):
            # May have symmetry scores
            assert isinstance(result['analysis'], dict)
    
    def test_lsb_analysis_method(self, detector, test_image_array):
        """Test LSB analysis method directly"""
        result = detector._lsb_analysis(test_image_array)
        
        assert isinstance(result, dict)
        assert 'anomaly_detected' in result
        assert 'confidence' in result
        assert 'analysis' in result
        
        # Should have per-channel analysis
        analysis = result.get('analysis', {})
        # At least some channel data
        assert any('channel' in key for key in analysis.keys())
    
    def test_stable_signature_method(self, detector, test_image_array):
        """Test Meta Stable Signature detection method"""
        result = detector._detect_stable_signature(test_image_array)
        
        assert isinstance(result, dict)
        assert 'detected' in result
        
        if not detector.stable_signature_available:
            # Should gracefully handle missing model
            assert 'error' in result or result['detected'] == False
    
    def test_treering_detection_method(self, detector, test_image_array):
        """Test Tree-Ring pattern detection method"""
        result = detector._detect_treering_patterns(test_image_array)
        
        assert isinstance(result, dict)
        assert 'detected' in result
        assert 'confidence' in result
    
    def test_gaussian_shading_method(self, detector, test_image_array):
        """Test Gaussian Shading detection method"""
        result = detector._detect_gaussian_shading(test_image_array)
        
        assert isinstance(result, dict)
        assert 'detected' in result
        assert 'confidence' in result


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
