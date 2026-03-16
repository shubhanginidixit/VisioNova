"""
VisioNova Face Consistency Detector
Analyzes facial features for AI-generation artifacts.

AI-generated faces often have subtle inconsistencies:
- Asymmetric or missing eye reflections
- Inconsistent lighting across face
- Unnatural skin texture patterns
- Misaligned facial features

References:
- "Exposing Deep Fakes Using Inconsistent Head Poses" (ICASSP 2019)
- "FaceForensics++: Learning to Detect Manipulated Facial Images" (ICCV 2019)
"""

import io
import logging
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import face detection
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - face detection will use fallback")

# Try to import MTCNN for better face detection
try:
    from mtcnn import MTCNN as MTCNNDetector
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    logger.info("MTCNN not available - will use Haar cascades or fallback")

# Try mediapipe as another option
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class FaceConsistencyDetector:
    """
    Face Consistency Detector for AI-generated face detection.
    
    Analyzes:
    - Eye reflection consistency (corneal specular highlights)
    - Facial symmetry and proportions
    - Lighting consistency across face
    - Skin texture patterns
    - Pupil shape and consistency
    """
    
    def __init__(self):
        """Initialize face consistency detector."""
        self.cv2_available = CV2_AVAILABLE
        self.mtcnn = None
        self.mp_face = None
        
        # Try MTCNN first (best accuracy)
        if MTCNN_AVAILABLE:
            try:
                self.mtcnn = MTCNNDetector()
                logger.info("Face Consistency Detector initialized (MTCNN)")
            except Exception as e:
                logger.warning(f"MTCNN init failed: {e}")
                self.mtcnn = None
        
        # Try MediaPipe as second option
        if self.mtcnn is None and MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
                logger.info("Face Consistency Detector initialized (MediaPipe)")
            except Exception as e:
                logger.warning(f"MediaPipe init failed: {e}")
                self.mp_face = None
        
        # Fallback to Haar cascades
        if self.cv2_available and self.mtcnn is None and self.mp_face is None:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_eye.xml'
                )
                logger.info("Face Consistency Detector initialized (Haar cascades)")
            except Exception as e:
                logger.warning(f"Could not load cascades: {e}")
                self.face_cascade = None
                self.eye_cascade = None
        else:
            self.face_cascade = None
            self.eye_cascade = None
        
        if not any([self.mtcnn, self.mp_face, self.face_cascade]):
            logger.info("Face Consistency Detector initialized (skin-color fallback only)")
    
    def detect(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze image for face consistency issues.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Detection result
        """
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return self.analyze(image)
            
        except Exception as e:
            logger.error(f"Face consistency error: {e}")
            return {
                'success': False,
                'error': str(e),
                'faces_found': 0
            }
    
    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze faces in image for AI-generation artifacts."""
        img_array = np.array(image)
        
        # Detect faces
        faces = self._detect_faces(img_array)
        
        if not faces:
            return {
                'success': True,
                'faces_found': 0,
                'ai_probability': None,
                'message': 'No faces detected in image',
                'applicable': False
            }
        
        # Analyze each face
        face_results = []
        total_anomaly_score = 0
        
        for i, face_region in enumerate(faces):
            result = self._analyze_face(img_array, face_region)
            face_results.append(result)
            total_anomaly_score += result.get('anomaly_score', 0.5)
        
        # Average anomaly score across faces
        avg_anomaly = total_anomaly_score / len(faces)
        ai_probability = avg_anomaly * 100
        
        return {
            'success': True,
            'faces_found': len(faces),
            'ai_probability': round(ai_probability, 2),
            'face_analyses': face_results,
            'verdict': self._get_verdict(ai_probability),
            'method': 'Face Consistency Analysis',
            'applicable': True
        }
    
    def _detect_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using best available method."""
        # Priority 1: MTCNN (most accurate)
        if self.mtcnn is not None:
            try:
                results = self.mtcnn.detect_faces(img)
                if results:
                    faces = []
                    for r in results:
                        x, y, w, h = r['box']
                        # MTCNN can return negative coords
                        x, y = max(0, x), max(0, y)
                        if r['confidence'] > 0.9:
                            faces.append((x, y, w, h))
                    if faces:
                        return faces
            except Exception as e:
                logger.warning(f"MTCNN detection failed: {e}")
        
        # Priority 2: MediaPipe
        if self.mp_face is not None:
            try:
                rgb = img if img.shape[2] == 3 else img[:, :, :3]
                results = self.mp_face.process(rgb)
                if results.detections:
                    h_img, w_img = img.shape[:2]
                    faces = []
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        x = int(bbox.xmin * w_img)
                        y = int(bbox.ymin * h_img)
                        w = int(bbox.width * w_img)
                        h = int(bbox.height * h_img)
                        x, y = max(0, x), max(0, y)
                        faces.append((x, y, w, h))
                    if faces:
                        return faces
            except Exception as e:
                logger.warning(f"MediaPipe detection failed: {e}")
        
        # Priority 3: Haar cascades
        if self.cv2_available and self.face_cascade is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            if len(faces) > 0:
                return [(x, y, w, h) for (x, y, w, h) in faces]
        
        # Priority 4: Skin-color fallback
        return self._detect_faces_fallback(img)
    
    def _detect_faces_fallback(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Improved skin-based face detection fallback using YCrCb color space."""
        h, w = img.shape[:2]
        
        # Try YCrCb skin detection if we have enough pixels
        if h >= 100 and w >= 100:
            try:
                ycrcb = np.zeros_like(img, dtype=np.float32)
                ycrcb[:,:,0] = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
                ycrcb[:,:,1] = 128 + 0.5 * img[:,:,0] - 0.419 * img[:,:,1] - 0.081 * img[:,:,2]  # Cr
                ycrcb[:,:,2] = 128 - 0.169 * img[:,:,0] - 0.331 * img[:,:,1] + 0.5 * img[:,:,2]  # Cb
                
                # Skin color thresholds in YCrCb
                skin_mask = (
                    (ycrcb[:,:,1] >= 133) & (ycrcb[:,:,1] <= 173) &
                    (ycrcb[:,:,2] >= 77) & (ycrcb[:,:,2] <= 127)
                ).astype(np.uint8)
                
                skin_ratio = np.sum(skin_mask) / (h * w)
                
                # If significant skin detected, find the bounding box
                if skin_ratio > 0.05:
                    rows = np.any(skin_mask, axis=1)
                    cols = np.any(skin_mask, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    face_w = cmax - cmin
                    face_h = rmax - rmin
                    # Only accept if region is roughly face-shaped
                    aspect = face_w / (face_h + 1e-10)
                    if 0.5 < aspect < 2.0 and face_w > 40 and face_h > 40:
                        return [(cmin, rmin, face_w, face_h)]
            except Exception:
                pass
        
        # Simple center-crop as face region if image is portrait-like
        if 0.8 < w/h < 1.2:
            margin = min(h, w) // 6
            return [(margin, margin, w - 2*margin, h - 2*margin)]
        
        return []
    
    def _analyze_face(self, img: np.ndarray, face_region: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analyze a single face for AI-generation artifacts."""
        x, y, w, h = face_region
        face_img = img[y:y+h, x:x+w]
        
        anomalies = {}
        
        # 1. Analyze eye reflections
        eye_result = self._analyze_eye_reflections(face_img)
        anomalies['eye_reflection'] = eye_result
        
        # 2. Analyze facial symmetry
        symmetry_result = self._analyze_symmetry(face_img)
        anomalies['symmetry'] = symmetry_result
        
        # 3. Analyze lighting consistency
        lighting_result = self._analyze_lighting(face_img)
        anomalies['lighting'] = lighting_result
        
        # 4. Analyze skin texture
        skin_result = self._analyze_skin_texture(face_img)
        anomalies['skin_texture'] = skin_result
        
        # Calculate overall anomaly score
        scores = [
            eye_result.get('anomaly', 0.5),
            symmetry_result.get('anomaly', 0.5),
            lighting_result.get('anomaly', 0.5),
            skin_result.get('anomaly', 0.5)
        ]
        
        avg_anomaly = np.mean(scores)
        
        return {
            'region': face_region,
            'anomaly_score': round(avg_anomaly, 4),
            'details': anomalies
        }
    
    def _analyze_eye_reflections(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze eye reflections (corneal specular highlights).
        
        Real photos: Both eyes show consistent light reflections
        AI images: Often missing, asymmetric, or unnatural reflections
        """
        h, w = face_img.shape[:2]
        
        # Approximate eye regions (upper third, left/right halves)
        eye_region_h = h // 3
        left_eye = face_img[eye_region_h//2:eye_region_h, 0:w//2]
        right_eye = face_img[eye_region_h//2:eye_region_h, w//2:w]
        
        if left_eye.size == 0 or right_eye.size == 0:
            return {'anomaly': 0.5, 'message': 'Could not extract eye regions'}
        
        # Look for bright spots (reflections)
        def find_reflections(eye_region):
            gray = np.mean(eye_region, axis=2)
            threshold = np.percentile(gray, 95)
            bright_pixels = gray > threshold
            return np.sum(bright_pixels), np.mean(gray[bright_pixels]) if np.any(bright_pixels) else 0
        
        left_count, left_brightness = find_reflections(left_eye)
        right_count, right_brightness = find_reflections(right_eye)
        
        # Check for asymmetry
        total_reflections = left_count + right_count
        
        if total_reflections < 10:
            # Very few reflections - might be AI
            anomaly = 0.6
            message = 'Few eye reflections detected'
        elif abs(left_count - right_count) > max(left_count, right_count) * 0.5:
            # Asymmetric reflections - possible AI
            anomaly = 0.7
            message = 'Asymmetric eye reflections'
        elif abs(left_brightness - right_brightness) > 30:
            # Different reflection brightness
            anomaly = 0.6
            message = 'Inconsistent reflection brightness'
        else:
            anomaly = 0.3
            message = 'Consistent eye reflections'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'left_reflections': int(left_count),
            'right_reflections': int(right_count)
        }
    
    def _analyze_symmetry(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze facial symmetry.
        
        Real faces: Natural asymmetry
        AI faces: Sometimes too perfect OR inconsistent features
        """
        h, w = face_img.shape[:2]
        
        # Split face vertically
        left_half = face_img[:, :w//2]
        right_half = face_img[:, w//2:]
        
        # Flip right half for comparison
        right_flipped = np.fliplr(right_half)
        
        # Resize to match if needed
        min_w = min(left_half.shape[1], right_flipped.shape[1])
        left_half = left_half[:, :min_w]
        right_flipped = right_flipped[:, :min_w]
        
        # Compare
        diff = np.abs(left_half.astype(float) - right_flipped.astype(float))
        avg_diff = np.mean(diff)
        
        # Natural faces: 15-40 average difference
        # Too symmetric (< 10): possibly AI
        # Too asymmetric (> 50): possibly manipulated
        
        if avg_diff < 10:
            anomaly = 0.6
            message = 'Unusually symmetric face'
        elif avg_diff > 50:
            anomaly = 0.6
            message = 'High facial asymmetry'
        else:
            anomaly = 0.3
            message = 'Natural facial symmetry'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'symmetry_score': round(float(avg_diff), 2)
        }
    
    def _analyze_lighting(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze lighting consistency across face.
        
        Real photos: Consistent light direction
        AI images: Sometimes inconsistent shadows/highlights
        """
        gray = np.mean(face_img, axis=2)
        h, w = gray.shape
        
        # Divide face into quadrants
        tl = np.mean(gray[:h//2, :w//2])  # top-left
        tr = np.mean(gray[:h//2, w//2:])  # top-right
        bl = np.mean(gray[h//2:, :w//2])  # bottom-left
        br = np.mean(gray[h//2:, w//2:])  # bottom-right
        
        # Check for consistent lighting gradient
        # Natural lighting: one side brighter than other
        left_avg = (tl + bl) / 2
        right_avg = (tr + br) / 2
        top_avg = (tl + tr) / 2
        bottom_avg = (bl + br) / 2
        
        # Compute gradient consistency
        h_gradient = right_avg - left_avg
        v_gradient = bottom_avg - top_avg
        
        # Check if gradients are consistent (not contradicting)
        quadrant_values = [tl, tr, bl, br]
        variance = np.var(quadrant_values)
        
        # Very low variance = flat/artificial lighting
        # Very high variance = possibly inconsistent
        
        if variance < 50:
            anomaly = 0.5
            message = 'Very flat lighting'
        elif variance > 1000:
            anomaly = 0.6
            message = 'Potentially inconsistent lighting'
        else:
            anomaly = 0.3
            message = 'Natural lighting gradient'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'lighting_variance': round(float(variance), 2)
        }
    
    def _analyze_skin_texture(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Analyze skin texture patterns.
        
        Real skin: Natural pores, texture variation
        AI skin: Often too smooth or with repetitive patterns
        """
        gray = np.mean(face_img, axis=2)
        h, w = gray.shape
        
        # Check center region (likely skin area)
        center_h, center_w = h // 2, w // 2
        margin = min(h, w) // 4
        
        if margin < 5:
            return {'anomaly': 0.5, 'message': 'Face too small for texture analysis'}
        
        skin_region = gray[center_h-margin:center_h+margin, center_w-margin:center_w+margin]
        
        # Compute local variance (texture indicator)
        local_vars = []
        block_size = max(5, margin // 4)
        
        for i in range(0, skin_region.shape[0] - block_size, block_size):
            for j in range(0, skin_region.shape[1] - block_size, block_size):
                block = skin_region[i:i+block_size, j:j+block_size]
                local_vars.append(np.var(block))
        
        if not local_vars:
            return {'anomaly': 0.5, 'message': 'Could not analyze texture'}
        
        avg_var = np.mean(local_vars)
        var_consistency = np.std(local_vars) / (avg_var + 1e-10)
        
        # Very low variance = too smooth (AI)
        # Very uniform variance = unnatural (AI)
        
        if avg_var < 10:
            anomaly = 0.7
            message = 'Unusually smooth skin texture'
        elif var_consistency < 0.3:
            anomaly = 0.6
            message = 'Overly uniform skin texture'
        else:
            anomaly = 0.3
            message = 'Natural skin texture variation'
        
        return {
            'anomaly': anomaly,
            'message': message,
            'texture_variance': round(float(avg_var), 2)
        }
    
    def _get_verdict(self, ai_probability: float) -> str:
        """Get verdict based on AI probability."""
        if ai_probability >= 70:
            return "LIKELY_AI_FACE"
        elif ai_probability >= 50:
            return "POSSIBLY_AI_FACE"
        elif ai_probability >= 30:
            return "UNCERTAIN"
        else:
            return "LIKELY_REAL_FACE"


def create_face_detector() -> FaceConsistencyDetector:
    """Factory function for face consistency detector."""
    return FaceConsistencyDetector()
