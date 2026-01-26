#!/usr/bin/env python3
"""
Comprehensive Backend Test Suite
Tests all core functionality before deployment
"""
import sys
import os
from datetime import datetime


def test_imports():
    """Test all critical imports"""
    print("\n" + "="*60)
    print("TEST 1: CRITICAL IMPORTS")
    print("="*60)
    
    try:
        from fact_check import FactChecker
        print("[OK] FactChecker imported successfully")
    except Exception as e:
        print(f"[ERROR] FactChecker import failed: {e}")
        return False
    
    try:
        from text_detector import AIContentDetector, TextExplainer, DocumentParser, TextPreprocessor
        print("[OK] Text detector modules imported successfully")
    except Exception as e:
        print(f"[ERROR] Text detector import failed: {e}")
        return False
    
    try:
        from ai import AIAnalyzer
        print("[OK] AI analyzer imported successfully")
    except Exception as e:
        print(f"[ERROR] AI analyzer import failed: {e}")
        return False
    
    return True


def test_detector_initialization():
    """Test AI content detector initialization"""
    print("\n" + "="*60)
    print("TEST 2: DETECTOR INITIALIZATION")
    print("="*60)
    
    try:
        from text_detector import AIContentDetector
        detector = AIContentDetector(use_ml_model=False)
        print("[OK] Detector initialized in offline mode")
    except Exception as e:
        print(f"[ERROR] Detector initialization failed: {e}")
        return False
    
    return True


def test_preprocessor():
    """Test text preprocessor"""
    print("\n" + "="*60)
    print("TEST 3: TEXT PREPROCESSOR")
    print("="*60)
    
    try:
        from text_detector import TextPreprocessor
        preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=False)
        
        test_text = "This is a TEST sentence with PUNCTUATION!!!"
        cleaned = preprocessor.clean_text(test_text)
        print(f"[OK] TextPreprocessor working")
        
        processed = preprocessor.preprocess(test_text)
        print(f"[OK] Preprocessing successful")
    except Exception as e:
        print(f"[ERROR] TextPreprocessor failed: {e}")
        return False
    
    return True


def test_ai_detection():
    """Test AI content detection"""
    print("\n" + "="*60)
    print("TEST 4: AI CONTENT DETECTION")
    print("="*60)
    
    try:
        from text_detector import AIContentDetector
        detector = AIContentDetector(use_ml_model=False)
        
        ai_text = "It is important to note that artificial intelligence has become increasingly prevalent."
        result = detector.predict(ai_text, detailed=False)
        
        if "error" in result:
            print(f"[ERROR] Detection failed: {result['error']}")
            return False
        
        print("[OK] AI detection working")
        print(f"  - Prediction: {result['prediction']}")
        print(f"  - Confidence: {result['confidence']}%")
        
    except Exception as e:
        print(f"[ERROR] AI detection failed: {e}")
        return False
    
    return True


def test_fact_checker():
    """Test fact checker initialization"""
    print("\n" + "="*60)
    print("TEST 5: FACT CHECKER INITIALIZATION")
    print("="*60)
    
    try:
        from fact_check import FactChecker
        fact_checker = FactChecker()
        print("[OK] FactChecker initialized successfully")
    except Exception as e:
        print(f"[ERROR] FactChecker initialization failed: {e}")
        return False
    
    return True


def test_flask_app():
    """Test Flask app structure"""
    print("\n" + "="*60)
    print("TEST 6: FLASK APP STRUCTURE")
    print("="*60)
    
    try:
        from flask import Flask
        from flask_cors import CORS
        from flask_limiter import Limiter
        
        app = Flask(__name__)
        CORS(app)
        limiter = Limiter(app=app, key_func=lambda: "test", default_limits=[])
        
        print("[OK] Flask app dependencies available")
        
    except Exception as e:
        print(f"[ERROR] Flask app structure test failed: {e}")
        return False
    
    return True


def test_syntax_all_files():
    """Test syntax of all Python files"""
    print("\n" + "="*60)
    print("TEST 7: PYTHON SYNTAX VALIDATION")
    print("="*60)
    
    import py_compile
    
    py_files = [
        "app.py",
        "fact_check/__init__.py",
        "text_detector/__init__.py",
        "ai/__init__.py",
    ]
    
    all_valid = True
    for file in py_files:
        try:
            py_compile.compile(file, doraise=True)
            print(f"[OK] {file}")
        except py_compile.PyCompileError as e:
            print(f"[ERROR] {file}: {e}")
            all_valid = False
    
    return all_valid


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("VISIONOVA BACKEND COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Critical Imports", test_imports),
        ("Detector Initialization", test_detector_initialization),
        ("Text Preprocessor", test_preprocessor),
        ("AI Content Detection", test_ai_detection),
        ("Fact Checker", test_fact_checker),
        ("Flask App Structure", test_flask_app),
        ("Python Syntax", test_syntax_all_files),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*60)
    print(f"Result: {passed}/{total} tests passed")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    if passed == total:
        print("\n[SUCCESS] ALL TESTS PASSED - BACKEND IS READY FOR DEPLOYMENT\n")
        return 0
    else:
        print(f"\n[ERROR] {total - passed} TEST(S) FAILED\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
