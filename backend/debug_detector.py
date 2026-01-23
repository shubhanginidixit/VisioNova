
import sys
import os

# Add backend directory to sys.path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from text_detector.detector import AIContentDetector
    print("Successfully imported AIContentDetector")
except ImportError as e:
    print(f"Error importing AIContentDetector: {e}")
    sys.exit(1)

def test_detector():
    print("Initializing Detector...")
    try:
        detector = AIContentDetector()
    except Exception as e:
        print(f"CRITICAL ERROR initializing detector: {e}")
        return

    # Strongly AI-generated text sample
    ai_text = """
    It is important to note that the impact of artificial intelligence on modern society cannot be overstated.
    Furthermore, as language models continue to evolve, they serve as a testament to human innovation.
    In conclusion, we must delve deeper into the ethical implications of this technology to ensure it is utilized responsibly.
    """

    print("\n--- Testing AI Text ---")
    try:
        result = detector.predict(ai_text)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Scores: {result['scores']}")
        print(f"Patterns: {result['detected_patterns']['total_count']}")
    except Exception as e:
        print(f"Error predicting AI text: {e}")

    # Human text sample
    human_text = """
    I went to the store yesterday and bought some milk. The cat was acting weird all morning, jumping on the counters.
    I really need to fix that leaky faucet in the kitchen eventually.
    """

    print("\n--- Testing Human Text ---")
    try:
        result_human = detector.predict(human_text)
        print(f"Prediction: {result_human['prediction']}")
        print(f"Confidence: {result_human['confidence']}")
        print(f"Scores: {result_human['scores']}")
    except Exception as e:
        print(f"Error predicting Human text: {e}")

if __name__ == "__main__":
    test_detector()
