"""
Evaluate trained model on test set and real-world samples.
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def load_model(model_path: str = "model_trained"):
    """Load trained model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return tokenizer, model, device


def predict(text: str, tokenizer, model, device):
    """Predict if text is AI or human."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    prob_human = probs[0][0].item()
    prob_ai = probs[0][1].item()
    
    prediction = "ai" if prob_ai > prob_human else "human"
    confidence = max(prob_human, prob_ai)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "prob_human": prob_human,
        "prob_ai": prob_ai
    }


def evaluate_test_set(tokenizer, model, device, test_file: str = "datasets/test.csv"):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    test_df = pd.read_csv(test_file)
    
    # Map labels
    label_map = {"human": 0, "ai": 1}
    test_df['label_int'] = test_df['label'].map(label_map)
    
    predictions = []
    true_labels = []
    
    print(f"\nProcessing {len(test_df)} test samples...")
    
    for idx, row in test_df.iterrows():
        result = predict(row['text'], tokenizer, model, device)
        pred_label = 1 if result['prediction'] == 'ai' else 0
        
        predictions.append(pred_label)
        true_labels.append(row['label_int'])
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_df)}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("Test Set Results")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(
        true_labels, 
        predictions, 
        target_names=['human', 'ai'],
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print(f"              Predicted")
    print(f"              Human  AI")
    print(f"Actual Human  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"       AI     {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # Calculate accuracy
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


def test_real_samples(tokenizer, model, device):
    """Test on real-world samples."""
    print("\n" + "="*60)
    print("Testing on Real-World Samples")
    print("="*60)
    
    # Sample texts (you can add more)
    samples = {
        "GPT-4 Sample": """
            Artificial intelligence has revolutionized numerous industries in recent years. 
            It's important to note that machine learning algorithms have become increasingly 
            sophisticated, enabling applications ranging from natural language processing to 
            computer vision. Furthermore, the ethical implications of AI deployment must be 
            carefully considered to ensure responsible development.
        """,
        
        "Human Sample": """
            I just got back from the grocery store and wow, prices are crazy right now! 
            My usual cart cost like $30 more than last month. Anyone else noticing this? 
            Had to skip buying some stuff I actually needed. This economy is really something.
        """,
        
        "Claude Sample": """
            When considering the complexities of climate change, it's crucial to examine 
            multiple perspectives. The scientific consensus indicates rising temperatures, 
            yet solutions require coordinated global action. Nevertheless, implementing 
            effective policies presents significant challenges that policymakers must address.
        """,
        
        "Human Creative": """
            The sunset was gorgeous tonight - all pink and orange streaks across the sky. 
            Made me stop whatever I was doing and just... look. Don't get many moments like 
            that anymore, everything's always so rush rush rush. Sometimes you just gotta 
            appreciate the simple stuff, ya know?
        """
    }
    
    for name, text in samples.items():
        result = predict(text.strip(), tokenizer, model, device)
        
        print(f"\n{name}:")
        print(f"  Prediction: {result['prediction'].upper()}")
        print(f"  Confidence: {result['confidence']*100:.1f}%")
        print(f"  Human: {result['prob_human']*100:.1f}% | AI: {result['prob_ai']*100:.1f}%")


def main():
    """Main evaluation pipeline."""
    print("\n" + "🔍 "*20)
    print("Model Evaluation")
    print("🔍 "*20 + "\n")
    
    try:
        # Load model
        tokenizer, model, device = load_model()
        
        # Evaluate on test set
        evaluate_test_set(tokenizer, model, device)
        
        # Test on real samples
        test_real_samples(tokenizer, model, device)
        
        print("\n" + "="*60)
        print("✅ Evaluation Complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
