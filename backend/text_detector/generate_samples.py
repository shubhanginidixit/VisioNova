"""
Generate AI text samples using Groq API for training dataset.
This script creates diverse AI-generated samples from multiple models to ensure
the detector works on GPT-5 and all future AI models.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add parent directory to path to import groq_client
sys.path.append(str(Path(__file__).parent.parent))
from ai.groq_client import GroqClient


# Diverse prompts covering multiple domains
PROMPTS = [
    # Academic/Formal
    "Write a detailed explanation of how photosynthesis works in plants.",
    "Explain the theory of relativity in simple terms.",
    "Discuss the impact of social media on modern society.",
    "Analyze the causes and effects of climate change.",
    "Explain the concept of artificial intelligence and machine learning.",
    
    # Technical
    "Explain how a computer processor works.",
    "Describe the basics of blockchain technology.",
    "What is cloud computing and how does it work?",
    "Explain the fundamentals of cybersecurity.",
    "Describe how neural networks learn from data.",
    
    # Creative/Casual
    "Write a short story about a day at the beach.",
    "Describe your ideal vacation destination.",
    "What are some tips for staying productive while working from home?",
    "Write about the benefits of reading books.",
    "Describe the process of learning a new language.",
    
    # Opinion/Analysis
    "What are the pros and cons of remote work?",
    "Discuss the importance of education in modern society.",
    "What makes a good leader?",
    "Analyze the role of technology in education.",
    "What are the ethical considerations in AI development?",
    
    # Informative
    "How to maintain a healthy lifestyle?",
    "What are the benefits of regular exercise?",
    "Explain the water cycle to a middle school student.",
    "How does the human immune system fight diseases?",
    "What are renewable energy sources and why are they important?",
    
    # Conversational
    "What's your opinion on electric vehicles?",
    "How has the internet changed communication?",
    "What advice would you give to someone starting their career?",
    "Describe the perfect morning routine.",
    "What are some ways to reduce stress in daily life?",
]


# Extended prompts for more diverse dataset
ADDITIONAL_PROMPTS = [
    "Explain quantum computing to a beginner.",
    "What are the main challenges in space exploration?",
    "Describe the process of writing a research paper.",
    "How do vaccines work to protect against diseases?",
    "What is cryptocurrency and how does it work?",
    "Explain the concept of sustainable development.",
    "What are the benefits of meditation and mindfulness?",
    "Describe the evolution of smartphones.",
    "How does GPS technology work?",
    "What are the key principles of effective time management?",
    "Explain the greenhouse effect and global warming.",
    "What makes a city livable and sustainable?",
    "Describe the impact of automation on jobs.",
    "How does the stock market work?",
    "What are the fundamentals of graphic design?",
    "Explain the concept of big data and its applications.",
    "What are the benefits of learning to code?",
    "Describe the history and evolution of the internet.",
    "How does 5G technology differ from 4G?",
    "What are the principles of sustainable agriculture?",
]


# Groq models to use for generation
GROQ_MODELS = [
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]


class SampleGenerator:
    """Generate AI text samples using Groq API."""
    
    def __init__(self, output_dir: str = "datasets"):
        self.client = GroqClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_sample(self, prompt: str, model: str, max_retries: int = 3) -> Dict:
        """Generate a single AI sample."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.7,
                    max_tokens=500
                )
                
                return {
                    "text": response,
                    "label": "ai",
                    "model": model,
                    "prompt": prompt,
                    "source": "groq_generated"
                }
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to generate sample for prompt: {prompt[:50]}...")
                    return None
        
        return None
    
    def generate_dataset(
        self, 
        num_samples: int = 1000,
        prompts: List[str] = None,
        models: List[str] = None
    ) -> pd.DataFrame:
        """Generate a dataset of AI samples."""
        if prompts is None:
            prompts = PROMPTS + ADDITIONAL_PROMPTS
        
        if models is None:
            models = GROQ_MODELS
        
        samples = []
        samples_per_model = num_samples // len(models)
        
        print(f"Generating {num_samples} samples ({samples_per_model} per model)...")
        print(f"Models: {', '.join(models)}")
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"Generating samples with {model}...")
            print(f"{'='*60}")
            
            model_samples = 0
            prompt_idx = 0
            
            while model_samples < samples_per_model:
                # Cycle through prompts
                prompt = prompts[prompt_idx % len(prompts)]
                prompt_idx += 1
                
                sample = self.generate_sample(prompt, model)
                
                if sample:
                    samples.append(sample)
                    model_samples += 1
                    
                    if model_samples % 10 == 0:
                        print(f"Progress: {model_samples}/{samples_per_model} samples")
                
                # Rate limiting - Groq allows ~30 requests/minute
                time.sleep(2)
        
        df = pd.DataFrame(samples)
        print(f"\n{'='*60}")
        print(f"Generated {len(df)} total samples")
        print(f"{'='*60}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "ai_samples_groq.csv"):
        """Save generated dataset to CSV."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\nDataset saved to: {filepath}")
        print(f"Total samples: {len(df)}")
        print(f"\nSamples per model:")
        print(df['model'].value_counts())
        
        return filepath


def main():
    """Main execution function."""
    print("="*60)
    print("AI Sample Generator for VisioNova Text Detection")
    print("="*60)
    
    generator = SampleGenerator()
    
    # Generate 1000 samples (250 per model) - adjust as needed
    # For full 30K dataset, set num_samples=30000
    num_samples = 1000  # Start small for testing
    
    print(f"\nGenerating {num_samples} AI samples...")
    print("This will take approximately {:.1f} minutes".format(num_samples * 2 / 60))
    print("\nPress Ctrl+C to stop generation\n")
    
    try:
        df = generator.generate_dataset(num_samples=num_samples)
        filepath = generator.save_dataset(df)
        
        print("\n✅ Dataset generation complete!")
        print(f"📁 Saved to: {filepath}")
        print("\nNext steps:")
        print("1. Download HC3 dataset using download_hc3.py")
        print("2. Combine datasets and split into train/val/test")
        print("3. Run training script")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Generation interrupted by user")
        if len(generator.samples) > 0:
            print(f"Saving {len(generator.samples)} samples generated so far...")
            df = pd.DataFrame(generator.samples)
            generator.save_dataset(df, "ai_samples_groq_partial.csv")
    
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
