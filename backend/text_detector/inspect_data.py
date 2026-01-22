
from datasets import load_dataset

def inspect():
    try:
        print("Loading dataset gsingh1-py/train...")
        dataset = load_dataset("gsingh1-py/train", split="train", streaming=True)
        
        print("Inspecting first 5 items:")
        for i, item in enumerate(dataset):
            print(f"\nItem {i}:")
            print(f"Keys: {list(item.keys())}")
            # Print first 100 chars of text-like distinct values
            for k, v in item.items():
                if isinstance(v, str):
                    print(f"  {k}: {v[:100]}...")
                else:
                    print(f"  {k}: {v}")
            
            if i >= 4:
                break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect()
