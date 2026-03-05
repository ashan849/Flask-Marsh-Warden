# debug_pickle.py
import pickle
import os

def debug_pickle_file():
    filenames = [
        "pdf_index_enhanced1.pkl", 
        "pdf_index_enhanced2.pkl",
        "pdf_index_enhanced3.pkl"
    ]
    
    for filename in filenames:
        print(f"\n{'='*60}")
        print(f"Checking file: {filename}")
        print(f"{'='*60}")
        
        if not os.path.exists(filename):
            print(f"❌ File not found: {filename}")
            continue  # Skip to next file
        
        print(f"📁 File exists: {filename}")
        print(f"📏 File size: {os.path.getsize(filename)} bytes")
        
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
            
            print("\n✅ Pickle loaded successfully!")
            print(f"\n📊 Data structure:")
            print(f"  - Type: {type(data)}")
            
            if isinstance(data, dict):
                print(f"  - Keys: {list(data.keys())}")
                
                for key, value in data.items():
                    print(f"\n  🔑 Key: {key}")
                    print(f"    Type: {type(value)}")
                    
                    if key == "documents":
                        if isinstance(value, list):
                            print(f"    Count: {len(value)}")
                            if value:
                                print(f"    First item type: {type(value[0])}")
                                if hasattr(value[0], 'page_content'):
                                    print(f"    First document preview: {value[0].page_content[:200]}...")
                    elif key == "embeddings":
                        if hasattr(value, 'shape'):
                            print(f"    Shape: {value.shape}")
                        print(f"    Type: {type(value)}")
                    else:
                        print(f"    Value: {value}")
            
            elif isinstance(data, list):
                print(f"  - List length: {len(data)}")
                if data:
                    print(f"  - First item type: {type(data[0])}")
            
        except Exception as e:
            print(f"\n❌ Error loading pickle: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_pickle_file()