import sys
import gzip
import json
from pathlib import Path
import pandas as pd

def clean_category(cat_val):
    if not cat_val:
        return "Unknown"
    # If list of lists (Amazon metadata format)
    if isinstance(cat_val, list):
        if len(cat_val) > 0 and isinstance(cat_val[0], list):
            # Take the first sublist and get the most specific category
            flat = [str(x) for x in cat_val[0] if x]
            if flat:
                # Return the last (most specific) category item, e.g. "Headphones"
                return flat[-1]
        else:
            flat = [str(x) for x in cat_val if x]
            if flat:
                return flat[-1]
    return str(cat_val)

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_categories.py <path_to_raw_dataset>")
        print("Examples:")
        print("  python extract_categories.py C:\\data\\meta_Electronics.json")
        print("  python extract_categories.py C:\\data\\meta_Electronics.json.gz")
        sys.exit(1)
        
    raw_path = Path(sys.argv[1])
    if not raw_path.exists():
        print(f"Error: File '{raw_path}' does not exist.")
        sys.exit(1)
        
    print(f"Processing raw data from '{raw_path}'...")
    mapping = {}
    
    # Check extension
    if raw_path.suffix == '.parquet':
        try:
            df = pd.read_parquet(raw_path, columns=['asin', 'category'])
            for _, row in df.iterrows():
                asin = str(row['asin'])
                cat = clean_category(row['category'])
                if asin and cat:
                    mapping[asin] = cat
        except Exception as e:
            print(f"Error loading Parquet: {str(e)}")
            sys.exit(1)
    elif raw_path.suffix == '.csv':
        try:
            df = pd.read_csv(raw_path, usecols=['asin', 'category'])
            for _, row in df.iterrows():
                asin = str(row['asin'])
                cat = clean_category(row['category'])
                if asin and cat:
                    mapping[asin] = cat
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            sys.exit(1)
    else:
        # Assume json or json.gz
        is_gz = raw_path.suffix == '.gz' or raw_path.name.endswith('.json.gz')
        open_fn = gzip.open if is_gz else open
        count = 0
        import ast
        try:
            with open_fn(raw_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        try:
                            row = json.loads(line)
                        except Exception:
                            row = ast.literal_eval(line)
                        
                        asin = row.get('asin')
                        cat_val = row.get('categories')
                        if cat_val is None:
                            cat_val = row.get('category')
                            
                        if asin:
                            cat = clean_category(cat_val)
                            mapping[asin] = cat
                        count += 1
                        if count % 100000 == 0:
                            print(f"  Processed {count:,} lines...")
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error reading JSON file: {str(e)}")
            sys.exit(1)
            
    # Save the mapping
    output_path = Path("item_categories.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
        
    print(f"Success! Extracted {len(mapping):,} categories and saved them to '{output_path}'.")

if __name__ == '__main__':
    main()
