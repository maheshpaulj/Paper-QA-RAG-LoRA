import json
from pathlib import Path
from src.ingest import extract_meta
from config import INDEX_DIR, DATA_DIR

def main():
    if not INDEX_DIR.exists():
        print("No indexes found.")
        return
        
    for index_path in INDEX_DIR.iterdir():
        if not index_path.is_dir():
            continue
            
        meta_file = index_path / "meta.json"
        if not meta_file.exists():
            continue
            
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        pdf_name = meta.get("source_pdf")
        if not pdf_name:
            continue
            
        pdf_path = Path(INDEX_DIR).parent / pdf_name
        if not pdf_path.exists():
            print(f"PDF {pdf_name} not found for index {index_path.name}")
            continue
            
        print(f"Re-extracting metadata for {index_path.name} from {pdf_name}...")
        new_meta = extract_meta(pdf_path)
        
        # Merge new fields while keeping old ones like source_pdf
        meta.update(new_meta)
        
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        print(f"Updated {meta_file}")

if __name__ == "__main__":
    main()
