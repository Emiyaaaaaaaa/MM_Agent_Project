import chromadb
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path(r"c:\Users\a2231\Desktop\RAG\chroma_db")
COLLECTION_NAME = "multimodal_papers"

def fix_types():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    
    print("Fetching all entries for cleanup...")
    all_data = collection.get(include=['metadatas'])
    ids = all_data['ids']
    metadatas = all_data['metadatas']
    
    updates_metadatas = []
    update_ids = []
    
    print("Checking for type mismatches...")
    for i, meta in enumerate(metadatas):
        changed = False
        if 'year' in meta:
            val = meta['year']
            if isinstance(val, str):
                try:
                    meta['year'] = int(val)
                    changed = True
                except:
                    pass
        
        if changed:
            updates_metadatas.append(meta)
            update_ids.append(ids[i])
            
    if not update_ids:
        print("No type mismatches found.")
        return

    print(f"Repairing {len(update_ids)} entries in batches...")
    batch_size = 500
    for start in tqdm(range(0, len(update_ids), batch_size)):
        end = start + batch_size
        collection.update(
            ids=update_ids[start:end],
            metadatas=updates_metadatas[start:end]
        )
    print("Repair complete!")

if __name__ == "__main__":
    fix_types()
