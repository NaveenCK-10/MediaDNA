import os
import csv
import random
from collections import defaultdict
import pandas as pd

random.seed(42)

def generate_v14_splits():
    META_PATH = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\meta_data.csv"
    ROOT_DIR = r"FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
    
    # Build graph using STRICT regex extraction from filename to avoid dirty metadata
    videos = []
    all_identities = set()
    
    with open(META_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 10:
                continue
                
            v_type = row[5]
            v_filename = row[8]
            v_folder = row[9].replace("FakeAVCeleb/", "")
            abs_path = os.path.join(ROOT_DIR, v_folder, v_filename)
            
            import re
            ids = set(re.findall(r'(id\d+)', abs_path))
            if not ids:
                continue
                
            all_identities.update(ids)
            videos.append((abs_path, v_type, ids))
            
    print(f"Total videos in metadata: {len(videos)}")
    print(f"Total unique identities: {len(all_identities)}")
    
    # Randomly select 100 identities for the locked test set, 50 for val, rest for train
    identity_list = list(all_identities)
    random.shuffle(identity_list)
    
    test_ids = set(identity_list[:100])
    val_ids = set(identity_list[100:150])
    train_ids = set(identity_list[150:])
    
    train_videos = []
    val_videos = []
    test_videos = []
    discarded = 0
    
    for v_path, v_type, ids in videos:
        label = 1 if v_type != 'RealVideo-RealAudio' else 0
        
        if all(i in train_ids for i in ids):
            train_videos.append((v_path, label, v_type))
        elif all(i in val_ids for i in ids):
            val_videos.append((v_path, label, v_type))
        elif all(i in test_ids for i in ids):
            test_videos.append((v_path, label, v_type))
        else:
            discarded += 1
            
    print(f"\n--- STRICT GRAPH CUT RESULTS ---")
    print(f"Train Videos: {len(train_videos)}")
    print(f"Val Videos: {len(val_videos)}")
    print(f"Test Videos: {len(test_videos)}")
    print(f"Discarded (Bridging) Videos: {discarded}")
    
    os.makedirs('data', exist_ok=True)
    def save_csv(path, v_list):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'label', 'type'])
            for item in v_list:
                writer.writerow([item[0], item[1], item[2]])
        print(f"Saved {len(v_list)} records to {path}")
        
    def balance(v_list):
        grouped = defaultdict(list)
        for v in v_list:
            grouped[v[2]].append(v)
            
        min_count = min([len(g) for k, g in grouped.items() if k != 'Unknown'])
        print(f"Balancing dataset to {min_count} per class...")
        
        sampled = []
        for t in ['RealVideo-RealAudio', 'FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio']:
            if len(grouped[t]) >= min_count:
                sampled.extend(random.sample(grouped[t], min_count))
            else:
                sampled.extend(grouped[t])
        random.shuffle(sampled)
        return sampled

    train_sub = balance(train_videos)
    val_sub = balance(val_videos)
    test_sub = balance(test_videos)
    
    save_csv('data/train_v14.csv', train_sub)
    save_csv('data/val_v14.csv', val_sub)
    save_csv('data/test_locked_v14.csv', test_sub)
    
    print("\nVerifying Strict Leakage...")
    def get_ids(v_list):
        import re
        s = set()
        for v in v_list:
            s.update(re.findall(r'(id\d+)', str(v[0])))
        return s
        
    tr_id = get_ids(train_sub)
    vl_id = get_ids(val_sub)
    te_id = get_ids(test_sub)
    
    assert len(tr_id.intersection(vl_id)) == 0, "Identity Leakage: Train vs Val"
    assert len(tr_id.intersection(te_id)) == 0, "Identity Leakage: Train vs Test"
    assert len(vl_id.intersection(te_id)) == 0, "Identity Leakage: Val vs Test"
    print("V14: ZERO IDENTITY LEAKAGE STRICTLY VERIFIED.")

if __name__ == '__main__':
    generate_v14_splits()
