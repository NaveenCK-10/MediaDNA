import os
import csv
import random
from collections import defaultdict

random.seed(42)

def generate_v8_splits():
    META_PATH = r"FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\meta_data.csv"
    ROOT_DIR = r"FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
    
    if not os.path.exists(META_PATH):
        print(f"Error: Could not find {META_PATH}")
        return

    # Parse metadata and build identity relationships
    videos = []
    
    id_adj = defaultdict(set)
    video_to_ids = {}
    
    with open(META_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 10:
                continue
            
            source, target1, target2 = row[0], row[1], row[2]
            v_type = row[5] # e.g. RealVideo-RealAudio
            v_race = row[6]
            v_gender = row[7]
            v_filename = row[8]
            v_folder = row[9].replace("FakeAVCeleb/", "")
            
            # Construct exact absolute path based on standard structure
            abs_path = os.path.join(ROOT_DIR, v_folder, v_filename)
            if not os.path.exists(abs_path):
                continue
                
            ids = set()
            if source != '-': ids.add(source)
            
            video_to_ids[abs_path] = ids
            videos.append((abs_path, v_type, ids))
            
            # Add edges for all pairs in ids
            id_list = list(ids)
            for i in range(len(id_list)):
                for j in range(i+1, len(id_list)):
                    id_adj[id_list[i]].add(id_list[j])
                    id_adj[id_list[j]].add(id_list[i])

    # Find connected components of identities
    visited_ids = set()
    clusters = []
    all_ids = set(id_adj.keys())
    for v in videos:
        all_ids.update(v[2])
        
    for i in all_ids:
        if i not in visited_ids:
            comp = set()
            q = [i]
            visited_ids.add(i)
            while q:
                curr = q.pop(0)
                comp.add(curr)
                for neighbor in id_adj[curr]:
                    if neighbor not in visited_ids:
                        visited_ids.add(neighbor)
                        q.append(neighbor)
            clusters.append(comp)
            
    print(f"Total videos found: {len(videos)}")
    print(f"Total isolated identity clusters: {len(clusters)}")
    
    # Shuffle clusters to randomize split
    random.shuffle(clusters)
    
    train_ids = set()
    val_ids = set()
    test_ids = set()
    
    n_clusters = len(clusters)
    train_c = int(n_clusters * 0.70)
    val_c = int(n_clusters * 0.15)
    
    for c in clusters[:train_c]: train_ids.update(c)
    for c in clusters[train_c:train_c+val_c]: val_ids.update(c)
    for c in clusters[train_c+val_c:]: test_ids.update(c)
    
    train_videos = []
    val_videos = []
    test_videos = []
    
    for v_path, v_type, ids in videos:
        rep = next(iter(ids)) if ids else None
        label = 1 if v_type != 'RealVideo-RealAudio' else 0
        
        if rep in train_ids:
            train_videos.append((v_path, label, v_type))
        elif rep in val_ids:
            val_videos.append((v_path, label, v_type))
        elif rep in test_ids:
            test_videos.append((v_path, label, v_type))
        else:
            pass

    os.makedirs('data', exist_ok=True)
    def save_csv(path, v_list):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['video_path', 'label', 'type'])
            for item in v_list:
                writer.writerow([item[0], item[1], item[2]])
        print(f"Saved {len(v_list)} records to {path}")
        
    def balance_and_sample(v_list, max_total):
        grouped = defaultdict(list)
        for v in v_list:
            grouped[v[2]].append(v)
            
        target_per_class = max_total // 4
        sampled = []
        for t in ['RealVideo-RealAudio', 'FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio']:
            population = grouped[t]
            if len(population) > target_per_class:
                sampled.extend(random.sample(population, target_per_class))
            else:
                sampled.extend(population)
        random.shuffle(sampled)
        return sampled

    # We use 800 train, 200 val, 200 test to keep it extremely fast for experiments on RTX 4070
    train_sub = balance_and_sample(train_videos, 800)
    val_sub = balance_and_sample(val_videos, 200)
    test_sub = balance_and_sample(test_videos, 200)
    
    save_csv('data/train_v8.csv', train_sub)
    save_csv('data/val_v8.csv', val_sub)
    save_csv('data/test_locked_v8.csv', test_sub)
    
    print("\nVerifying Leakage...")
    train_types = set([x[0] for x in train_sub])
    val_types = set([x[0] for x in val_sub])
    test_types = set([x[0] for x in test_sub])
    
    assert len(train_types.intersection(val_types)) == 0, "Video Leakage: Train vs Val"
    assert len(train_types.intersection(test_types)) == 0, "Video Leakage: Train vs Test"
    assert len(val_types.intersection(test_types)) == 0, "Video Leakage: Val vs Test"
    
    train_id_set = set([id for v in train_sub for id in video_to_ids[v[0]]])
    val_id_set = set([id for v in val_sub for id in video_to_ids[v[0]]])
    test_id_set = set([id for v in test_sub for id in video_to_ids[v[0]]])
    
    assert len(train_id_set.intersection(val_id_set)) == 0, "Identity Leakage: Train vs Val"
    assert len(train_id_set.intersection(test_id_set)) == 0, "Identity Leakage: Train vs Test"
    assert len(val_id_set.intersection(test_id_set)) == 0, "Identity Leakage: Val vs Test"
    print("Zero Identity/Video Leakage Verified.")

if __name__ == '__main__':
    generate_v8_splits()
