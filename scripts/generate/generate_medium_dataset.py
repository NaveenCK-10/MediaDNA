"""
Phase 5-6: Generate identity-based train/val/test CSVs for medium-scale OpenAVFF baseline.
Then validate every video path, audio extraction, and split integrity.

Split strategy: 300 train / 100 val / 100 test identities
Per identity: 1 video from each of the 4 categories
Total: 1200 train + 400 val + 400 test = 2000 videos
"""
import os
import csv
import random
import subprocess
import tempfile
import collections
import sys

SEED = 42
ROOT = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
FFMPEG = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
DATA_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data"

CATEGORIES = ['FakeVideo-FakeAudio', 'FakeVideo-RealAudio', 'RealVideo-FakeAudio', 'RealVideo-RealAudio']
# Labels: anything not RealVideo-RealAudio is Fake (1)
LABEL_MAP = {
    'FakeVideo-FakeAudio': 1,
    'FakeVideo-RealAudio': 1,
    'RealVideo-FakeAudio': 1,
    'RealVideo-RealAudio': 0,
}

TRAIN_IDS = 300
VAL_IDS = 100
TEST_IDS = 100

def discover_videos():
    """Build {category: {identity: [video_paths]}} mapping."""
    cat_id_videos = {}
    for cat in CATEGORIES:
        cat_path = os.path.join(ROOT, cat)
        id_videos = collections.defaultdict(list)
        if os.path.isdir(cat_path):
            for root, dirs, files in os.walk(cat_path):
                for f in files:
                    if f.endswith('.mp4'):
                        full = os.path.join(root, f)
                        parts = full.replace(cat_path, '').strip(os.sep).split(os.sep)
                        if len(parts) >= 3:
                            identity = parts[2]
                            id_videos[identity].append(full)
        cat_id_videos[cat] = dict(id_videos)
    return cat_id_videos

def find_common_identities(cat_id_videos):
    """Find identities present in ALL 4 categories."""
    id_sets = [set(cat_id_videos[cat].keys()) for cat in CATEGORIES]
    common = id_sets[0]
    for s in id_sets[1:]:
        common = common & s
    return sorted(common)

def split_identities(common_ids, seed=SEED):
    """Split identities into train/val/test groups."""
    random.seed(seed)
    shuffled = list(common_ids)
    random.shuffle(shuffled)
    
    train_ids = shuffled[:TRAIN_IDS]
    val_ids = shuffled[TRAIN_IDS:TRAIN_IDS + VAL_IDS]
    test_ids = shuffled[TRAIN_IDS + VAL_IDS:TRAIN_IDS + VAL_IDS + TEST_IDS]
    
    return train_ids, val_ids, test_ids

def build_csv_data(identity_list, cat_id_videos, seed=SEED):
    """For each identity, select 1 video from each category. Returns list of (path, label, category)."""
    random.seed(seed)
    rows = []
    for identity in identity_list:
        for cat in CATEGORIES:
            videos = cat_id_videos[cat].get(identity, [])
            if videos:
                selected = random.choice(videos)
                rows.append((selected, LABEL_MAP[cat], cat))
    return rows

def write_csv(filepath, rows):
    """Write CSV with header video_name,target (matching repository format)."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'target'])
        for path, label, cat in rows:
            writer.writerow([path, label])
    print(f"  Written {filepath}: {len(rows)} samples")

def write_csv_with_category(filepath, rows):
    """Write CSV with extra category column for analysis."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'target', 'category'])
        for path, label, cat in rows:
            writer.writerow([path, label, cat])
    print(f"  Written {filepath}: {len(rows)} samples (with categories)")

def validate_split(train_rows, val_rows, test_rows):
    """Verify no path overlap between splits."""
    train_paths = set(r[0] for r in train_rows)
    val_paths = set(r[0] for r in val_rows)
    test_paths = set(r[0] for r in test_rows)
    
    assert len(train_paths & val_paths) == 0, "LEAK: train ∩ val overlap!"
    assert len(train_paths & test_paths) == 0, "LEAK: train ∩ test overlap!"
    assert len(val_paths & test_paths) == 0, "LEAK: val ∩ test overlap!"
    assert len(train_paths) == len(train_rows), "Duplicate paths in train!"
    assert len(val_paths) == len(val_rows), "Duplicate paths in val!"
    assert len(test_paths) == len(test_rows), "Duplicate paths in test!"
    print("  ✓ No path overlap between splits")
    print("  ✓ No duplicate paths within splits")

def validate_identity_separation(train_ids, val_ids, test_ids):
    """Verify no identity overlap between splits."""
    assert len(set(train_ids) & set(val_ids)) == 0, "LEAK: train ∩ val identity overlap!"
    assert len(set(train_ids) & set(test_ids)) == 0, "LEAK: train ∩ test identity overlap!"
    assert len(set(val_ids) & set(test_ids)) == 0, "LEAK: val ∩ test identity overlap!"
    print("  ✓ No identity overlap between splits")

def validate_videos(rows, max_check=20):
    """Spot-check video paths exist and audio is extractable."""
    report = []
    total = len(rows)
    bad = 0
    
    # Check ALL paths exist
    for path, label, cat in rows:
        if not os.path.isfile(path):
            report.append(f"MISSING: {path}")
            bad += 1
    
    if bad > 0:
        print(f"  ✗ {bad}/{total} paths are missing!")
    else:
        print(f"  ✓ All {total} video paths exist")
    
    # Spot-check audio extraction on a subset
    check_indices = random.sample(range(total), min(max_check, total))
    audio_ok = 0
    audio_fail = 0
    for idx in check_indices:
        path = rows[idx][0]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            temp_wav = tmp.name
        try:
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", temp_wav]
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and os.path.getsize(temp_wav) > 0:
                audio_ok += 1
            else:
                audio_fail += 1
                report.append(f"AUDIO_FAIL: {path} (returncode={result.returncode})")
        except Exception as e:
            audio_fail += 1
            report.append(f"AUDIO_ERROR: {path} ({e})")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    
    print(f"  Audio spot-check: {audio_ok}/{audio_ok + audio_fail} OK (checked {len(check_indices)} of {total})")
    
    return report

def print_distribution(rows, name):
    """Print category and label distribution."""
    cat_counts = collections.Counter(r[2] for r in rows)
    label_counts = collections.Counter(r[1] for r in rows)
    print(f"\n  {name} distribution ({len(rows)} total):")
    for cat in CATEGORIES:
        print(f"    {cat}: {cat_counts.get(cat, 0)}")
    print(f"    Label 0 (Real): {label_counts.get(0, 0)}")
    print(f"    Label 1 (Fake): {label_counts.get(1, 0)}")


def main():
    print("=" * 60)
    print("PHASE 5: Dataset Split Generation")
    print("=" * 60)
    
    print("\n1. Discovering videos...")
    cat_id_videos = discover_videos()
    for cat in CATEGORIES:
        print(f"  {cat}: {sum(len(v) for v in cat_id_videos[cat].values())} videos, {len(cat_id_videos[cat])} identities")
    
    print("\n2. Finding common identities...")
    common = find_common_identities(cat_id_videos)
    print(f"  {len(common)} identities present in all 4 categories")
    
    needed = TRAIN_IDS + VAL_IDS + TEST_IDS
    if len(common) < needed:
        print(f"  WARNING: Only {len(common)} common identities, need {needed}. Adjusting...")
        # Proportionally reduce
        ratio = len(common) / needed
        TRAIN_IDS_ADJ = int(300 * ratio)
        VAL_IDS_ADJ = int(100 * ratio)
        TEST_IDS_ADJ = len(common) - TRAIN_IDS_ADJ - VAL_IDS_ADJ
    else:
        TRAIN_IDS_ADJ = TRAIN_IDS
        VAL_IDS_ADJ = VAL_IDS
        TEST_IDS_ADJ = TEST_IDS
    
    print(f"\n3. Splitting {len(common)} identities: {TRAIN_IDS_ADJ} train / {VAL_IDS_ADJ} val / {TEST_IDS_ADJ} test")
    train_ids, val_ids, test_ids = split_identities(common)
    
    print("\n4. Building CSV rows...")
    train_rows = build_csv_data(train_ids, cat_id_videos, seed=SEED)
    val_rows = build_csv_data(val_ids, cat_id_videos, seed=SEED + 1)
    test_rows = build_csv_data(test_ids, cat_id_videos, seed=SEED + 2)
    
    print_distribution(train_rows, "TRAIN")
    print_distribution(val_rows, "VAL")
    print_distribution(test_rows, "TEST")
    
    print("\n5. Writing CSV files...")
    os.makedirs(DATA_DIR, exist_ok=True)
    write_csv(os.path.join(DATA_DIR, 'train_medium.csv'), train_rows)
    write_csv(os.path.join(DATA_DIR, 'val_medium.csv'), val_rows)
    write_csv(os.path.join(DATA_DIR, 'test_medium.csv'), test_rows)
    # Also write category-annotated versions for analysis
    write_csv_with_category(os.path.join(DATA_DIR, 'test_medium_annotated.csv'), test_rows)
    
    print("\n" + "=" * 60)
    print("PHASE 6: Dataset Validation")
    print("=" * 60)
    
    print("\n6. Validating identity separation...")
    validate_identity_separation(train_ids, val_ids, test_ids)
    
    print("\n7. Validating path uniqueness and split integrity...")
    validate_split(train_rows, val_rows, test_rows)
    
    print("\n8. Validating video files (train)...")
    report_train = validate_videos(train_rows, max_check=10)
    
    print("\n9. Validating video files (val)...")
    report_val = validate_videos(val_rows, max_check=10)
    
    print("\n10. Validating video files (test)...")
    report_test = validate_videos(test_rows, max_check=10)
    
    # Write validation report
    report_path = os.path.join(DATA_DIR, 'medium_dataset_validation.txt')
    with open(report_path, 'w') as f:
        f.write("OpenAVFF Medium Dataset Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Common identities: {len(common)}\n")
        f.write(f"Train identities: {len(train_ids)}\n")
        f.write(f"Val identities: {len(val_ids)}\n")
        f.write(f"Test identities: {len(test_ids)}\n\n")
        f.write(f"Train samples: {len(train_rows)}\n")
        f.write(f"Val samples: {len(val_rows)}\n")
        f.write(f"Test samples: {len(test_rows)}\n\n")
        
        for name, rows in [("TRAIN", train_rows), ("VAL", val_rows), ("TEST", test_rows)]:
            cat_counts = collections.Counter(r[2] for r in rows)
            f.write(f"{name} distribution:\n")
            for cat in CATEGORIES:
                f.write(f"  {cat}: {cat_counts.get(cat, 0)}\n")
            f.write("\n")
        
        all_issues = report_train + report_val + report_test
        if all_issues:
            f.write("ISSUES:\n")
            for issue in all_issues:
                f.write(f"  {issue}\n")
        else:
            f.write("No issues found.\n")
    
    print(f"\nValidation report written to: {report_path}")
    
    total_issues = len(report_train) + len(report_val) + len(report_test)
    if total_issues > 0:
        print(f"\n⚠ {total_issues} issues found. Check the validation report.")
    else:
        print("\n✓ All validation checks passed. Dataset is ready for training.")

if __name__ == '__main__':
    main()
