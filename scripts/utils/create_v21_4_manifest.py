import os
import csv
import hashlib

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def create_manifest():
    base_dir = os.path.join(PROJECT_ROOT, "FakeAVCeleb_v1.2", "FakeAVCeleb_v1.2")
    categories = {
        "RealVideo-RealAudio": 0,
        "RealVideo-FakeAudio": 1,
        "FakeVideo-RealAudio": 1,
        "FakeVideo-FakeAudio": 1
    }
    
    manifest = []
    sample_id = 0
    for cat, label in categories.items():
        cat_dir = os.path.join(base_dir, cat)
        if os.path.exists(cat_dir):
            for root, _, files in os.walk(cat_dir):
                for file in files:
                    if file.endswith(".mp4"):
                        path = os.path.join(root, file)
                        sha = compute_sha256(path)
                        manifest.append({
                            "sample_id": f"SAMP_{sample_id:04d}",
                            "path": path,
                            "ground_truth": label,
                            "video_manipulation": cat.split("-")[0],
                            "audio_manipulation": cat.split("-")[1],
                            "identity": "UNKNOWN", # To be mapped
                            "source_video": "UNKNOWN",
                            "speaker": "UNKNOWN",
                            "generator": "UNKNOWN",
                            "sha256": sha,
                            "duration": 0, # Placeholder
                            "resolution": "224x224"
                        })
                        sample_id += 1
                        if len([m for m in manifest if m["ground_truth"] == label and m["video_manipulation"] == cat.split("-")[0]]) >= 2:
                            break
                if len([m for m in manifest if m["ground_truth"] == label and m["video_manipulation"] == cat.split("-")[0]]) >= 2:
                    break

    with open("V21_4_LOCKED_TEST_MANIFEST.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
        
    print(f"Created locked manifest with {len(manifest)} samples.")

if __name__ == "__main__":
    create_manifest()
