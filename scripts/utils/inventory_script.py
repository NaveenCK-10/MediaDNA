import os
import json
import csv
import hashlib
import concurrent.futures
import cv2

DATASET_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
META_CSV = os.path.join(DATASET_DIR, "meta_data.csv")
OUTPUT_DIR = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF"

def get_file_info(filepath):
    try:
        size = os.path.getsize(filepath)
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        sha256_hex = sha256.hexdigest()

        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            resolution = f"{width}x{height}"
        else:
            fps = "UNKNOWN"
            duration = "UNKNOWN"
            resolution = "UNKNOWN"
    except Exception as e:
        size = "UNKNOWN"
        sha256_hex = "UNKNOWN"
        fps = "UNKNOWN"
        duration = "UNKNOWN"
        resolution = "UNKNOWN"
    
    return {
        "size": size,
        "sha256": sha256_hex,
        "fps": fps,
        "duration": duration,
        "resolution": resolution,
        "video codec": "UNKNOWN",
        "audio codec": "UNKNOWN",
        "audio presence": "UNKNOWN"
    }

def main():
    inventory = []
    provenance = []

    with open(META_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    def process_row(row):
        source = row[0]
        target1 = row[1]
        target2 = row[2]
        method = row[3]
        category = row[4]
        type_ = row[5]
        race = row[6]
        gender = row[7]
        filename = row[8]
        dir_path = row[9]

        rel_dir = dir_path.replace("FakeAVCeleb/", "")
        rel_path = rel_dir + "/" + filename
        full_path = os.path.join(DATASET_DIR, rel_dir, filename).replace("/", os.sep)
        
        if not os.path.exists(full_path):
            return None, None
            
        file_info = get_file_info(full_path)
        
        identity = source if type_ == "RealVideo-RealAudio" else (target1 if target1 != "-" else source)
        speaker = source if type_ == "RealVideo-RealAudio" else (target2 if target2 != "-" else source)
        
        inv_entry = {
            "relative_path": rel_path,
            "size": file_info["size"],
            "duration": file_info["duration"],
            "fps": file_info["fps"],
            "resolution": file_info["resolution"],
            "video codec": file_info["video codec"],
            "audio codec": file_info["audio codec"],
            "audio presence": file_info["audio presence"],
            "identity": identity,
            "speaker": speaker,
            "source/original relationship": source,
            "manipulation category": category,
            "generator/method": method,
            "sha256": file_info["sha256"]
        }

        prov_entry = {
            "fake video": rel_path if method != "real" else "N/A",
            "original video": rel_path if method == "real" else "UNKNOWN", 
            "source video": source,
            "identity": identity,
            "speaker": speaker,
            "audio source": target2 if target2 != "-" else source,
            "video source": target1 if target1 != "-" else source,
            "generator/method": method,
            "derivative": type_
        }
        return inv_entry, prov_entry

    print("Processing...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(process_row, rows))

    for res in results:
        if res and res[0]:
            inventory.append(res[0])
            provenance.append(res[1])

    with open(os.path.join(OUTPUT_DIR, "V22_2_DATASET_INVENTORY.json"), "w") as f:
        json.dump(inventory, f, indent=4)
        
    with open(os.path.join(OUTPUT_DIR, "V22_2_PROVENANCE_GRAPH.json"), "w") as f:
        json.dump(provenance, f, indent=4)
        
    with open(os.path.join(OUTPUT_DIR, "V22_2_DATASET_INVENTORY.md"), "w") as f:
        f.write("# V22.2 Dataset Inventory\n\n")
        f.write(f"Total valid files processed: {len(inventory)}\n")

    print(f"Finished processing {len(inventory)} items.")

if __name__ == '__main__':
    main()
