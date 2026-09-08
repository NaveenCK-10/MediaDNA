import os
import pandas as pd
import re

def extract_identity(path):
    """Extract identity (e.g., id00076) from a path string."""
    match = re.search(r'(id\d+)', str(path))
    if match:
        return match.group(1)
    return None

def audit_leakage():
    train_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\train_v8.csv"
    val_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\val_v8.csv"
    test_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\test_locked_v8.csv"
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print("ERROR: One or more data splits not found.")
        return
        
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Loaded: Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)})")
    
    # 1. Identity Overlap Check
    train_ids = set(df_train['video_path'].apply(extract_identity).dropna())
    val_ids = set(df_val['video_path'].apply(extract_identity).dropna())
    test_ids = set(df_test['video_path'].apply(extract_identity).dropna())
    
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)
    
    print("\n--- IDENTITY LEAKAGE AUDIT ---")
    if len(overlap_train_test) > 0:
        print(f"CRITICAL LEAKAGE: {len(overlap_train_test)} identities found in both Train and Test.")
        print(f"Leaked IDs: {overlap_train_test}")
    else:
        print("PASS: No identity overlap between Train and Test.")
        
    if len(overlap_val_test) > 0:
        print(f"CRITICAL LEAKAGE: {len(overlap_val_test)} identities found in both Val and Test.")
    else:
        print("PASS: No identity overlap between Val and Test.")
        
    # 2. Duplicate File Check
    train_files = set(df_train['video_path'])
    test_files = set(df_test['video_path'])
    
    file_overlap = train_files.intersection(test_files)
    print("\n--- FILE LEAKAGE AUDIT ---")
    if len(file_overlap) > 0:
        print(f"CRITICAL LEAKAGE: {len(file_overlap)} exact file paths found in both Train and Test.")
    else:
        print("PASS: No exact file path overlap between Train and Test.")

    os.makedirs("reports", exist_ok=True)
    with open("reports/leakage_audit_report.txt", "w") as f:
        f.write("--- IDENTITY LEAKAGE AUDIT ---\n")
        f.write(f"Overlap Train/Test: {len(overlap_train_test)}\n")
        f.write(f"Overlap Val/Test: {len(overlap_val_test)}\n")
        f.write("--- FILE LEAKAGE AUDIT ---\n")
        f.write(f"Overlap Train/Test Files: {len(file_overlap)}\n")

if __name__ == "__main__":
    audit_leakage()
