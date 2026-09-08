import os
import csv
import pandas as pd
from collections import defaultdict
import re

def extract_all_identities(path):
    return set(re.findall(r'(id\d+)', str(path)))

def deep_audit():
    train_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\train_v8.csv"
    val_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\val_v8.csv"
    test_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\test_locked_v8.csv"
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    train_ids = set()
    val_ids = set()
    test_ids = set()
    
    for _, row in df_train.iterrows(): train_ids.update(extract_all_identities(row['video_path']))
    for _, row in df_val.iterrows(): val_ids.update(extract_all_identities(row['video_path']))
    for _, row in df_test.iterrows(): test_ids.update(extract_all_identities(row['video_path']))
    
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)
    
    print("\n--- DEEP IDENTITY LEAKAGE AUDIT (ALL IDENTITIES IN PATH) ---")
    if len(overlap_train_test) > 0:
        print(f"CRITICAL LEAKAGE: {len(overlap_train_test)} true identities found in both Train and Test.")
    else:
        print("PASS: No identity overlap between Train and Test.")

if __name__ == "__main__":
    deep_audit()
