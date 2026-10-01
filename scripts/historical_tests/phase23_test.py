import os
import requests
import time
import json

BASE_URL = "http://localhost:8000"

videos = {
    "Exact Video": r"C:\Users\navee\Downloads\Create_a_photorealistic_AI_gen.mp4",
    "Known Real 1": r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00143_clean.mp4",
    "Known Synthetic 1": r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00160_id01098_wavtolip_clean.mp4",
    "No Audio": r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\scratch\no_audio.mp4",
    "Corrupt": r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\scratch\corrupt.mp4"
}

results = []

for name, path in videos.items():
    print(f"\n========================================")
    print(f"TESTING: {name}")
    print(f"========================================")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
        
    with open(path, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"video": f})
        
    if res.status_code != 200:
        print(f"API Error: {res.text}")
        continue
        
    job_id = res.json()["job_id"]
    print(f"Job ID: {job_id}")
    
    job = None
    while True:
        res = requests.get(f"{BASE_URL}/api/history")
        history = res.json()
        job = next((j for j in history if j.get("id") == job_id), None)
        if job and job.get("status") in ["COMPLETED", "FAILED"]:
            break
        time.sleep(1)
        
    print(f"Status: {job.get('status')}")
    
    if job.get("status") == "COMPLETED":
        v_score = job.get("visual", {}).get("raw_model_score")
        a_score = job.get("audio", {}).get("raw_model_score")
        f_score = job.get("fusion", {}).get("raw_model_score")
        c_score = job.get("trust", {}).get("calibrated_probability")
        decision = job.get("trust", {}).get("abstention_state")
        
        print(f"Visual Score: {v_score}")
        print(f"Audio Score: {a_score}")
        print(f"Fusion Score: {f_score}")
        print(f"Calibrated Prob: {c_score}")
        print(f"Decision: {decision}")
    else:
        print(f"Error: {job.get('error_message')}")
        
    # Test PDF Gen
    print("Generating PDF...")
    res = requests.post(f"{BASE_URL}/api/report/{job_id}")
    if res.status_code == 200:
        print("PDF Generated successfully.")
        
        # Test Download
        res = requests.get(f"{BASE_URL}/api/report/{job_id}?download=true")
        if res.status_code == 200 and res.headers.get("content-type") == "application/pdf":
            print("PDF Download successful.")
        else:
            print("PDF Download failed.")
    else:
        print(f"PDF Generation failed: {res.text}")

