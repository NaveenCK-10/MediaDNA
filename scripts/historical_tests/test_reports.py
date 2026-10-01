import os
import requests
import time
import json

BASE_URL = "http://localhost:8000"

test_files = [
    r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00143_clean.mp4",
    r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00160_id01098_wavtolip_clean.mp4"
]

def analyze_video(filepath):
    print(f"\nAnalyzing: {os.path.basename(filepath)}")
    with open(filepath, "rb") as f:
        res = requests.post(f"{BASE_URL}/api/analyze", files={"video": f})
    if res.status_code != 200:
        print(f"Failed to submit: {res.text}")
        return None
        
    job_id = res.json()["job_id"]
    print(f"Job ID: {job_id}")
    
    # Wait for completion
    while True:
        res = requests.get(f"{BASE_URL}/api/history")
        history = res.json()
        job = next((j for j in history if j.get("id") == job_id), None)
        if job and job.get("status") in ["COMPLETED", "FAILED"]:
            return job_id
        time.sleep(2)
        
def generate_report(job_id):
    print(f"Generating report for {job_id}")
    res = requests.post(f"{BASE_URL}/api/report/{job_id}")
    if res.status_code != 200:
        print(f"Report generation failed: {res.text}")
        return False
    return True
    
def validate_pdf(job_id):
    pdf_path = os.path.join(r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\experiments\final_demo\reports", f"MediaDNA_Forensic_Report_{job_id}.pdf")
    
    if not os.path.exists(pdf_path):
        print("PDF file does not exist.")
        return False
        
    size = os.path.getsize(pdf_path)
    print(f"PDF Size: {size} bytes")
    if size < 100:
        print("PDF too small.")
        return False
        
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            if not header.startswith(b"%PDF-"):
                print("Missing '%PDF-' header.")
                return False
                
            print("PDF Validation PASSED.")
            return True
    except Exception as e:
        print(f"PDF parsing error: {e}")
        return False

for f in test_files:
    jid = analyze_video(f)
    if jid:
        if generate_report(jid):
            validate_pdf(jid)
