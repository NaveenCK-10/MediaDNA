import os
import requests
import time
import shutil

BASE_URL = "http://localhost:8000"

def create_corrupt_file():
    path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\scratch\corrupt.mp4"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("This is not a video file.")
    return path

def create_no_audio_file():
    path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\scratch\no_audio.mp4"
    src = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\robustness_suite_v14\00143_clean.mp4"
    ffmpeg = r"C:\Users\navee\Downloads\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"
    os.system(f'"{ffmpeg}" -y -i "{src}" -an -vcodec copy "{path}" > NUL 2>&1')
    return path

test_files = [
    create_no_audio_file(),
    create_corrupt_file()
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
