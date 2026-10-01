import sys
import os
import json
import asyncio

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import OpenAVFFService
from backend.modules.report_generator.generator import ReportGenerator

def generate_report_for_video(video_path: str, output_path: str):
    print(f"Loading OpenAVFFService to analyze {video_path}...")
    service = OpenAVFFService()
    profile = service.analyze_video(video_path)
    
    print("Generating Forensic Report PDF...")
    generator = ReportGenerator()
    pdf_path = generator.generate_pdf(profile, output_path)
    print(f"Report generated successfully: {pdf_path}")
    return pdf_path

if __name__ == "__main__":
    test_video = os.path.join(PROJECT_ROOT, "data", "raw", "FakeAVCeleb", "FakeVideo-FakeAudio", "00109_10_id00476_wavtolip.mp4")
    
    if not os.path.exists(test_video):
        print("Test video not found. Checking if any paired_demo video is available...")
        # Fallback to another video
        import csv
        with open(os.path.join(PROJECT_ROOT, "data", "paired_demo.csv"), "r") as f:
            reader = list(csv.DictReader(f))
            test_video = os.path.join(PROJECT_ROOT, reader[0]['video_path'])
            
    out_dir = os.path.join(PROJECT_ROOT, "experiments", "final_demo", "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "MediaDNA_Forensic_Report_TEST.pdf")
    
    generate_report_for_video(test_video, out_file)
