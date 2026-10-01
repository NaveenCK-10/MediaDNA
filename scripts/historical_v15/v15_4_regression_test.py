import os
import sys
import time
import requests
from typing import List

# FastAPI server URL
API_URL = "http://127.0.0.1:8000/api/analyze"

def analyze_video(video_path: str):
    print(f"\nAnalyzing: {video_path}")
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return None
        
    start_time = time.time()
    with open(video_path, 'rb') as f:
        files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
        try:
            response = requests.post(API_URL, files=files, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            elapsed = time.time() - start_time
            print(f"Success in {elapsed:.2f}s")
            print(f"  Final Prediction:  {result['prediction']}")
            print(f"  MediaDNA P(Fake):  {result['fake_probability']:.4f}")
            print(f"  OpenAVFF P(Fake):  {result['openavff_fake_prob']:.4f}")
            return result
        except Exception as e:
            print(f"API Error: {e}")
            if hasattr(response, 'text'):
                print(f"Response: {response.text}")
            return None

def main():
    print("==================================================")
    print("V15.4 REGRESSION AND REPEATABILITY TEST")
    print("==================================================")
    
    naveen_path = "src/naveen.mp4"
    fake_path = "FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.mp4"
    real_path = "FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id00076/00109.mp4"

    print("\n--- TEST 1: REPEATABILITY ON NAVEEN.MP4 ---")
    results = []
    for i in range(3):
        print(f"\nRun {i+1}:")
        res = analyze_video(naveen_path)
        if res:
            results.append(res['openavff_fake_prob'])
            
    if len(results) == 3:
        if results[0] == results[1] == results[2]:
            print("\nREPEATABILITY PASSED: Probabilities are exactly identical across runs.")
        else:
            print("\nREPEATABILITY FAILED: Probabilities fluctuated!")
            print(results)
            
    print("\n--- TEST 2: GENUINE MEDIA (REAL) ---")
    analyze_video(real_path)
    
    print("\n--- TEST 3: AUTHORIZED FAKE MEDIA (FAKE) ---")
    analyze_video(fake_path)
    
if __name__ == "__main__":
    main()
