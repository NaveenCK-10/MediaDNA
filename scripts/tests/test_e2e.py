import os
import time
import requests
import sys

def test_api():
    print("Testing MediaDNA End-to-End API...")
    
    # 1. Health check
    try:
        res = requests.get("http://localhost:8000/api/health")
        assert res.status_code == 200
        print("[PASS] Backend Health Check")
    except Exception as e:
        print(f"[FAIL] Backend Health Check: {e}")
        return
        
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from final_baseline_eval import load_excluded_paths, get_test_videos
    
    try:
        excluded = load_excluded_paths()
        all_videos = get_test_videos(excluded)
    except Exception as e:
        print(f"[FAIL] Failed to load videos: {e}")
        return
        
    fake_video = None
    real_video = None
    fakes = [v for v in all_videos if v["label"] == 1]
    if fakes: fake_video = fakes[0]["path"]
    reals = [v for v in all_videos if v["label"] == 0]
    if reals: real_video = reals[0]["path"]
    
    if not fake_video or not os.path.exists(fake_video):
        print("[FAIL] Fake video not found on disk")
        return
    
    # 2. Upload Fake Video
    try:
        with open(fake_video, 'rb') as f:
            res = requests.post("http://localhost:8000/api/analyze", files={'video': f})
        assert res.status_code == 200
        data = res.json()
        assert 'prediction' in data
        assert 'visual_signals' in data
        assert 'metadata' in data
        print(f"[PASS] Fake Video Upload & Analysis (Prediction: {data['prediction']})")
    except Exception as e:
        print(f"[FAIL] Fake Video Upload: {e}")
        
    # 3. Upload Real Video
    try:
        with open(real_video, 'rb') as f:
            res = requests.post("http://localhost:8000/api/analyze", files={'video': f})
        assert res.status_code == 200
        data = res.json()
        print(f"[PASS] Real Video Upload & Analysis (Prediction: {data['prediction']})")
    except Exception as e:
        print(f"[FAIL] Real Video Upload: {e}")
        
    # 4. History check
    try:
        res = requests.get("http://localhost:8000/api/history")
        assert res.status_code == 200
        assert len(res.json()) >= 2
        print("[PASS] History retrieval")
    except Exception as e:
        print(f"[FAIL] History Check: {e}")
        
    # 5. Invalid video (text file)
    try:
        with open("backend/main.py", 'rb') as f:
            res = requests.post("http://localhost:8000/api/analyze", files={'video': f})
        assert res.status_code in [400, 422, 500]  # Should error
        print(f"[PASS] Invalid Video Rejection (Status: {res.status_code})")
    except Exception as e:
        print(f"[FAIL] Invalid Video Check: {e}")

if __name__ == "__main__":
    test_api()
