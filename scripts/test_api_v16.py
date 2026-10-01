import requests
import json
import time
import os
import GPUtil

# Paths
REAL_VIDEO = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\RealVideo-RealAudio\African\men\id00076\00109.mp4"
FAKE_VIDEO = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\FakeVideo-FakeAudio\African\men\id00076\00109_10_id00476_wavtolip.mp4"
API_URL = "http://127.0.0.1:8000/api/analyze"

def measure_vram():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].memoryUsed
    return 0

def test_video(path, label):
    print(f"\n--- Testing {label} Video ---")
    start_vram = measure_vram()
    start_time = time.time()
    
    with open(path, 'rb') as f:
        files = {'video': (os.path.basename(path), f, 'video/mp4')}
        try:
            response = requests.post(API_URL, files=files)
            response.raise_for_status()
            result = response.json()
        except Exception as e:
            print(f"Error: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(response.text)
            return None
            
    latency = time.time() - start_time
    end_vram = measure_vram()
    
    print(f"Latency: {latency:.2f}s")
    print(f"VRAM used: {end_vram} MB (delta: {end_vram - start_vram} MB)")
    
    return {
        "label": label,
        "latency": latency,
        "vram": end_vram,
        "result": result
    }

if __name__ == "__main__":
    print("Waiting for server to fully initialize...")
    time.sleep(5)
    
    health = requests.get("http://127.0.0.1:8000/api/health")
    print("Health:", health.json())
    
    real_results = test_video(REAL_VIDEO, "Genuine")
    fake_results = test_video(FAKE_VIDEO, "Same-Source Manipulated")
    
    out_dir = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\experiments\final_demo"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "demo_results.json"), "w") as f:
        json.dump({"real": real_results, "fake": fake_results}, f, indent=2)
        
    print(f"\nResults saved to {out_dir}/demo_results.json")
