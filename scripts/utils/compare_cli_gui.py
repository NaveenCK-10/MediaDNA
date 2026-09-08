import os
import sys
import pandas as pd
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService

from final_baseline_eval import load_excluded_paths, get_test_videos

def main():
    print("Testing CLI vs GUI Consistency...")
    
    ckpt_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\exp\stage-3-local\models\best_audio_model.pth"
    base_dir = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
    
    excluded = load_excluded_paths()
    all_videos = get_test_videos(excluded)
    
    reals = [v for v in all_videos if v["label"] == 0][:2]
    fakes = [v for v in all_videos if v["label"] == 1][:2]
    
    videos = [
        {"path": reals[0]["path"], "type": "Real"},
        {"path": reals[1]["path"], "type": "Real"},
        {"path": fakes[0]["path"], "type": "Fake"},
        {"path": fakes[1]["path"], "type": "Fake"},
    ]
    
    # Create CSV for eval.py
    csv_path = "cli_test.csv"
    with open(csv_path, "w") as f:
        f.write("video_id,label\n")
        for v in videos:
            rel_path = os.path.relpath(v["path"], base_dir).replace('\\', '/')
            label = 1 if v["type"] == "Fake" else 0
            f.write(f"{rel_path},{label}\n")
            
    # Run CLI
    print("Running CLI inference (eval.py)...")
    cmd = ["python", "eval.py", "--checkpoint", ckpt_path, "--csv_file", csv_path]
    subprocess.run(cmd, check=True)
    
    # Load CLI predictions
    cli_preds = {}
    with open("prediction.csv", "r") as f:
        lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split(',')
            cli_preds[parts[0]] = float(parts[1])
            
    # Run GUI
    print("Running GUI inference (OpenAVFFService)...")
    service = OpenAVFFService(checkpoint_path=ckpt_path)
    
    results = []
    
    for v in videos:
        vid_name = os.path.basename(v["path"])
        print(f"Testing {vid_name}...")
        
        # GUI
        res = service.analyze_video(v["path"])
        gui_prob = res.fake_probability
        
        # CLI
        cli_prob = cli_preds.get(vid_name, -1.0)
            
        diff = abs(gui_prob - cli_prob)
        consistent = diff <= 0.001
        
        results.append({
            "Video": vid_name,
            "Type": v["type"],
            "GUI_Fake_Prob": gui_prob,
            "CLI_Fake_Prob": cli_prob,
            "Absolute_Difference": diff,
            "Consistent": consistent
        })
        
    df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    df.to_csv(r"reports\cli_vs_gui.csv", index=False)
    
    print("\nConsistency Results:")
    print(df)
    
    if all(df["Consistent"]):
        print("PASS: GUI and CLI match within 0.001 tolerance.")
    else:
        print("FAIL: Mismatch found!")

if __name__ == "__main__":
    main()
