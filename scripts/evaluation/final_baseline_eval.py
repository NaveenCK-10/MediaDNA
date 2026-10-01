import os
import sys
import csv
import json
import random
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService

def load_excluded_paths():
    """Load all video paths from training and validation CSVs to avoid data leakage."""
    excluded = set()
    csv_files = glob.glob(r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\train*.csv") + \
                glob.glob(r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\data\val*.csv")
                
    for file in csv_files:
        try:
            with open(file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and len(row) >= 1:
                        # Extract just the filename to be safe, or relative path
                        # The dataset usually stores relative paths like FakeVideo-FakeAudio/African/men/...
                        path = row[0].strip()
                        excluded.add(path)
                        # Also add just the basename for extra safety
                        excluded.add(os.path.basename(path))
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    return excluded

def get_test_videos(excluded):
    base_dir = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2"
    
    categories = {
        "RealVideo-RealAudio": {"label": 0, "paths": []},
        "RealVideo-FakeAudio": {"label": 1, "paths": []},
        "FakeVideo-RealAudio": {"label": 1, "paths": []},
        "FakeVideo-FakeAudio": {"label": 1, "paths": []}
    }
    
    for cat, info in categories.items():
        cat_dir = os.path.join(base_dir, cat)
        print(f"Scanning {cat}...")
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if file.endswith(".mp4"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir).replace('\\', '/')
                    basename = os.path.basename(file)
                    
                    if rel_path not in excluded and basename not in excluded:
                        info["paths"].append(full_path)
        
        print(f"  Found {len(info['paths'])} clean videos in {cat}")
        
    # Sample 125 from each category to get 500 total, or as many as possible
    selected_videos = []
    
    for cat, info in categories.items():
        # Set seed for reproducibility
        random.seed(42)
        sample_size = min(125, len(info["paths"]))
        sampled = random.sample(info["paths"], sample_size)
        
        for path in sampled:
            selected_videos.append({
                "path": path,
                "label": info["label"],
                "category": cat
            })
            
    return selected_videos

def evaluate_test_set(videos):
    ckpt_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\exp\stage-3-local\models\best_audio_model.pth"
    print(f"\nInitializing model from: {ckpt_path}")
    service = OpenAVFFService(checkpoint_path=ckpt_path)
    
    results = []
    
    print(f"Starting evaluation of {len(videos)} videos...")
    for i, v in enumerate(videos):
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(videos)}...")
            
        try:
            res = service.analyze_video(v["path"])
            
            results.append({
                "video": os.path.basename(v["path"]),
                "category": v["category"],
                "true_label": v["label"],
                "prediction": 1 if res.prediction == 'fake' else 0,
                "fake_prob": res.fake_probability,
                "real_prob": res.real_probability
            })
        except Exception as e:
            print(f"Error on {v['path']}: {e}")
            
    return results

def compute_metrics(results):
    df = pd.DataFrame(results)
    
    metrics_by_cat = {}
    
    def calc_metrics(y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                auc = float('nan')
        except:
            auc = float('nan')
            
        return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc}
        
    # Global metrics
    global_metrics = calc_metrics(df["true_label"], df["prediction"], df["fake_prob"])
    metrics_by_cat["Global"] = global_metrics
    
    # Per-category metrics
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        metrics_by_cat[cat] = calc_metrics(cat_df["true_label"], cat_df["prediction"], cat_df["fake_prob"])
        
    return df, metrics_by_cat

def plot_confusion_matrix(df, out_path):
    cm = confusion_matrix(df["true_label"], df["prediction"], labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Real', 'Predicted Fake'],
                yticklabels=['Actual Real', 'Actual Fake'])
    plt.title('Confusion Matrix - OpenAVFF Final Baseline')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_markdown_report(df, metrics_by_cat, out_path):
    with open(out_path, 'w') as f:
        f.write("# Final Baseline Evaluation Report - MediaDNA / OpenAVFF\n\n")
        
        f.write("## Overview\n")
        f.write("- **Model**: VideoCAVMAEFT\n")
        f.write("- **Checkpoint**: `exp/stage-3-local/models/best_audio_model.pth`\n")
        f.write(f"- **Test Set Size**: {len(df)} videos\n")
        f.write("- **Leakage Prevention**: Videos from train/val splits were explicitly excluded.\n\n")
        
        f.write("## Global Metrics\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for k, v in metrics_by_cat["Global"].items():
            f.write(f"| {k} | {v:.4f} |\n")
            
        f.write("\n## Metrics by Category\n")
        f.write("| Category | Count | Accuracy | Precision | Recall | F1 | ROC-AUC |\n")
        f.write("|----------|-------|----------|-----------|--------|----|---------|\n")
        
        for cat in ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
            if cat in metrics_by_cat:
                m = metrics_by_cat[cat]
                count = len(df[df["category"] == cat])
                auc_str = f"{m['ROC-AUC']:.4f}" if not pd.isna(m['ROC-AUC']) else "N/A"
                f.write(f"| {cat} | {count} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1']:.4f} | {auc_str} |\n")
                
        f.write("\n## Confusion Matrix\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        f.write("## Conclusion\n")
        acc = metrics_by_cat["Global"]["Accuracy"]
        if acc > 0.8:
            f.write(f"Based on the evaluation, the current OpenAVFF model demonstrates strong performance (Accuracy: {acc:.1%}) and is **SUITABLE** as the frozen BASELINE for the MediaDNA research project. The evaluation utilized a clean held-out test set and matched the exact GUI preprocessing pipeline.\n")
        else:
            f.write(f"Based on the evaluation, the current OpenAVFF model has limited performance (Accuracy: {acc:.1%}). It may be used as a baseline for comparison, but significant improvements will be needed in subsequent research stages.\n")

def main():
    os.makedirs("reports", exist_ok=True)
    
    # 1. Load exclusions
    excluded = load_excluded_paths()
    print(f"Loaded {len(excluded)} excluded paths to prevent data leakage.")
    
    # 2. Get balanced test set
    videos = get_test_videos(excluded)
    print(f"Selected {len(videos)} videos for baseline evaluation.")
    
    # 3. Run evaluation
    results = evaluate_test_set(videos)
    
    # 4. Compute metrics
    df, metrics_by_cat = compute_metrics(results)
    
    # 5. Save outputs
    df.to_csv(r"reports\final_baseline_evaluation.csv", index=False)
    plot_confusion_matrix(df, r"reports\confusion_matrix.png")
    generate_markdown_report(df, metrics_by_cat, r"reports\final_baseline_report.md")
    
    print("\nEvaluation complete! Results saved to reports/ directory.")
    
if __name__ == "__main__":
    main()
