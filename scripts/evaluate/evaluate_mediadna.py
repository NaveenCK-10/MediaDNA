import os
import sys
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from backend.inference import OpenAVFFService
from backend.modules import analyze_visual_signals
from backend.modules.visual import BLUR_CV_THRESHOLD, FRAME_MAE_STD_THRESHOLD
from final_baseline_eval import load_excluded_paths, get_test_videos

def compute_detailed_metrics(df, label_col="true_label", pred_col="prediction", prob_col="fake_prob"):
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
        
    metrics_by_cat["Global"] = calc_metrics(df[label_col], df[pred_col], df[prob_col])
    
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        metrics_by_cat[cat] = calc_metrics(cat_df[label_col], cat_df[pred_col], cat_df[prob_col])
        
    return metrics_by_cat

def calc_anomaly(value, threshold):
    return min(1.0, max(0.0, (value - (threshold * 0.5)) / (threshold * 0.5))) if threshold > 0 else 0.0

def main():
    os.makedirs("reports", exist_ok=True)
    
    excluded = load_excluded_paths()
    videos = get_test_videos(excluded)
    
    ckpt_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\exp\stage-3-local\models\best_audio_model.pth"
    service = OpenAVFFService(checkpoint_path=ckpt_path)
    
    results = []
    
    print(f"Starting MediaDNA Evaluation on {len(videos)} videos...")
    for i, v in enumerate(videos):
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(videos)}...")
            
        try:
            # 1. OpenAVFF Baseline
            res = service.analyze_video(v["path"])
            
            # 2. Visual Module (returns raw + normalized overall)
            visual = analyze_visual_signals(v["path"])
            
            blur_cv = visual["blur_coefficient_of_variation"]
            diff_std = visual["frame_mae_std"]
            
            # Recompute individual normalized anomalies for ablation
            blur_anomaly = calc_anomaly(blur_cv, BLUR_CV_THRESHOLD)
            temporal_anomaly = calc_anomaly(diff_std, FRAME_MAE_STD_THRESHOLD)
            
            # Fusions
            openavff_prob = res.fake_probability
            
            prob_blur = (openavff_prob * 0.8) + (blur_anomaly * 0.2)
            prob_temporal = (openavff_prob * 0.8) + (temporal_anomaly * 0.2)
            prob_full = (openavff_prob * 0.8) + (visual["visual_anomaly_score"] * 0.2)
            
            results.append({
                "video": os.path.basename(v["path"]),
                "category": v["category"],
                "true_label": v["label"],
                
                # Baseline
                "baseline_pred": 1 if openavff_prob >= 0.5 else 0,
                "baseline_prob": openavff_prob,
                
                # Blur only
                "blur_cv": blur_cv,
                "blur_anomaly": blur_anomaly,
                "blur_fusion_pred": 1 if prob_blur >= 0.5 else 0,
                "blur_fusion_prob": max(0.0, min(1.0, prob_blur)),
                
                # Temporal only
                "temporal_std": diff_std,
                "temporal_anomaly": temporal_anomaly,
                "temporal_fusion_pred": 1 if prob_temporal >= 0.5 else 0,
                "temporal_fusion_prob": max(0.0, min(1.0, prob_temporal)),
                
                # MediaDNA Full Fusion
                "mediadna_pred": 1 if prob_full >= 0.5 else 0,
                "mediadna_prob": max(0.0, min(1.0, prob_full))
            })
        except Exception as e:
            print(f"Error on {v['path']}: {e}")
            
    df = pd.DataFrame(results)
    df.to_csv(r"reports\mediDNA_experiment_results.csv", index=False)
    
    # Compute ablation metrics
    baseline_metrics = compute_detailed_metrics(df, pred_col="baseline_pred", prob_col="baseline_prob")
    blur_metrics = compute_detailed_metrics(df, pred_col="blur_fusion_pred", prob_col="blur_fusion_prob")
    temporal_metrics = compute_detailed_metrics(df, pred_col="temporal_fusion_pred", prob_col="temporal_fusion_prob")
    fusion_metrics = compute_detailed_metrics(df, pred_col="mediadna_pred", prob_col="mediadna_prob")
    
    # Generate Ablation CSV
    ablation_data = [
        {"Model": "OpenAVFF Baseline", "Accuracy": baseline_metrics["Global"]["Accuracy"], "FakeVideo-RealAudio": baseline_metrics.get("FakeVideo-RealAudio", {}).get("Accuracy", 0)},
        {"Model": "OpenAVFF + Blur", "Accuracy": blur_metrics["Global"]["Accuracy"], "FakeVideo-RealAudio": blur_metrics.get("FakeVideo-RealAudio", {}).get("Accuracy", 0)},
        {"Model": "OpenAVFF + Temporal", "Accuracy": temporal_metrics["Global"]["Accuracy"], "FakeVideo-RealAudio": temporal_metrics.get("FakeVideo-RealAudio", {}).get("Accuracy", 0)},
        {"Model": "MediaDNA Full", "Accuracy": fusion_metrics["Global"]["Accuracy"], "FakeVideo-RealAudio": fusion_metrics.get("FakeVideo-RealAudio", {}).get("Accuracy", 0)}
    ]
    pd.DataFrame(ablation_data).to_csv(r"reports\mediDNA_ablation.csv", index=False)
    
    # Generate Research Report
    with open(r"reports\mediDNA_research_report.md", 'w') as report_file:
        report_file.write("# MediaDNA Research Report\n\n")
        report_file.write("## 1. Experimental Setup\n")
        report_file.write("- **Test Set**: 303 strictly held-out FakeAVCeleb videos\n")
        report_file.write("- **Baseline**: OpenAVFF `best_audio_model.pth`\n")
        report_file.write("- **Visual Features**: Laplacian variance (Blur), MAE (Temporal)\n")
        report_file.write("- **Fusion Strategy**: 80% OpenAVFF, 20% Visual Anomaly (Experimental Baseline)\n\n")
        
        report_file.write("## 2. Global Results (Ablation)\n")
        report_file.write("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n")
        report_file.write("|-------|----------|-----------|--------|----|---------|\n")
        
        for name, metrics in [
            ("OpenAVFF Baseline", baseline_metrics["Global"]),
            ("OpenAVFF + Blur", blur_metrics["Global"]),
            ("OpenAVFF + Temporal", temporal_metrics["Global"]),
            ("MediaDNA Fusion", fusion_metrics["Global"])
        ]:
            report_file.write(f"| {name} | {metrics['Accuracy']:.4f} | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1']:.4f} | {metrics['ROC-AUC']:.4f} |\n")
            
        report_file.write("\n## 3. Category Breakdown: Baseline vs MediaDNA\n")
        report_file.write("| Category | Baseline Acc | MediaDNA Acc | Baseline Recall | MediaDNA Recall |\n")
        report_file.write("|----------|--------------|--------------|-----------------|-----------------|\n")
        
        for cat in ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
            if cat in baseline_metrics:
                b = baseline_metrics[cat]
                f = fusion_metrics[cat]
                report_file.write(f"| {cat} | {b['Accuracy']:.4f} | {f['Accuracy']:.4f} | {b['Recall']:.4f} | {f['Recall']:.4f} |\n")
                
        report_file.write("\n## 4. Conclusion and Blindspot Analysis\n")
        acc_diff = fusion_metrics["Global"]['Accuracy'] - baseline_metrics["Global"]['Accuracy']
        
        report_file.write(f"The MediaDNA fusion method changed overall accuracy by {acc_diff:+.1%}. ")
        report_file.write("We observed the following effects on the FakeVideo-RealAudio blindspot: ")
        
        blindspot_diff = fusion_metrics.get("FakeVideo-RealAudio", {}).get("Accuracy", 0) - baseline_metrics.get("FakeVideo-RealAudio", {}).get("Accuracy", 0)
        
        if blindspot_diff > 0:
            report_file.write(f"The fusion successfully improved detection on the blindspot by {blindspot_diff:+.1%}. ")
        else:
            report_file.write(f"The fusion did not improve the blindspot (change: {blindspot_diff:+.1%}). ")
            
        report_file.write("The experimental 80/20 weighting scheme provides a transparent baseline, but learning an adaptive fusion or decision tree using these visual anomaly signals may be required for robust production use.")
            
    print("\nMediaDNA Evaluation complete! Results saved to reports/mediDNA_experiment_results.csv, reports/mediDNA_ablation.csv, and reports/mediDNA_research_report.md")

if __name__ == "__main__":
    main()
