import json
import os
import pandas as pd

def generate_report():
    candidates = ["Cand_A", "Cand_B", "Cand_C", "Cand_D"]
    noise_profiles = ["Clean", "Mild", "Moderate", "Strong"]
    
    threshold = '0.60'
    
    rows = []
    
    for cand in candidates:
        for noise in noise_profiles:
            exp_name = f"Phase3_{cand}_{noise}"
            json_file = f"results/{exp_name}.json"
            
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    
                # The data structure has metrics and confusion matrix at the threshold
                metrics = data['metrics']
                
                # Check if threshold 0.60 exists
                if threshold in metrics:
                    t_metrics = metrics[threshold]
                    
                    row = {
                        "Candidate": cand.replace('Cand_', ''),
                        "Noise": noise,
                        "ROC-AUC": f"{data.get('auc', 0):.4f}",
                        "PR-AUC": f"{data.get('ap', 0):.4f}",
                        "Accuracy": f"{t_metrics.get('acc', 0):.4f}",
                        "Balanced Acc": f"{t_metrics.get('balanced_acc', 0):.4f}",
                        "F1": f"{t_metrics.get('f1', 0):.4f}",
                        "Precision": f"{t_metrics.get('precision', 0):.4f}",
                        "Recall": f"{t_metrics.get('recall', 0):.4f}",
                        "FPR": f"{t_metrics.get('fpr', 0):.4f}",
                        "FNR": f"{t_metrics.get('fnr', 0):.4f}",
                        "TN": t_metrics.get('cm_tn', 0),
                        "FP": t_metrics.get('cm_fp', 0),
                        "FN": t_metrics.get('cm_fn', 0),
                        "TP": t_metrics.get('cm_tp', 0)
                    }
                    rows.append(row)
    
    df = pd.DataFrame(rows)
    
    report_md = "# V16.1 Phase 3 Candidate Evaluation Report\n\n"
    report_md += "> [!IMPORTANT]\n> This report compares the four V16.1 candidates across all validation noise profiles. Threshold is locked at **0.60**.\n\n"
    
    for cand in df['Candidate'].unique():
        report_md += f"## Candidate {cand}\n\n"
        cand_df = df[df['Candidate'] == cand].drop(columns=['Candidate'])
        report_md += cand_df.to_markdown(index=False) + "\n\n"
        
    report_md += "## Summary of Findings\n"
    report_md += "- Review the ROC-AUC and Balanced Accuracy drops across noise levels.\n"
    
    with open("reports/V16_1_PHASE3_COMPARISON_REPORT.md", "w") as f:
        f.write(report_md)
        
    print("Report written to reports/V16_1_PHASE3_COMPARISON_REPORT.md")

if __name__ == "__main__":
    generate_report()
