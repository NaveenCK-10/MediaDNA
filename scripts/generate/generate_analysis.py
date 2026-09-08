import pandas as pd
import os

def generate_analysis():
    csv_path = "v16_experiments/V16_EXPERIMENT_RESULTS.csv"
    df = pd.read_csv(csv_path, names=["timestamp", "exp_name", "checkpoint", "dataset", "mode", "noise_level", "threshold", "roc_auc", "pr_auc", "acc", "bal_acc", "f1", "prec", "fpr", "fnr", "real_mean", "fake_mean"])
    
    # Filter for Phase3
    df = df[df['exp_name'].str.startswith('Phase3_')]
    
    # Parse candidate and noise
    df['Candidate'] = df['exp_name'].apply(lambda x: x.split('_')[1] + '_' + x.split('_')[2])
    df['Noise'] = df['exp_name'].apply(lambda x: x.split('_')[-1])
    
    # Calculate CM
    df['fpr'] = df['fpr'].astype(float)
    df['fnr'] = df['fnr'].astype(float)
    df['FP'] = (df['fpr'] * 50).round().astype(int)
    df['FN'] = (df['fnr'] * 150).round().astype(int)
    df['TN'] = 50 - df['FP']
    df['TP'] = 150 - df['FN']
    
    # Calculate Recall
    df['recall'] = df['TP'] / 150
    
    # Reorder columns
    cols = ['Candidate', 'Noise', 'roc_auc', 'pr_auc', 'acc', 'bal_acc', 'f1', 'prec', 'recall', 'fpr', 'fnr', 'TN', 'FP', 'FN', 'TP', 'real_mean', 'fake_mean']
    df = df[cols]
    
    # Format float
    for c in ['roc_auc', 'pr_auc', 'acc', 'bal_acc', 'f1', 'prec', 'recall', 'fpr', 'fnr', 'real_mean', 'fake_mean']:
        df[c] = df[c].astype(float)
        df[c] = df[c].apply(lambda x: f"{x:.4f}")
        
    md = "# V16.1 Phase 3 Robustness Analysis & Candidate Selection\n\n"
    md += "## Candidate Performance Comparison\n\n"
    
    for cand in df['Candidate'].unique():
        md += f"### {cand}\n\n"
        cand_df = df[df['Candidate'] == cand].drop(columns=['Candidate'])
        md += cand_df.to_markdown(index=False) + "\n\n"
        
    md += """## Robustness Analysis

### A. Clean Performance
- **Cand_A_Clean (V15.4 Baseline)**: ROC-AUC = 0.9145, Accuracy = 0.8700, FPR = 0.0400, FNR = 0.1600.
- **Cand_B_Mild**: Slightly better clean ROC-AUC (0.9216) but worse FPR (0.2800) and much worse Balanced Accuracy (0.8167).
- **Cand_C_Moderate**: Better clean ROC-AUC (0.9263), lower FPR (0.0200). Accuracy is slightly lower (0.8650) with similar Balanced Acc (0.9033).
- **Cand_D_Mixed**: Clean ROC-AUC is lower (0.8949). Accuracy is 0.8600. FPR is extremely high (0.2800).

### B. Mild-Noise Performance
- **Cand_A_Clean**: ROC-AUC = 0.9126. FPR drops to 0.0200.
- **Cand_B_Mild**: ROC-AUC = 0.9189. FPR explodes to 1.0000. Catastrophic real-video failure.
- **Cand_C_Moderate**: ROC-AUC = 0.9161. FPR is excellent (0.0200). Accuracy drops significantly due to FNR = 0.4067.
- **Cand_D_Mixed**: ROC-AUC = 0.9223. FPR explodes to 0.8000. Catastrophic real-video failure.

### C. Moderate-Noise Performance
- **Cand_A_Clean**: ROC-AUC = 0.8910. FPR spikes to 0.3200.
- **Cand_B_Mild**: ROC-AUC = 0.9124. FPR is 1.0000.
- **Cand_C_Moderate**: ROC-AUC = 0.8751. FPR is 0.0000 (perfect real retention), but FNR spikes to 0.9000.
- **Cand_D_Mixed**: ROC-AUC = 0.8966. FPR is 1.0000.

### D. Strong-Noise Performance
- **Cand_A_Clean**: ROC-AUC = 0.7528. FPR = 1.0000.
- **Cand_B_Mild**: ROC-AUC = 0.8257. FPR = 1.0000.
- **Cand_C_Moderate**: ROC-AUC = 0.7679. FPR = 0.0600. FNR is very high (0.7400).
- **Cand_D_Mixed**: ROC-AUC = 0.8213. FPR = 1.0000.

### E. Conclusion on Degradation & False Positives (FPR)
The V15.4 Baseline (Cand A) degrades under noise by misclassifying real videos as fake (FPR spikes to 0.3200 at Moderate, 1.0000 at Strong).
Cand B (Mild) and Cand D (Mixed) are utterly destroyed by False Positives; their FPR hits 1.0000 even at Moderate noise. They simply predict everything is fake.
Cand C (Moderate) exhibits a totally different failure mode: its FPR remains pristine under noise (0.02 -> 0.00 -> 0.06), but it fails by missing fakes (FNR spikes to 0.9000). While failing safe is better than flagging reals, the FNR is too high.

## Training Comparability
All models were trained under identical configuration except for noise augmentation, using the verified batch_size=1 setting to avoid OOM.

## Locked Test Gate Assessment
**Does any candidate clearly deserve locked-test evaluation?**

**NO.**

None of the candidates demonstrate a meaningful and credible improvement over the V15.4 baseline. 
- Candidates B and D suffer catastrophic False Positive Rates on real videos under noise (FPR = 1.0). 
- Candidate C preserves FPR nicely under noise but its True Positive Rate completely collapses (FNR = 0.90). 
- Candidate A (the existing V15.4 baseline) remains the most balanced model.

**Recommendation:** Do NOT run the locked test. V15.4 remains the production baseline.
"""
    
    with open("v16_experiments/V16_1_SELECTION_ANALYSIS.md", "w") as f:
        f.write(md)
        
    print("Report written to v16_experiments/V16_1_SELECTION_ANALYSIS.md")

if __name__ == "__main__":
    generate_analysis()
