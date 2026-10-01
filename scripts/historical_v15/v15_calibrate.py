import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from src.dataloader import VideoAudioEvalDataset
from backend.inference import OpenAVFFService
import json
from tqdm import tqdm

def evaluate_thresholds():
    service = OpenAVFFService()
    
    val_df = pd.read_csv("data/val_v14.csv")
    audio_conf = {
        'num_mel_bins': 128, 
        'target_length': 1024, 
        'freqm': 0, 
        'timem': 0, 
        'mixup': 0, 
        'mean': -5.081, 
        'std': 4.4849, 
        'noise': False,
        'mode': 'eval',
        'im_res': 224,
        'skip_norm': False
    }
    
    dataset = VideoAudioEvalDataset(
        csv_file="data/val_v14.csv",
        audio_conf=audio_conf,
        num_frames=16
    )
    
    y_true = []
    y_prob = []
    
    print("Evaluating Validation Set...")
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            fbank, vframes, label, _ = dataset[i]
            # label is a tensor [real_prob, fake_prob]. The actual integer label is index of max, or just use val_df
            real_label = val_df.iloc[i]['label']
            
            fbank_input = fbank.unsqueeze(0).to(service.device)
            out = service.model(fbank_input, vframes.unsqueeze(0).to(service.device))
            probs = torch.sigmoid(out)
            fake_prob = probs[0][0].item()
            
            y_true.append(real_label)
            y_prob.append(fake_prob)

    results = []
    thresholds = np.arange(0.10, 0.95, 0.05)
    for t in thresholds:
        y_pred = [1 if p >= t else 0 for p in y_prob]
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bacc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        else:
            fpr, fnr = 0, 0
            
        results.append({
            'threshold': round(t, 2),
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'balanced_accuracy': round(bacc, 4),
            'fpr': round(fpr, 4),
            'fnr': round(fnr, 4)
        })

    with open('v15_thresholds.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done. Saved to v15_thresholds.json")

if __name__ == "__main__":
    evaluate_thresholds()
