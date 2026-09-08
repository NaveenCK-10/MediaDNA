import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_plots():
    csv_path = "reports/v9_evaluation_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("CSV not found. Ensure evaluate_robustness.py has finished.")
        return
        
    df = df[df['success'].notnull()]
    
    # 1. Overall Accuracy per Corruption Type
    corruption_acc = df.groupby('corruption')['success'].mean().reset_index()
    corruption_acc['success'] *= 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=corruption_acc, x='corruption', y='success', palette='viridis')
    plt.title('MediaDNA Robustness: Accuracy by Corruption Type')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Corruption')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/accuracy_vs_corruption.png')
    plt.close()
    
    # 2. Accuracy Heatmap (Modality vs Corruption)
    heatmap_data = df.groupby(['category', 'corruption'])['success'].mean().unstack() * 100
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.1f', vmin=0, vmax=100)
    plt.title('MediaDNA Robustness Heatmap (Modality vs Corruption)')
    plt.ylabel('Modality')
    plt.xlabel('Corruption')
    plt.tight_layout()
    plt.savefig('reports/robustness_heatmap.png')
    plt.close()
    
    print("Plots generated successfully in reports/ directory.")

if __name__ == '__main__':
    generate_plots()
