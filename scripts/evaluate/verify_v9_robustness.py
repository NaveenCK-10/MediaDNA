import pandas as pd
import os

def verify_v9():
    csv_path = r"C:\Users\navee\Desktop\Projects\MediaDna\OpenAVFF\reports\v9_evaluation_results.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions from V9 Robustness suite.")
    
    # Use the pre-computed success column from the V9 runner
    df['correct'] = df['success']
    
    print("\n--- ROBUSTNESS ACCURACY RECALCULATION ---")
    grouped = df.groupby('corruption')['correct'].agg(['mean', 'count'])
    
    for idx, row in grouped.iterrows():
        acc = row['mean'] * 100
        count = row['count']
        print(f"{idx} (N={count}): {acc:.2f}% Accuracy")

    print("\n--- CATEGORY BREAKDOWN ---")
    grouped_cat = df.groupby(['corruption', 'category'])['correct'].agg(['mean', 'count'])
    print(grouped_cat)
    
if __name__ == "__main__":
    verify_v9()
