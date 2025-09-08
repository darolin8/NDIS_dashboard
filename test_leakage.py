import pandas as pd
import numpy as np

def simple_leakage_check(df, target="reportable_bin"):
    print("=== SIMPLE LEAKAGE CHECK ===")
    
    if target not in df.columns:
        print(f"Target '{target}' not found!")
        print("Available columns:", list(df.columns))
        return
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df[target].value_counts().to_dict()}")
    
    # Check correlations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    high_corr = []
    
    for col in numeric_cols:
        if col != target:
            try:
                corr = abs(df[col].corr(df[target]))
                if corr > 0.8:
                    high_corr.append((col, corr))
            except:
                pass
    
    if high_corr:
        print("\nHIGH CORRELATIONS (potential leakage):")
        for col, corr in sorted(high_corr, key=lambda x: x[1], reverse=True):
            print(f"  {col}: {corr:.3f}")
    else:
        print("\nNo high correlations found")

if __name__ == "__main__":
    df = pd.read_csv("ndis_incidents_1000.csv")
    simple_leakage_check(df)
