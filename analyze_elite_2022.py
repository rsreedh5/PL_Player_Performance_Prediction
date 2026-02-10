import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

sns.set_theme(style="whitegrid")

def run_cohort_analysis(df, min_ga, cohort_name):
    print(f"\n==================================================")
    print(f"   ANALYSIS: {cohort_name} (G+A > {min_ga})")
    print(f"==================================================")
    
    # Filter
    df_cohort = df[df['Pre_PL_G_A'] > min_ga].copy()
    
    # Split (2022 Cutoff)
    split_year = 2022
    train = df_cohort[df_cohort['Season_Control'] < split_year]
    test = df_cohort[df_cohort['Season_Control'] >= split_year]
    
    print(f"Training: {len(train)} | Testing: {len(test)}")
    
    if len(train) < 10 or len(test) < 5:
        print("Not enough data.")
        return

    features = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score",
        "Season_Control"
    ]
    target = "PL_Raw_GA"
    
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"]),
            ('num', StandardScaler(), ["Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score", "Season_Control"])
        ]
    )
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
    
    pipeline.fit(train[features], train[target])
    y_pred = pipeline.predict(test[features])
    
    # Metrics
    r2 = r2_score(test[target], y_pred)
    rmse = np.sqrt(mean_squared_error(test[target], y_pred))
    
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE:     {rmse:.3f}")
    
    # Show top errors
    results = test.copy()
    results['Predicted'] = y_pred
    results['Error'] = results['Predicted'] - results['PL_Raw_GA']
    
    print("\n--- Prediction Samples ---")
    print(results[['Name', 'Origin_League', 'Predicted', 'PL_Raw_GA', 'Error']].head(5))

def main():
    if not os.path.exists("Pl_foreign_transfers_enhanced.csv"):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv("Pl_foreign_transfers_enhanced.csv")
    
    # Run 1: Proven (> 0.20)
    run_cohort_analysis(df, 0.20, "PROVEN TALENT (>0.20)")
    
    # Run 2: Elite (> 0.55)
    run_cohort_analysis(df, 0.55, "ELITE TALENT (>0.55)")

if __name__ == "__main__":
    main()
