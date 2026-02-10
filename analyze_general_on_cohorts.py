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

def main():
    print("Training GENERAL Model (All Data < 2022)...")
    
    if not os.path.exists("Pl_foreign_transfers_enhanced.csv"):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv("Pl_foreign_transfers_enhanced.csv")
    
    # 1. Split Data
    split_year = 2022
    train_df = df[df['Season_Control'] < split_year].copy()
    test_df_full = df[df['Season_Control'] >= split_year].copy()
    
    print(f"Training Set: {len(train_df)} players")
    print(f"Full Test Set: {len(test_df_full)} players")

    # 2. Train General Model
    features = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"
    ]
    target = "PL_Raw_GA"
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"]),
            ('num', StandardScaler(), ["Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"])
        ]
    )
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
    
    pipeline.fit(train_df[features], train_df[target])
    
    # 3. Evaluate on COHORTS
    def evaluate_cohort(name, min_ga):
        cohort = test_df_full[test_df_full['Pre_PL_G_A'] > min_ga].copy()
        
        if len(cohort) == 0:
            print(f"\n--- Cohort: {name} ---")
            print("No players in this cohort.")
            return

        y_true = cohort[target]
        y_pred = pipeline.predict(cohort[features])
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"\n--- Cohort: {name} (n={len(cohort)}) ---")
        print(f"R²:   {r2:.3f}")
        print(f"RMSE: {rmse:.3f}")
        
        # Show specific predictions
        cohort['Predicted'] = y_pred
        cohort['Error'] = y_pred - y_true
        print(cohort[['Name', 'Pre_PL_G_A', 'Predicted', 'PL_Raw_GA']].head(5))

    # Evaluate
    evaluate_cohort("Proven Talent (> 0.20)", 0.20)
    evaluate_cohort("Elite Stars (> 0.55)", 0.55)

if __name__ == "__main__":
    main()
