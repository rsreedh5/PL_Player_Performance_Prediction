import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def main():
    print("Running XGBoost Feature Ablation Study...")
    
    input_file = "Pl_transfers_20apps_enriched.csv"
    if not os.path.exists(input_file):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['Pre_PL_Shots_p90'])
    
    # Split
    split_year = 2022
    train_df = df[df['Season_Control'] < split_year].copy()
    test_df_full = df[df['Season_Control'] >= split_year].copy()
    
    # Test Cohort: Proven Talent (> 0.20)
    cohort = test_df_full[test_df_full['Pre_PL_G_A'] > 0.20].copy()
    y_true = cohort["PL_Raw_GA"]
    
    if len(cohort) == 0:
        print("No players in cohort.")
        return

    # Full Feature List
    all_features = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score",
        "Pre_PL_Shots_p90", "Pre_PL_KeyPasses_p90",
        "Origin_League_Avg_xG_p90", "Origin_League_Avg_xA_p90",
        "Origin_League_Avg_Shots_p90", "Origin_League_Avg_KP_p90"
    ]
    
    target = "PL_Raw_GA"
    
    results = []
    
    # 1. Baseline (All Features)
    def train_eval(feats):
        # Identify numericals dynamically
        nums = [f for f in feats if f != "Origin_League"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"] if "Origin_League" in feats else []),
                ('num', StandardScaler(), nums)
            ]
        )
        
        model = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))])
        model.fit(train_df[feats], train_df[target])
        y_pred = model.predict(cohort[feats])
        
        return r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

    base_r2, base_rmse = train_eval(all_features)
    results.append({"Removed Feature": "(None - Baseline)", "R²": base_r2, "RMSE": base_rmse})
    
    # 2. Ablation Loop
    for feature in all_features:
        # Create subset list
        subset = [f for f in all_features if f != feature]
        
        # Train/Eval
        r2, rmse = train_eval(subset)
        results.append({"Removed Feature": feature, "R²": r2, "RMSE": rmse})
        
    # 3. Output Table
    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("R²", ascending=False)
    
    print("\n--- Feature Ablation Results (Proven Cohort) ---")
    print(res_df.to_string(index=False))
    
    # Interpret
    best_config = res_df.iloc[0]
    print(f"\nBest Configuration: Removing '{best_config['Removed Feature']}' achieved R² {best_config['R²']:.3f}")

if __name__ == "__main__":
    main()
