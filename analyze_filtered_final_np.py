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
    print("Training FINAL OPTIMIZED Model (Target: Non-Penalty G+A)...")
    
    input_file = "Pl_transfers_20apps_enriched.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['Pre_PL_Shots_p90'])
    
    # Split
    split_year = 2022
    train_df = df[df['Season_Control'] < split_year].copy()
    test_df_full = df[df['Season_Control'] >= split_year].copy()
    
    # Cohort for Evaluation
    cohort = test_df_full[test_df_full['Pre_PL_G_A'] > 0.20].copy()
    
    # --- CHANGED TARGET TO NP GA ---
    target = "PL_NP_GA" 
    y_true = cohort[target]

    # Full Feature Set to Test
    base_features = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "League_Quality_Score",
        "Pre_PL_KeyPasses_p90", "Pre_PL_Shots_p90",
        "Finishing_Overperf",
        "Origin_League_Avg_xG_p90", "Origin_League_Avg_xA_p90",
        "Origin_League_Avg_Shots_p90", "Origin_League_Avg_KP_p90"
    ]

    # Helper to train
    def train_eval(feats):
        nums = [f for f in feats if f != "Origin_League"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"] if "Origin_League" in feats else []),
                ('num', StandardScaler(), nums)
            ]
        )
        
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        model = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
        
        model.fit(train_df[feats], train_df[target])
        y_pred = model.predict(cohort[feats])
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Get importance if full set
        imp_dict = {}
        if len(feats) == len(base_features):
            feature_names = (model.named_steps['preprocessor']
                             .transformers_[0][1]
                             .get_feature_names_out(list(set(feats) & {"Origin_League"})).tolist() + nums)
            importances = model.named_steps['model'].feature_importances_
            
            # Map back to original feature names (aggregating OneHot for League)
            for i, name in enumerate(feature_names):
                base_name = name
                if "Origin_League" in name:
                    base_name = "Origin_League"
                
                if base_name in imp_dict:
                    imp_dict[base_name] += importances[i]
                else:
                    imp_dict[base_name] = importances[i]
                    
        return r2, rmse, imp_dict

    # 1. Baseline Run
    base_r2, base_rmse, base_imps = train_eval(base_features)
    
    print(f"\n--- Baseline NP Performance (Proven Cohort) ---")
    print(f"R²: {base_r2:.3f}")
    print(f"RMSE: {base_rmse:.3f}")

    # 2. Ablation Loop
    results = []
    
    # Add Baseline Row
    results.append({
        "Feature": "(None - Baseline)",
        "R² (if removed)": base_r2,
        "RMSE (if removed)": base_rmse,
        "Tree Importance": "N/A"
    })

    for feature in base_features:
        subset = [f for f in base_features if f != feature]
        r2, rmse, _ = train_eval(subset)
        
        # Get importance from baseline dict
        imp = base_imps.get(feature, 0.0)
        
        results.append({
            "Feature": feature,
            "R² (if removed)": r2,
            "RMSE (if removed)": rmse,
            "Tree Importance": round(imp, 4)
        })

    # 3. Print Table
    res_df = pd.DataFrame(results)
    # Sort by R² desc (Higher R² when removed = Feature was Bad. Lower R² = Feature was Good)
    res_df = res_df.sort_values("R² (if removed)", ascending=True)
    
    print("\n--- Feature Impact Report (Target: PL Non-Penalty G+A) ---")
    print("Interpretation: Low 'R² (if removed)' means the feature was CRITICAL.")
    print("High 'Tree Importance' means the model used it often for splits.\n")
    print(res_df.to_string(index=False))
    
    # Save table
    res_df.to_csv("Variable_decision_analysis_NP.csv", index=False)
    print("\nSaved NP variable analysis to 'Variable_decision_analysis_NP.csv'")

if __name__ == "__main__":
    main()
