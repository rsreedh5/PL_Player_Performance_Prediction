import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

sns.set_theme(style="whitegrid")

def train_and_evaluate(name, features, num_features, df, train_df, test_df):
    print(f"\n==================================================")
    print(f"   MODEL: {name}")
    print(f"==================================================")
    
    target = "PL_Raw_GA"
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"]),
            ('num', StandardScaler(), num_features)
        ]
    )
    
    # Linear Regression (Ridge for stability)
    model = Pipeline(steps=[('preprocessor', preprocessor), ('model', Ridge(alpha=1.0))])
    
    model.fit(train_df[features], train_df[target])
    
    # Evaluate on Proven Cohort (> 0.20)
    cohort = test_df[test_df['Pre_PL_G_A'] > 0.20].copy()
    
    if len(cohort) == 0:
        print("No players in cohort.")
        return

    y_true = cohort[target]
    y_pred = model.predict(cohort[features])
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"R² (Proven Cohort): {r2:.3f}")
    print(f"RMSE (Proven Cohort): {rmse:.3f}")
    
    # Show coefficients (for interpretability)
    # Get feature names
    try:
        feature_names = (model.named_steps['preprocessor']
                         .transformers_[0][1]
                         .get_feature_names_out(["Origin_League"]).tolist() + num_features)
        
        coeffs = model.named_steps['model'].coef_
        coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coeffs})
        print("\n--- Key Drivers (Coefficients) ---")
        print(coef_df.sort_values("Coefficient", ascending=False).head(5))
        print(coef_df.sort_values("Coefficient", ascending=True).head(5))
    except:
        pass

def main():
    print("Running LINEAR Benchmarks...")
    
    input_file = "Pl_transfers_20apps_enriched.csv"
    if not os.path.exists(input_file):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['Pre_PL_Shots_p90']) # Ensure clean data for enriched
    
    split_year = 2022
    train_df = df[df['Season_Control'] < split_year].copy()
    test_df = df[df['Season_Control'] >= split_year].copy()
    
    # --- MODEL 1: Enriched Linear ---
    feat_enriched = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score",
        "Pre_PL_Shots_p90", "Pre_PL_KeyPasses_p90",
        "Origin_League_Avg_xG_p90", "Origin_League_Avg_xA_p90",
        "Origin_League_Avg_Shots_p90", "Origin_League_Avg_KP_p90"
    ]
    num_enriched = [
        "Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score",
        "Pre_PL_Shots_p90", "Pre_PL_KeyPasses_p90",
        "Origin_League_Avg_xG_p90", "Origin_League_Avg_xA_p90",
        "Origin_League_Avg_Shots_p90", "Origin_League_Avg_KP_p90"
    ]
    
    train_and_evaluate("Enriched Linear (Ridge)", feat_enriched, num_enriched, df, train_df, test_df)
    
    # --- MODEL 2: Original V2 Linear ---
    feat_v2 = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"
    ]
    num_v2 = [
        "Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"
    ]
    
    train_and_evaluate("Original V2 Linear (Ridge)", feat_v2, num_v2, df, train_df, test_df)

if __name__ == "__main__":
    main()
