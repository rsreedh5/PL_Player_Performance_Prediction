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

def get_data_split(df):
    # Train: < 2022 (Origin Seasons 2014-2021)
    # Test: >= 2022 (Origin Seasons 2022+)
    split_year = 2022
    train = df[df['Season_Control'] < split_year].copy()
    test = df[df['Season_Control'] >= split_year].copy()
    return train, test

def run_model_a_season_feature(train_df, test_df):
    print("\n--- Model A: Explicit Season Feature ---")
    
    features = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score",
        "Season_Control" # <--- Added Feature
    ]
    target = "PL_Raw_GA"
    
    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"]),
            ('num', StandardScaler(), ["Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score", "Season_Control"])
        ]
    )
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
    
    pipeline.fit(train_df[features], train_df[target])
    y_pred = pipeline.predict(test_df[features])
    
    r2 = r2_score(test_df[target], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df[target], y_pred))
    
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return r2

def run_model_b_time_decay(train_df, test_df):
    print("\n--- Model B: Time Decay Weighting ---")
    
    features = [
        "Origin_League", "Pre_PL_G_A", "Pre_PL_npxG", 
        "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"
        # No Season_Control here, we use it for weights instead
    ]
    target = "PL_Raw_GA"
    
    # Calculate Weights
    # Logic: Decay factor 0.85 per year back from 2022
    # 2021 -> 1.0
    # 2020 -> 0.85
    # 2019 -> 0.72 ...
    
    max_year = 2022
    decay_rate = 0.85
    
    train_df['Time_Weight'] = decay_rate ** (max_year - train_df['Season_Control'])
    
    # Preprocessing (Manual split needed for weights)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"]),
            ('num', StandardScaler(), ["Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"])
        ]
    )
    
    # Transform X manually so we can pass sample_weight to fit
    X_train_processed = preprocessor.fit_transform(train_df[features])
    X_test_processed = preprocessor.transform(test_df[features])
    
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    
    # FIT WITH WEIGHTS
    xgb.fit(X_train_processed, train_df[target], sample_weight=train_df['Time_Weight'])
    
    y_pred = xgb.predict(X_test_processed)
    
    r2 = r2_score(test_df[target], y_pred)
    rmse = np.sqrt(mean_squared_error(test_df[target], y_pred))
    
    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    return r2

def main():
    if not os.path.exists("Pl_foreign_transfers_enhanced.csv"):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv("Pl_foreign_transfers_enhanced.csv")
    train, test = get_data_split(df)
    
    print(f"Training on {len(train)} transfers, Testing on {len(test)} transfers.")
    
    score_a = run_model_a_season_feature(train, test)
    score_b = run_model_b_time_decay(train, test)
    
    # Comparison Chart
    plt.figure(figsize=(8, 6))
    plt.bar(["Feature: Season", "Weight: Time Decay"], [score_a, score_b], color=['skyblue', 'salmon'])
    plt.title("Handling Tactical Evolution:\nWhich Method Predicts 2023 Better?")
    plt.ylabel("R² Score (Higher is Better)")
    plt.ylim(0, 0.5)
    plt.savefig("time_aware_comparison.png")
    print("\nSaved comparison to 'time_aware_comparison.png'")

if __name__ == "__main__":
    main()
