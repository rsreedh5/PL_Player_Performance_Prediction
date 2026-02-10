import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

sns.set_theme(style="whitegrid")

def main():
    print("Initializing Rolling Window Analysis (Train: 4 Years, Test: Final Year)...")
    
    if not os.path.exists("Pl_foreign_transfers_enhanced.csv"):
        print("Error: Dataset not found.")
        return
        
    df = pd.read_csv("Pl_foreign_transfers_enhanced.csv")
    
    # 1. Rolling Window Setup
    # Dataset seasons: 2014-2023
    # Target Test Season: 2023 (Most recent)
    # Training Seasons: 2019, 2020, 2021, 2022
    
    # Logic Correction:
    # 'Season_Control' is the ORIGIN season.
    # To predict transfers happening in Summer 2023 (for 23/24 PL season), 
    # we look at players whose last origin season was 2022.
    
    test_origin_season = 2022
    start_train_season = test_origin_season - 4
    
    # Filter dataset to just this window
    window_df = df[
        (df['Season_Control'] >= start_train_season) & 
        (df['Season_Control'] <= test_origin_season)
    ].copy()
    
    print(f"Window (Origin Seasons): {start_train_season} -> {test_origin_season}")
    print(f"Total Transfers in Window: {len(window_df)}")
    
    # Split
    train_df = window_df[window_df['Season_Control'] < test_origin_season]
    test_df = window_df[window_df['Season_Control'] == test_origin_season]
    
    print(f"Training Set (Origin {start_train_season}-{test_origin_season-1}): {len(train_df)} transfers")
    print(f"Testing Set (Origin {test_origin_season} -> PL 23/24): {len(test_df)} transfers")
    
    if len(train_df) < 10 or len(test_df) < 5:
        print("Error: Not enough data in this specific window.")
        return

    # Features (Same as successful regression model)
    feature_cols = [
        "Origin_League", 
        "Pre_PL_G_A", 
        "Pre_PL_npxG", 
        "Pre_PL_xA", 
        "Finishing_Overperf",
        "League_Quality_Score"
    ]
    target_col = "PL_Raw_GA"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League"]),
            ('num', StandardScaler(), ["Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"])
        ]
    )
    
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3, # Lower depth to prevent overfitting on smaller training set
        objective='reg:squarederror',
        random_state=42
    )
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Results for Season {test_origin_season} (Transfers Summer {test_origin_season+1}) ---")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")
    
    # Predictions
    results = test_df.copy()
    results['Predicted_PL_GA'] = y_pred
    results['Error'] = results['Predicted_PL_GA'] - results['PL_Raw_GA']
    
    print("\n--- 2023 Season Predictions vs Reality ---")
    print(results[['Name', 'Origin_League', 'Predicted_PL_GA', 'PL_Raw_GA', 'Error']].sort_values('Error', ascending=False).head(10))
    
    # Visualization
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([0, 1.0], [0, 1.0], 'r--')
    plt.title(f"Rolling Window Test: {test_origin_season} Transfers")
    plt.xlabel("Actual G+A/90")
    plt.ylabel("Predicted G+A/90")
    plt.savefig("rolling_window_test.png")
    print("Saved 'rolling_window_test.png'")

if __name__ == "__main__":
    main()
