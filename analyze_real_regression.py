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

# Set style
sns.set_theme(style="whitegrid")

def main():
    print("Initializing STRICT Real-Data Regression Analysis...")
    
    # 1. Load Data
    if not os.path.exists("Pl_foreign_transfers_enhanced.csv"):
        print("Error: Dataset 'Pl_foreign_transfers_enhanced.csv' not found.")
        return
        
    df = pd.read_csv("Pl_foreign_transfers_enhanced.csv")
    print(f"Loaded {len(df)} transfers.")

    # 2. Strict Filtering (No Synthetic Features)
    # We drop 'Age' and 'Transfer_Fee_M' because they were synthetic/missing.
    # We drop 'PrgC' and 'PrgP' if they were synthetic in previous steps (check logic).
    # The 'enhanced' csv had synthetic age/fee added in previous steps? 
    # Let's check the columns in the CSV.
    # The CSV columns are: Name, Origin_Season, Origin_League, team, Pre_PL_G_A, Pre_PL_npxG, Pre_PL_xA, ...
    # 'Season_Control' is real (derived from season year).
    
    # We will use ONLY the columns we extracted from Understat + Season info.
    
    feature_cols = [
        "Origin_League", 
        "Pre_PL_G_A", 
        "Pre_PL_npxG", 
        "Pre_PL_xA", 
        "Finishing_Overperf",
        "League_Quality_Score"
    ]
    
    target_col = "PL_Raw_GA" # Predicting raw G+A/90 in PL
    
    # Drop rows with NaNs in these specific columns
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    
    # 3. Time-Series Split
    # Train: 2013/14 -> 2021/22 (Seasons < 2022)
    # Test:  2022/23+ (Seasons >= 2022)
    # Note: Season 2022 means they played abroad in 22/23 and moved to PL in 23/24
    
    split_year = 2022
    train_df = df_clean[df_clean['Season_Control'] < split_year]
    test_df = df_clean[df_clean['Season_Control'] >= split_year]
    
    print(f"\n--- Temporal Split (Cutoff: {split_year}) ---")
    print(f"Training Set (Historic): {len(train_df)} transfers")
    print(f"Testing Set (Modern):    {len(test_df)} transfers")
    
    if len(train_df) < 10 or len(test_df) < 10:
        print("Error: Not enough data in split.")
        return

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 4. Pipeline
    categorical_features = ["Origin_League"]
    numerical_features = ["Pre_PL_G_A", "Pre_PL_npxG", "Pre_PL_xA", "Finishing_Overperf", "League_Quality_Score"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )
    
    # XGBoost Regressor
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective='reg:squarederror',
        random_state=42
    )
    
    model = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb)])
    
    # 5. Train
    model.fit(X_train, y_train)
    
    # 6. Predict & Evaluate
    y_pred = model.predict(X_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Regression Results (Predicting PL G+A/90) ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")
    print(f"MAE  (Mean Absolute Error):     {mae:.3f}")
    print(f"R²   (Variance Explained):      {r2:.3f}")
    
    # 7. Analysis of Errors
    results = test_df.copy()
    results['Predicted_PL_GA'] = y_pred
    results['Error'] = results['Predicted_PL_GA'] - results['PL_Raw_GA']
    
    print("\n--- Biggest Over-Predictions (Flops?) ---")
    # Model predicted high, actual was low -> Positive Error
    print(results.sort_values("Error", ascending=False).head(5)[['Name', 'Origin_League', 'Predicted_PL_GA', 'PL_Raw_GA', 'Error']])
    
    print("\n--- Biggest Under-Predictions (Breakout Stars?) ---")
    # Model predicted low, actual was high -> Negative Error
    print(results.sort_values("Error", ascending=True).head(5)[['Name', 'Origin_League', 'Predicted_PL_GA', 'PL_Raw_GA', 'Error']])
    
    # Feature Importance
    feature_names = (model.named_steps['preprocessor']
                     .transformers_[0][1]
                     .get_feature_names_out(categorical_features).tolist() + numerical_features)
    
    importances = model.named_steps['model'].feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp.sort_values("Importance", ascending=False), palette="viridis")
    plt.title("What actually predicts Premier League Output? (Regression)")
    plt.tight_layout()
    plt.savefig("regression_importance.png")
    print("\nSaved importance plot to 'regression_importance.png'")
    
    # Scatter Plot: Actual vs Predicted
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([0, 1.5], [0, 1.5], 'r--') # Perfect prediction line
    plt.xlabel("Actual PL G+A/90")
    plt.ylabel("Predicted PL G+A/90")
    plt.title("Prediction vs Reality: 2020-2024 Transfers")
    plt.savefig("regression_scatter.png")
    print("Saved scatter plot to 'regression_scatter.png'")

if __name__ == "__main__":
    main()
