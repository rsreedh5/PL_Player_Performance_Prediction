import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from data_generator import get_data

# Set visual style
sns.set_theme(style="whitegrid")

def main():
    print("Loading Data...")
    df = get_data()
    
    # 1. Exploratory Analysis
    print("\n--- Data Overview ---")
    print(df.groupby("Origin_League")[["Pre_PL_G_A_per_90", "PL_G_A_per_90"]].mean().sort_values("PL_G_A_per_90", ascending=False))

    # Calculate "Transfer Tax" (Percentage of output retained)
    df["Retention_Rate"] = df["PL_G_A_per_90"] / df["Pre_PL_G_A_per_90"]
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Origin_League", y="Retention_Rate", data=df)
    plt.title("Performance Retention by Origin League (1.0 = Same Output)")
    plt.axhline(1.0, color='r', linestyle='--')
    plt.savefig("retention_by_league.png")
    print("\nPlot saved: retention_by_league.png")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Transfer_Fee_M", y="PL_G_A_per_90", hue="Origin_League", data=df, alpha=0.7)
    plt.title("Transfer Fee vs. PL Success")
    plt.savefig("fee_vs_success.png")
    print("Plot saved: fee_vs_success.png")

    # 2. Predictive Modeling
    print("\n--- Training Prediction Model ---")
    
    X = df[["Origin_League", "Age", "Transfer_Fee_M", "Pre_PL_G_A_per_90"]]
    y = df["PL_G_A_per_90"]

    # Preprocessing: OneHotEncode League
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Origin_League'])
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance -> RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # 3. Specific Player Analysis
    print("\n--- Specific Player Analysis (Actual vs Predicted) ---")
    
    # Filter for our named real players (excluding synthetic 'Player_XXXX')
    real_players_df = df[~df["Name"].str.startswith("Player_")]
    
    # Predict for them
    real_X = real_players_df[["Origin_League", "Age", "Transfer_Fee_M", "Pre_PL_G_A_per_90"]]
    real_players_df = real_players_df.copy() # Avoid SettingWithCopyWarning
    real_players_df["Predicted_PL_G_A"] = model.predict(real_X)
    real_players_df["Difference"] = real_players_df["PL_G_A_per_90"] - real_players_df["Predicted_PL_G_A"]
    
    # Define "Effectiveness" categories
    def categorize(row):
        if row["PL_G_A_per_90"] > 0.6: return "Star"
        if row["PL_G_A_per_90"] > 0.4: return "Effective"
        return "Flop/Struggle"

    real_players_df["Verdict"] = real_players_df.apply(categorize, axis=1)

    # Display Table
    cols_to_show = ["Name", "Origin_League", "Pre_PL_G_A_per_90", "PL_G_A_per_90", "Predicted_PL_G_A", "Verdict"]
    print(real_players_df[cols_to_show].sort_values("PL_G_A_per_90", ascending=False).to_string(index=False))

    # Plot Actual vs Predicted for known players
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="Predicted_PL_G_A", y="PL_G_A_per_90", data=real_players_df, s=100, color='blue')
    
    # Add labels
    for i, row in real_players_df.iterrows():
        plt.text(row["Predicted_PL_G_A"]+0.01, row["PL_G_A_per_90"]+0.01, row["Name"], fontsize=9)
        
    # Perfect prediction line
    plt.plot([0, 1.5], [0, 1.5], 'r--', label="Perfect Prediction")
    plt.xlabel("Predicted G+A/90")
    plt.ylabel("Actual G+A/90")
    plt.title("Model Prediction vs Reality for Key Transfers")
    plt.legend()
    plt.savefig("prediction_vs_reality.png")
    print("Plot saved: prediction_vs_reality.png")

if __name__ == "__main__":
    main()