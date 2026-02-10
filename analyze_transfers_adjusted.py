import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import os

# Re-use the generator logic but modify it to load the adjusted data
def get_adjusted_data():
    """
    Loads 'Pl_foreign_transfers_adjusted.csv' and prepares it for training.
    Logic: 
    - Loads real history
    - Maps 'Target_Relative_GA' to Success Classes (Value Added)
    - Synthesizes comparable synthetic data for feeder leagues
    """
    
    real_players = []
    
    if os.path.exists("Pl_foreign_transfers_adjusted.csv"):
        print("Loading Team-Adjusted Transfer History...")
        history_df = pd.read_csv("Pl_foreign_transfers_adjusted.csv")
        
        for _, row in history_df.iterrows():
            # Synthesize missing advanced metrics
            prgC = row['Pre_PL_xA'] * 8.0 + np.random.normal(1, 0.5) 
            prgP = row['Pre_PL_npxG'] * 5.0 + np.random.normal(1, 0.5)
            
            # --- NEW LABELING LOGIC based on RELATIVE PERFORMANCE ---
            # If a player is > +0.15 above team avg -> Star (Elevates team)
            # If a player is between -0.05 and +0.15 -> Effective (Solid contributor)
            # If a player is < -0.05 below team avg -> Flop (Drags team down/Bench)
            
            val = row['Target_Relative_GA']
            if val >= 0.15: verdict = "Star"
            elif val >= -0.05: verdict = "Effective"
            else: verdict = "Flop"
            
            # Buying Club Tier Logic (approximation from PL_Team name if needed, or stick to generic)
            # We'll stick to a generic placeholder or infer from Team Avg GA?
            # Higher Team Avg GA = Better Team
            if row['PL_Team_Avg_GA'] > 0.45: tier = "Elite"
            elif row['PL_Team_Avg_GA'] > 0.35: tier = "Upper-Mid"
            elif row['PL_Team_Avg_GA'] > 0.25: tier = "Mid-Table"
            else: tier = "Relegation"
            
            real_players.append({
                "Name": row['Name'],
                "Origin_League": row['Origin_League'].split("-")[-1] if "-" in row['Origin_League'] else row['Origin_League'],
                "Age": np.random.randint(20, 29), # Missing in source
                "Transfer_Fee_M": round(row['Pre_PL_npxG'] * 40 + np.random.normal(10, 5), 1),
                "G_A_per_90": round(row['Pre_PL_G_A'], 2),
                "npxG_per_90": round(row['Pre_PL_npxG'], 2),
                "xAG_per_90": round(row['Pre_PL_xA'], 2),
                "PrgC_per_90": round(max(0, prgC), 2),
                "PrgP_per_90": round(max(0, prgP), 2),
                "Buying_Club_Tier": tier, # Now data-driven!
                "PL_Success": verdict,
                "Relative_GA": val
            })
            
    # Synthetic Data (must match new schema)
    # We generate synthetic players for missing leagues, but apply the same label logic
    np.random.seed(42)
    n_samples = 1500
    leagues = {"Liga NOS": 0.65, "Eredivisie": 0.60, "Brasileirao": 0.55}
    
    synthetic_data = []
    for _ in range(n_samples):
        league_name = np.random.choice(list(leagues.keys()))
        league_strength = leagues[league_name]
        talent = np.random.normal(0.5, 0.15)
        if talent < 0.4: continue
        
        raw_prod = talent / league_strength
        g_a = raw_prod + np.random.normal(0, 0.1)
        
        # ... (Same metric generation as before) ...
        npxG = g_a * 0.6; xAG = g_a * 0.2; prgC = np.random.normal(2, 1); prgP = np.random.normal(2, 1)
        
        # Label Logic: Simulate PL transition
        adaptability = np.random.normal(0, 0.1)
        buying_tier = np.random.choice(["Elite", "Upper-Mid", "Mid-Table", "Relegation"])
        
        # In this model, success is RELATIVE.
        # A player with 0.6 talent going to an Elite team (0.8 baseline) might have -0.2 relative performance (Flop).
        # A player with 0.6 talent going to Relegation team (0.3 baseline) might have +0.3 relative performance (Star).
        
        team_baselines = {"Elite": 0.50, "Upper-Mid": 0.40, "Mid-Table": 0.30, "Relegation": 0.20}
        pl_raw_output = talent + adaptability
        relative_perf = pl_raw_output - team_baselines[buying_tier]
        
        if relative_perf >= 0.15: verdict = "Star"
        elif relative_perf >= -0.05: verdict = "Effective"
        else: verdict = "Flop"
        
        synthetic_data.append({
            "Name": f"Sim_{np.random.randint(1000)}",
            "Origin_League": league_name,
            "Age": np.random.randint(18, 28),
            "Transfer_Fee_M": 20.0,
            "G_A_per_90": round(g_a, 2),
            "npxG_per_90": round(npxG, 2),
            "xAG_per_90": round(xAG, 2),
            "PrgC_per_90": round(prgC, 2),
            "PrgP_per_90": round(prgP, 2),
            "Buying_Club_Tier": buying_tier,
            "PL_Success": verdict,
            "Relative_GA": relative_perf
        })

    return pd.DataFrame(real_players + synthetic_data)

def main():
    print("Running Team-Adjusted Analysis (XGBoost)...")
    df = get_adjusted_data()
    
    # Encoder
    le = LabelEncoder()
    df["Verdict_Encoded"] = le.fit_transform(df["PL_Success"])
    class_names = list(le.classes_)
    
    print("\n--- Adjusted Class Distribution ---")
    print(df["PL_Success"].value_counts(normalize=True))
    
    # Feature Engineering
    X = df[["Origin_League", "Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90", "Buying_Club_Tier"]]
    y = df["Verdict_Encoded"]
    
    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League", "Buying_Club_Tier"]),
            ('num', StandardScaler(), ["Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90"])
        ]
    )
    
    xgb = XGBClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='multi:softprob',
        num_class=len(class_names),
        eval_metric='mlogloss'
    )
    
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb)])
    
    # Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n--- Evaluation on Adjusted Targets ---")
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
    
    # Feature Importance
    print("\n--- Key Predictors of Value Added ---")
    feature_names = (clf.named_steps['preprocessor']
                     .transformers_[0][1]
                     .get_feature_names_out(["Origin_League", "Buying_Club_Tier"]).tolist() + 
                     ["Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90"])
    
    importances = clf.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    print(feat_imp.sort_values("Importance", ascending=False).head(10))
    
    # Real World Predictions on Current Targets (Simulated with new logic)
    # We will grab the 2023-24 players from the previous scraping (Understat 2023)
    # and predict their *potential* relative success
    
    # Load 2023 source data directly for prediction
    if os.path.exists("real_understat_data.csv"):
        print("\n--- Predicting Value Added for Current Targets ---")
        current_df = pd.read_csv("real_understat_data.csv")
        # Filter for non-PL
        current_df = current_df[current_df['league'] != 'ENG-Premier League']
        current_df = current_df[current_df['minutes'] > 1000].copy()
        
        # Prepare for model
        # We need to simulate them going to an "Average" PL team (Mid-Table) to judge raw quality
        # Or predict for an "Elite" move
        
        target_rows = []
        for _, row in current_df.iterrows():
            target_rows.append({
                "Name": row['player'],
                "Origin_League": row['league'].split("-")[-1] if "-" in row['league'] else row['league'],
                "Age": 24, # Dummy
                "Transfer_Fee_M": 30.0, # Dummy
                "G_A_per_90": (row['goals'] + row['assists']) / (row['minutes']/90),
                "npxG_per_90": row['np_xg'] / (row['minutes']/90),
                "Buying_Club_Tier": "Elite" # Let's see who is good enough for an ELITE team
            })
            
        target_df = pd.DataFrame(target_rows)
        target_X = target_df[X.columns]
        
        probs = clf.predict_proba(target_X)
        star_idx = list(le.classes_).index("Star")
        target_df["Star_Prob"] = probs[:, star_idx]
        target_df["Predicted_Verdict"] = le.inverse_transform(clf.predict(target_X))
        
        print(target_df.sort_values("Star_Prob", ascending=False).head(15).to_string(index=False))
        target_df.to_csv("final_predictions_adjusted.csv", index=False)

if __name__ == "__main__":
    main()
