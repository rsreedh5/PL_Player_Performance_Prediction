import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import os

# Set visual style
sns.set_theme(style="whitegrid")

def train_and_evaluate(df, filter_name):
    print(f"\n==================================================")
    print(f"   ANALYSIS: {filter_name}")
    print(f"   Training Data Size: {len(df)} players")
    print(f"==================================================")
    
    if len(df) < 50:
        print("Not enough data to train a reliable model.")
        return

    # Encoder
    le = LabelEncoder()
    df["Verdict_Encoded"] = le.fit_transform(df["PL_Success"])
    class_names = list(le.classes_)
    
    print("\n--- PL Success Distribution (for this group) ---")
    print(df["PL_Success"].value_counts(normalize=True))
    
    # Features
    X = df[["Origin_League", "Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90", "Buying_Club_Tier"]]
    y = df["Verdict_Encoded"]
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League", "Buying_Club_Tier"]),
            ('num', StandardScaler(), ["Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90"])
        ]
    )
    
    # XGBoost
    xgb = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        objective='multi:softprob',
        num_class=len(class_names),
        eval_metric='mlogloss',
        random_state=42
    )
    
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb)])
    
    # Train/Test Split
    # Stratify might fail if classes are too small, so try/except
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n--- Model Performance ---")
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
    
    # Feature Importance
    print("\n--- What matters for THIS group? ---")
    feature_names = (clf.named_steps['preprocessor']
                     .transformers_[0][1]
                     .get_feature_names_out(["Origin_League", "Buying_Club_Tier"]).tolist() + 
                     ["Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90"])
    
    importances = clf.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    print(feat_imp.sort_values("Importance", ascending=False).head(5))
    
    return clf, le

def main():
    # 1. Load the Adjusted 10-Year History (so we have the 'Relative' labels and metrics)
    # We will reconstruct the dataframe using the logic from 'data_generator' but only keeping Real history
    if not os.path.exists("Pl_foreign_transfers_adjusted.csv"):
        print("Error: Historical file not found.")
        return

    raw_df = pd.read_csv("Pl_foreign_transfers_adjusted.csv")
    
    # Construct working DF with our standard labels
    # We need to recreate the "PL_Success" label column first
    def get_label(val):
        if val >= 0.15: return "Star"
        elif val >= -0.05: return "Effective"
        return "Flop"
        
    raw_df['PL_Success'] = raw_df['Target_Relative_GA'].apply(get_label)
    
    # Also infer Buying Club Tier roughly for context
    def get_tier(val):
        if val > 0.45: return "Elite"
        elif val > 0.35: return "Upper-Mid"
        elif val > 0.25: return "Mid-Table"
        return "Relegation"
    
    raw_df['Buying_Club_Tier'] = raw_df['PL_Team_Avg_GA'].apply(get_tier)
    
    # Rename cols to match training schema
    df = raw_df.rename(columns={
        'Pre_PL_G_A': 'G_A_per_90',
        'Pre_PL_npxG': 'npxG_per_90',
        # Transfer fee wasn't in the raw csv, we need to proxy it or drop it.
        # Let's proxy it again for consistency with previous models
    })
    
    # Proxy Fee
    df['Transfer_Fee_M'] = df['npxG_per_90'] * 40 + np.random.normal(10, 5, len(df))
    df['Origin_League'] = df['Origin_League'].apply(lambda x: x.split("-")[-1] if "-" in x else x)
    df['Age'] = np.random.randint(20, 29, len(df)) # Proxy age as before

    # --- FILTER 1: Proven Talent (Effective + Star in Origin) ---
    # Threshold: Pre_PL_G_A >= 0.30
    df_proven = df[df['G_A_per_90'] >= 0.30].copy()
    clf_proven, le_proven = train_and_evaluate(df_proven, "PROVEN TALENT (Origin G+A > 0.30)")

    # --- FILTER 2: Elite Only (Stars in Origin) ---
    # Threshold: Pre_PL_G_A >= 0.55
    df_stars = df[df['G_A_per_90'] >= 0.55].copy()
    clf_stars, le_stars = train_and_evaluate(df_stars, "ELITE STARS ONLY (Origin G+A > 0.55)")
    
    # Visualization: Success Rate Comparison
    plt.figure(figsize=(10, 6))
    
    # Calculate rates
    rates = {
        "All Transfers": df['PL_Success'].value_counts(normalize=True).get('Star', 0),
        "Proven (>0.30)": df_proven['PL_Success'].value_counts(normalize=True).get('Star', 0),
        "Elite (>0.55)": df_stars['PL_Success'].value_counts(normalize=True).get('Star', 0)
    }
    
    # Bar plot
    plt.bar(rates.keys(), rates.values(), color=['gray', 'blue', 'gold'])
    plt.title("Does Buying Better Players Work?\nProbability of becoming a PL 'Star'")
    plt.ylabel("Success Rate (Star)")
    plt.ylim(0, 1.0)
    
    # Add text
    for i, v in enumerate(rates.values()):
        plt.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
        
    plt.savefig("elite_filter_analysis.png")
    print("\nSaved chart to 'elite_filter_analysis.png'")

if __name__ == "__main__":
    main()
