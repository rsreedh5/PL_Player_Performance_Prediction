import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import os

sns.set_theme(style="whitegrid")

def main():
    print("Running Final Analysis with Finishing Overperformance...")
    
    if not os.path.exists("Pl_foreign_transfers_enhanced.csv"):
        print("Error: Enhanced dataset not found.")
        return

    # Load
    raw_df = pd.read_csv("Pl_foreign_transfers_enhanced.csv")
    
    # 1. Label Logic (Relative)
    def get_label(val):
        if val >= 0.15: return "Star"
        elif val >= -0.05: return "Effective"
        return "Flop"
        
    raw_df['PL_Success'] = raw_df['Target_Relative_GA'].apply(get_label)
    
    # 2. Buying Tier Logic
    def get_tier(val):
        if val > 0.45: return "Elite"
        elif val > 0.35: return "Upper-Mid"
        elif val > 0.25: return "Mid-Table"
        return "Relegation"
    
    raw_df['Buying_Club_Tier'] = raw_df['PL_Team_Avg_GA'].apply(get_tier)
    
    # Rename cols for model
    df = raw_df.rename(columns={
        'Pre_PL_G_A': 'G_A_per_90',
        'Pre_PL_npxG': 'npxG_per_90'
    })
    
    # Handle missing Age (Create random distribution since column is missing)
    df['Age'] = np.random.randint(20, 29, len(df))
    
    # Proxy Fee
    df['Transfer_Fee_M'] = df['npxG_per_90'] * 40 + np.random.normal(10, 5, len(df))
    df['Origin_League'] = df['Origin_League'].apply(lambda x: x.split("-")[-1] if "-" in x else x)

    # --- ENCODING ---
    le = LabelEncoder()
    df["Verdict_Encoded"] = le.fit_transform(df["PL_Success"])
    class_names = list(le.classes_)
    
    # --- FEATURES ---
    # Now including Finishing_Overperf
    X = df[[
        "Origin_League", "Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90", 
        "Finishing_Overperf", "Buying_Club_Tier"
    ]]
    y = df["Verdict_Encoded"]
    
    # --- PIPELINE ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ["Origin_League", "Buying_Club_Tier"]),
            ('num', StandardScaler(), ["Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90", "Finishing_Overperf"])
        ]
    )
    
    xgb = XGBClassifier(
        n_estimators=150, 
        learning_rate=0.05, 
        max_depth=5, 
        objective='multi:softprob',
        num_class=len(class_names),
        eval_metric='mlogloss',
        random_state=42
    )
    
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb)])
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n--- Model Performance (Enhanced Features) ---")
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
    
    # Feature Importance
    print("\n--- Does Finishing Overperformance Matter? ---")
    feature_names = (clf.named_steps['preprocessor']
                     .transformers_[0][1]
                     .get_feature_names_out(["Origin_League", "Buying_Club_Tier"]).tolist() + 
                     ["Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90", "Finishing_Overperf"])
    
    importances = clf.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    print(feat_imp.sort_values("Importance", ascending=False).head(10))
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp.sort_values("Importance", ascending=False).head(10), palette="viridis")
    plt.title("Impact of Finishing Overperformance on PL Success")
    plt.savefig("enhanced_importance.png")
    print("\nSaved chart to 'enhanced_importance.png'")

if __name__ == "__main__":
    main()
