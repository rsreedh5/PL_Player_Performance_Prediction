import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_generator import get_data

# Set visual style
sns.set_theme(style="whitegrid")

def categorize_performance(g_a_per_90):
    if g_a_per_90 >= 0.55: return "Star"
    if g_a_per_90 >= 0.30: return "Effective"
    return "Flop"

def main():
    print("Loading Enhanced Data...")
    df = get_data()
    
    # --- 1. Target Engineering ---
    # We are now predicting a CLASS, not a number.
    df["Verdict"] = df["PL_G_A_per_90"].apply(categorize_performance)
    
    print("\n--- Class Distribution ---")
    print(df["Verdict"].value_counts(normalize=True))

    # --- 2. Feature Engineering ---
    # Calculate the 'Delta' that ChatGPT suggested, but for analysis, not direct training target
    # (Since we are doing classification, we keep raw features but the model will find the delta patterns)
    
    X = df[["Origin_League", "Age", "Transfer_Fee_M", "Pre_PL_G_A_per_90", "Buying_Club_Tier"]]
    y = df["Verdict"]

    # --- 3. Pipeline Setup ---
    # Categorical: League, Club Tier
    # Numerical: Age, Fee, Pre-Stats
    
    categorical_features = ["Origin_League", "Buying_Club_Tier"]
    numerical_features = ["Age", "Transfer_Fee_M", "Pre_PL_G_A_per_90"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # Random Forest Classifier
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ])

    # --- 4. Cross-Validation (Robustness Check) ---
    print("\n--- Running 5-Fold Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    
    print(f"CV Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # --- 5. Training Final Model ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    # --- 6. Evaluation ---
    y_pred = clf.predict(X_test)
    print("\n--- Test Set Evaluation ---")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    print("\n--- Feature Importance ---")
    feature_names = (clf.named_steps['preprocessor']
                     .transformers_[0][1]
                     .get_feature_names_out(categorical_features).tolist() + numerical_features)
    
    importances = clf.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    print(feat_imp.sort_values("Importance", ascending=False).head(10))

    # --- 7. Real Player Analysis ---
    print("\n--- Real World Predictions ---")
    real_players_df = df[~df["Name"].str.startswith("Player_")].copy()
    
    # Predict probabilities for nuance
    real_X = real_players_df[X.columns]
    predicted_classes = clf.predict(real_X)
    predicted_probs = clf.predict_proba(real_X)
    
    real_players_df["Predicted_Verdict"] = predicted_classes
    
    # Get probability of being a Star
    star_idx = list(clf.classes_).index("Star")
    real_players_df["Star_Prob"] = predicted_probs[:, star_idx]
    
    cols = ["Name", "Origin_League", "Buying_Club_Tier", "Pre_PL_G_A_per_90", "PL_G_A_per_90", "Predicted_Verdict", "Verdict"]
    display_df = real_players_df[cols].sort_values("PL_G_A_per_90", ascending=False)
    
    print(display_df.to_string(index=False))

    # Save to CSV
    display_df.to_csv("final_predictions.csv", index=False)
    print("\nSaved detailed predictions to 'final_predictions.csv'")

    # --- 8. Visualizations ---
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Player Success Classification')
    plt.savefig("confusion_matrix.png")
    
    # Star Probability vs Actual
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=real_players_df, x="Star_Prob", y="PL_G_A_per_90", hue="Verdict", style="Verdict", s=150, palette="deep")
    
    for i, row in real_players_df.iterrows():
        plt.text(row["Star_Prob"]+0.01, row["PL_G_A_per_90"], row["Name"], fontsize=9)
        
    plt.axvline(0.5, color='gray', linestyle='--')
    plt.title("Model Confidence (Star Probability) vs. Reality")
    plt.xlabel("Estimated Probability of being a 'Star'")
    plt.ylabel("Actual PL G+A/90")
    plt.savefig("star_probability_analysis.png")
    print("Plots saved: confusion_matrix.png, star_probability_analysis.png")

if __name__ == "__main__":
    main()
