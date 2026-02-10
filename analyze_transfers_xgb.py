import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from data_generator import get_data

# Set visual style
sns.set_theme(style="whitegrid")

def main():
    print("Loading Data for Gradient Boosting Analysis...")
    df = get_data()
    
    # --- 1. Label Encoding for Target ---
    # XGBoost needs integer labels (0, 1, 2)
    le = LabelEncoder()
    df["Verdict_Encoded"] = le.fit_transform(df["PL_Success"])
    class_names = list(le.classes_)
    print(f"Target Classes: {class_names}")
    
    print("\n--- Class Distribution ---")
    print(df["PL_Success"].value_counts(normalize=True))

    # --- 2. Feature Engineering ---
    feature_cols = [
        "Origin_League", 
        "Age", 
        "Transfer_Fee_M", 
        "G_A_per_90", 
        "npxG_per_90", 
        "xAG_per_90", 
        "PrgC_per_90", 
        "PrgP_per_90", 
        "Buying_Club_Tier"
    ]
    
    X = df[feature_cols]
    y = df["Verdict_Encoded"]

    # --- 3. Pipeline Setup ---
    categorical_features = ["Origin_League", "Buying_Club_Tier"]
    numerical_features = ["Age", "Transfer_Fee_M", "G_A_per_90", "npxG_per_90", "xAG_per_90", "PrgC_per_90", "PrgP_per_90"]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ]
    )

    # XGBoost Classifier
    # Use 'multi:softprob' for multi-class classification
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample=0.8,
        objective='multi:softprob',
        num_class=len(class_names),
        random_state=42,
        eval_metric='mlogloss'
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])

    # --- 4. Cross-Validation ---
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
    
    # Decode labels for report
    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)
    
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    # --- 7. Feature Importance ---
    # XGBoost provides importance natively
    print("\n--- Feature Importance ---")
    feature_names = (clf.named_steps['preprocessor']
                     .transformers_[0][1]
                     .get_feature_names_out(categorical_features).tolist() + numerical_features)
    
    importances = clf.named_steps['classifier'].feature_importances_
    feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    print(feat_imp.sort_values("Importance", ascending=False).head(12))

    # --- 8. Real Player Analysis ---
    print("\n--- Real World Predictions (XGBoost) ---")
    real_players_df = df[~df["Name"].str.startswith("Player_")].copy()
    
    # Predict probabilities
    real_X = real_players_df[X.columns]
    predicted_probs = clf.predict_proba(real_X)
    
    # Get probability of being a Star
    # Need to find index of "Star" in the encoded classes
    star_class_idx = list(le.classes_).index("Star")
    real_players_df["Star_Prob"] = predicted_probs[:, star_class_idx]
    
    # Decode predicted class
    real_players_df["Predicted_Verdict"] = le.inverse_transform(clf.predict(real_X))
    
    cols = ["Name", "Origin_League", "G_A_per_90", "npxG_per_90", "Predicted_Verdict", "PL_Success", "Star_Prob"]
    display_df = real_players_df[cols].sort_values("Star_Prob", ascending=False)
    
    print(display_df.to_string(index=False))

    # Save to CSV
    display_df.to_csv("final_predictions_xgb.csv", index=False)
    print("\nSaved detailed predictions to 'final_predictions_xgb.csv'")

    # --- 9. Visualizations ---
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: XGBoost Success Classification')
    plt.savefig("confusion_matrix_xgb.png")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp.sort_values("Importance", ascending=False).head(10), palette="magma")
    plt.title("XGBoost: Key Predictors for Transfer Success")
    plt.savefig("feature_importance_xgb.png")
    print("Plots saved: confusion_matrix_xgb.png, feature_importance_xgb.png")

if __name__ == "__main__":
    main()
