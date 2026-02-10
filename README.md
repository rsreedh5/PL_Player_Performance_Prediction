# Soccer Analytics: Transfer Success Predictor (V2)

## 🚀 Version 2.0: The Regression Update
**Date:** February 9, 2026

This project builds a Machine Learning pipeline to predict the success of football players transferring to the Premier League. 

**V2 Major Changes:**
*   **Shift to Regression:** We now predict exact **G+A per 90** output instead of vague "Star/Flop" labels.
*   **Team-Adjusted Metrics:** Introduced `PL_Relative_Performance` to measure if a player improves their team or just rides the wave.
*   **Time-Aware Analysis:** Added `Season_Control` to allow the model to learn from tactical evolution over the last 10 years.
*   **Appearance Filtering:** Models now filter for players with **20+ PL Appearances** to remove noise from injury-prone signings.

## 📊 Key Findings (V2)
Based on 10 years of history (2014-2024):
1.  **Predictive Power:** The best model achieves an **R² of ~0.33** and **RMSE of 0.17** for proven talent.
    *   *Translation:* We can predict a signing's output within ±0.17 G+A/90.
2.  **The "Star" Cliff:** Only **47%** of elite stars maintain their output level in the PL.
3.  **League Tax:** Players from **Ligue 1** and **Serie A** historically face a stricter adaptation curve than those from Bundesliga or La Liga.
4.  **Team Context:** Joining a Relegation-tier team is the single biggest predictor of failure for elite talent.

## 📂 Project Structure

### 1. Data Processing
*   `fetch_history.py`: Scrapes 10 years of data from Understat.
*   `process_filtered_20apps.py`: (V2) Builds the clean dataset, filtering for players who actually played 20+ games.
*   `process_transfers_adjusted.py`: (V2) Calculates team-relative performance.

### 2. Analysis Scripts
*   `analyze_real_regression.py`: **(Main)** The core XGBoost regression model trained on 2014-2021 and tested on 2022+.
*   `analyze_general_on_cohorts.py`: Tests the general model specifically on "Elite" and "Proven" player groups.
*   `analyze_time_aware.py`: Compares "Explicit Season Features" vs "Time Decay" (Season Features won).

### 3. Datasets
*   `Pl_transfers_20apps.csv`: The Gold Standard training set (Filtered).
*   `Pl_foreign_transfers_enhanced.csv`: The full raw transfer history with extra metrics.

## 🛠️ Usage
1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the regression analysis:
    ```bash
    python analyze_real_regression.py
    ```
3.  Run the cohort deep-dive:
    ```bash
    python analyze_filtered_final.py
    ```

## 📈 Visualizations
Check the root directory for generated plots:
*   `regression_importance.png`: Feature importance chart.
*   `regression_scatter.png`: Predicted vs Actual performance.
