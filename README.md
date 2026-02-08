# Soccer Transfer Analytics: Predicting Premier League Success

This project uses machine learning (Random Forest) to predict how attacking players from other leagues will perform when they transfer to the English Premier League (EPL). It analyzes historical transfer data, player age, transfer fees, and league difficulty factors ("League Tax") to generate predictions.

## Project Structure

*   **`analyze_transfers.py`**: The main script. It loads data, performs exploratory data analysis, trains a predictive model, and outputs a comparison of predicted vs. actual performance for notable recent transfers.
*   **`data_generator.py`**: A helper script that generates the dataset. It combines real-world examples (e.g., Haaland, Bruno Fernandes, Antony) with synthetic data to create a large enough training set for the model.
*   **`requirements.txt`**: A list of Python dependencies required to run the project.

## Setup & Installation

1.  **Prerequisites:** Ensure you have Python installed.
2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main analysis script:

```bash
python analyze_transfers.py
```

## Outputs

The script will produce:

1.  **Console Output:**
    *   **Data Overview:** Average stats drop-off by origin league.
    *   **Model Performance:** RMSE (Root Mean Squared Error) and R2 Score.
    *   **Player Analysis Table:** A detailed list of real players, their actual stats, and the model's prediction ("Star", "Effective", or "Flop").

2.  **Plots (Saved as PNG files):**
    *   `retention_by_league.png`: A boxplot showing the "Transfer Tax" — how much output players typically retain when moving from specific leagues (e.g., Eredivisie players often see a larger drop than Bundesliga players).
    *   `fee_vs_success.png`: A scatter plot analyzing if higher transfer fees correlate with actual success.
    *   `prediction_vs_reality.png`: A visualization comparing the model's predictions against what actually happened for key players.

## Methodology

*   **Synthetic Data:** To supplement the small sample size of high-profile transfers, the project generates synthetic player data based on realistic distributions and "league difficulty" coefficients.
*   **Random Forest Regressor:** The model learns non-linear relationships between age, fee, origin league, and pre-EPL statistics to predict EPL Goal+Assist output.
