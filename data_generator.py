import pandas as pd
import numpy as np

def get_data():
    """
    Returns a DataFrame containing transfer data.
    Includes some real hardcoded examples and synthetic data for training.
    """
    
    # Real examples (Approximate stats for demonstration)
    # Leagues: 1=Bundesliga, 2=Liga NOS, 3=Eredivisie, 4=Ligue 1, 5=Serie A, 6=La Liga
    # We will use string names for clarity then encode them.
    real_players = [
        {"Name": "Bruno Fernandes", "Origin_League": "Liga NOS", "Age": 25, "Transfer_Fee_M": 68.0, "Pre_PL_G_A_per_90": 1.15, "PL_G_A_per_90": 0.95, "Season": 2020},
        {"Name": "Jadon Sancho", "Origin_League": "Bundesliga", "Age": 21, "Transfer_Fee_M": 73.0, "Pre_PL_G_A_per_90": 1.10, "PL_G_A_per_90": 0.35, "Season": 2021},
        {"Name": "Antony", "Origin_League": "Eredivisie", "Age": 22, "Transfer_Fee_M": 85.0, "Pre_PL_G_A_per_90": 0.85, "PL_G_A_per_90": 0.25, "Season": 2022},
        {"Name": "Erling Haaland", "Origin_League": "Bundesliga", "Age": 21, "Transfer_Fee_M": 51.0, "Pre_PL_G_A_per_90": 1.30, "PL_G_A_per_90": 1.45, "Season": 2022},
        {"Name": "Darwin Nunez", "Origin_League": "Liga NOS", "Age": 22, "Transfer_Fee_M": 64.0, "Pre_PL_G_A_per_90": 1.05, "PL_G_A_per_90": 0.75, "Season": 2022},
        {"Name": "Luis Diaz", "Origin_League": "Liga NOS", "Age": 25, "Transfer_Fee_M": 40.0, "Pre_PL_G_A_per_90": 0.90, "PL_G_A_per_90": 0.65, "Season": 2022},
        {"Name": "Nicolas Pepe", "Origin_League": "Ligue 1", "Age": 24, "Transfer_Fee_M": 72.0, "Pre_PL_G_A_per_90": 0.95, "PL_G_A_per_90": 0.40, "Season": 2019},
        {"Name": "Timo Werner", "Origin_League": "Bundesliga", "Age": 24, "Transfer_Fee_M": 47.0, "Pre_PL_G_A_per_90": 1.10, "PL_G_A_per_90": 0.50, "Season": 2020},
        {"Name": "Kai Havertz", "Origin_League": "Bundesliga", "Age": 21, "Transfer_Fee_M": 70.0, "Pre_PL_G_A_per_90": 0.75, "PL_G_A_per_90": 0.45, "Season": 2020},
        {"Name": "Christian Pulisic", "Origin_League": "Bundesliga", "Age": 20, "Transfer_Fee_M": 58.0, "Pre_PL_G_A_per_90": 0.60, "PL_G_A_per_90": 0.55, "Season": 2019},
        {"Name": "Hakim Ziyech", "Origin_League": "Eredivisie", "Age": 27, "Transfer_Fee_M": 36.0, "Pre_PL_G_A_per_90": 1.05, "PL_G_A_per_90": 0.40, "Season": 2020},
        {"Name": "Sebastien Haller", "Origin_League": "Bundesliga", "Age": 25, "Transfer_Fee_M": 45.0, "Pre_PL_G_A_per_90": 0.85, "PL_G_A_per_90": 0.35, "Season": 2019},
        {"Name": "Takumi Minamino", "Origin_League": "Austria", "Age": 24, "Transfer_Fee_M": 7.25, "Pre_PL_G_A_per_90": 1.10, "PL_G_A_per_90": 0.30, "Season": 2020},
        {"Name": "Cody Gakpo", "Origin_League": "Eredivisie", "Age": 23, "Transfer_Fee_M": 37.0, "Pre_PL_G_A_per_90": 1.30, "PL_G_A_per_90": 0.60, "Season": 2023},
        {"Name": "Alexander Isak", "Origin_League": "La Liga", "Age": 22, "Transfer_Fee_M": 63.0, "Pre_PL_G_A_per_90": 0.65, "PL_G_A_per_90": 0.70, "Season": 2022},
    ]

    # Generate Synthetic Data for background stats
    np.random.seed(42)
    n_samples = 300
    
    leagues = ["Bundesliga", "Liga NOS", "Eredivisie", "Ligue 1", "Serie A", "La Liga", "Other"]
    # League difficulty factors (Higher = Harder to score in -> easier transition to PL potentially? 
    # Actually, we define "Transfer Tax": How much stats drop.
    # Eredivisie has high tax (drop lots). Bundesliga moderate.
    league_tax = {
        "Eredivisie": 0.60, # Retain 40%
        "Austria": 0.30,
        "Liga NOS": 0.65,
        "Bundesliga": 0.70,
        "Ligue 1": 0.75,
        "Serie A": 0.80,
        "La Liga": 0.85,
        "Other": 0.50
    }

    synthetic_data = []
    for _ in range(n_samples):
        league = np.random.choice(leagues)
        age = np.random.randint(18, 30)
        
        # Base talent level (0.2 to 1.0 scale)
        talent = np.random.beta(2, 5) * 1.5 
        
        # Pre PL Stats usually inflated by weaker leagues
        pre_stats = talent / (league_tax.get(league, 0.5) * 0.8) 
        pre_stats += np.random.normal(0, 0.1) # Noise
        pre_stats = np.clip(pre_stats, 0.2, 1.5)

        # Transfer Fee correlates with Age, Stats, and "Hype"
        fee = (pre_stats * 40) + (30 - age) * 2 + np.random.normal(0, 10)
        fee = np.clip(fee, 5, 120)
        
        # PL Stats = Talent + Noise (The harsh reality)
        pl_stats = talent + np.random.normal(0, 0.15)
        pl_stats = np.clip(pl_stats, 0, 1.5)

        synthetic_data.append({
            "Name": f"Player_{np.random.randint(1000,9999)}",
            "Origin_League": league,
            "Age": age,
            "Transfer_Fee_M": round(fee, 2),
            "Pre_PL_G_A_per_90": round(pre_stats, 2),
            "PL_G_A_per_90": round(pl_stats, 2),
            "Season": np.random.randint(2014, 2024)
        })

    df = pd.DataFrame(real_players + synthetic_data)
    return df

if __name__ == "__main__":
    print(get_data().head())
