import pandas as pd
import numpy as np

def get_data():
    """
    Returns a DataFrame containing transfer data.
    Includes real hardcoded examples and enhanced synthetic data.
    """
    
    # Real examples with added context (Buying Club Tier, etc.)
    real_players = [
        {"Name": "Bruno Fernandes", "Origin_League": "Liga NOS", "Age": 25, "Transfer_Fee_M": 68.0, "Pre_PL_G_A_per_90": 1.15, "PL_G_A_per_90": 0.95, "Buying_Club_Tier": "Elite"},
        {"Name": "Jadon Sancho", "Origin_League": "Bundesliga", "Age": 21, "Transfer_Fee_M": 73.0, "Pre_PL_G_A_per_90": 1.10, "PL_G_A_per_90": 0.35, "Buying_Club_Tier": "Elite"},
        {"Name": "Antony", "Origin_League": "Eredivisie", "Age": 22, "Transfer_Fee_M": 85.0, "Pre_PL_G_A_per_90": 0.85, "PL_G_A_per_90": 0.25, "Buying_Club_Tier": "Elite"},
        {"Name": "Erling Haaland", "Origin_League": "Bundesliga", "Age": 21, "Transfer_Fee_M": 51.0, "Pre_PL_G_A_per_90": 1.30, "PL_G_A_per_90": 1.45, "Buying_Club_Tier": "Elite"},
        {"Name": "Darwin Nunez", "Origin_League": "Liga NOS", "Age": 22, "Transfer_Fee_M": 64.0, "Pre_PL_G_A_per_90": 1.05, "PL_G_A_per_90": 0.75, "Buying_Club_Tier": "Elite"},
        {"Name": "Luis Diaz", "Origin_League": "Liga NOS", "Age": 25, "Transfer_Fee_M": 40.0, "Pre_PL_G_A_per_90": 0.90, "PL_G_A_per_90": 0.65, "Buying_Club_Tier": "Elite"},
        {"Name": "Nicolas Pepe", "Origin_League": "Ligue 1", "Age": 24, "Transfer_Fee_M": 72.0, "Pre_PL_G_A_per_90": 0.95, "PL_G_A_per_90": 0.40, "Buying_Club_Tier": "Elite"},
        {"Name": "Timo Werner", "Origin_League": "Bundesliga", "Age": 24, "Transfer_Fee_M": 47.0, "Pre_PL_G_A_per_90": 1.10, "PL_G_A_per_90": 0.50, "Buying_Club_Tier": "Elite"},
        {"Name": "Kai Havertz", "Origin_League": "Bundesliga", "Age": 21, "Transfer_Fee_M": 70.0, "Pre_PL_G_A_per_90": 0.75, "PL_G_A_per_90": 0.45, "Buying_Club_Tier": "Elite"},
        {"Name": "Christian Pulisic", "Origin_League": "Bundesliga", "Age": 20, "Transfer_Fee_M": 58.0, "Pre_PL_G_A_per_90": 0.60, "PL_G_A_per_90": 0.55, "Buying_Club_Tier": "Elite"},
        {"Name": "Hakim Ziyech", "Origin_League": "Eredivisie", "Age": 27, "Transfer_Fee_M": 36.0, "Pre_PL_G_A_per_90": 1.05, "PL_G_A_per_90": 0.40, "Buying_Club_Tier": "Elite"},
        {"Name": "Sebastien Haller", "Origin_League": "Bundesliga", "Age": 25, "Transfer_Fee_M": 45.0, "Pre_PL_G_A_per_90": 0.85, "PL_G_A_per_90": 0.35, "Buying_Club_Tier": "Mid-Table"},
        {"Name": "Takumi Minamino", "Origin_League": "Austria", "Age": 24, "Transfer_Fee_M": 7.25, "Pre_PL_G_A_per_90": 1.10, "PL_G_A_per_90": 0.30, "Buying_Club_Tier": "Elite"},
        {"Name": "Cody Gakpo", "Origin_League": "Eredivisie", "Age": 23, "Transfer_Fee_M": 37.0, "Pre_PL_G_A_per_90": 1.30, "PL_G_A_per_90": 0.60, "Buying_Club_Tier": "Elite"},
        {"Name": "Alexander Isak", "Origin_League": "La Liga", "Age": 22, "Transfer_Fee_M": 63.0, "Pre_PL_G_A_per_90": 0.65, "PL_G_A_per_90": 0.70, "Buying_Club_Tier": "Upper-Mid"},
    ]

    # Generate Synthetic Data for background stats
    np.random.seed(42)
    n_samples = 1000  # Increased sample size
    
    leagues = ["Bundesliga", "Liga NOS", "Eredivisie", "Ligue 1", "Serie A", "La Liga", "Austria"]
    club_tiers = ["Elite", "Upper-Mid", "Mid-Table", "Relegation"]
    
    # Updated League Strength (Coefficient): Proxy for average defensive quality
    # 1.0 = PL Level, Lower = Weaker League (Easier to score)
    league_strength = {
        "La Liga": 0.90,
        "Serie A": 0.85,
        "Bundesliga": 0.80,
        "Ligue 1": 0.75,
        "Liga NOS": 0.65,
        "Eredivisie": 0.55,
        "Austria": 0.40
    }

    synthetic_data = []
    for _ in range(n_samples):
        league = np.random.choice(leagues)
        buying_tier = np.random.choice(club_tiers, p=[0.15, 0.25, 0.40, 0.20])
        age = np.random.randint(18, 32)
        
        # 1. True Talent Generation (Latent Variable)
        # Distributed normally, but elite clubs buy better players
        base_talent = np.random.normal(0.5, 0.15)
        if buying_tier == "Elite": base_talent += 0.2
        if buying_tier == "Upper-Mid": base_talent += 0.1
        
        # 2. League Stats Generation (Reverse Engineering)
        # Weaker league = higher stats for same talent
        strength_factor = league_strength[league]
        pre_stats = base_talent / strength_factor
        pre_stats += np.random.normal(0, 0.1) # Statistical noise
        pre_stats = np.clip(pre_stats, 0.2, 1.8)

        # 3. Adaptability Factor (Hidden Variable)
        # Some players just fit, some don't.
        adaptability = np.random.normal(0, 0.1) 
        
        # 4. PL Stats Generation
        # Talent + Adaptability + Club Context Bonus (Better teams create more chances)
        club_context = {"Elite": 0.1, "Upper-Mid": 0.05, "Mid-Table": 0.0, "Relegation": -0.05}
        pl_stats = base_talent + adaptability + club_context[buying_tier] + np.random.normal(0, 0.05)
        pl_stats = np.clip(pl_stats, 0.0, 1.5)

        # 5. Decoupled Transfer Fee Generation
        # Fee = Stats + Hype + English Tax (if applicable) + Buying Club Wealth
        hype = np.random.choice([0, 10, 30], p=[0.7, 0.2, 0.1]) # Occasional massive hype
        wealth_premium = {"Elite": 20, "Upper-Mid": 10, "Mid-Table": 5, "Relegation": 0}
        
        fee = (pre_stats * 25) + wealth_premium[buying_tier] + hype + (28 - age)
        fee += np.random.normal(0, 5)
        fee = np.clip(fee, 2, 150)

        synthetic_data.append({
            "Name": f"Player_{np.random.randint(10000,99999)}",
            "Origin_League": league,
            "Age": age,
            "Transfer_Fee_M": round(fee, 2),
            "Pre_PL_G_A_per_90": round(pre_stats, 2),
            "PL_G_A_per_90": round(pl_stats, 2),
            "Buying_Club_Tier": buying_tier
        })

    df = pd.DataFrame(real_players + synthetic_data)
    return df

if __name__ == "__main__":
    print(get_data().head())