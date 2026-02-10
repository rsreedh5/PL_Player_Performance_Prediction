import pandas as pd
import numpy as np
import os

def get_data():
    """
    Returns a DataFrame containing transfer data with advanced metrics.
    Merges REAL 10-year transfer history (Top 5 Leagues) with SYNTHETIC data for others.
    """
    
    # --- 1. Load Real 10-Year Transfer History ---
    real_players = []
    
    if os.path.exists("Pl_foreign_transfers.csv"):
        print("Integrating 10-Year Premier League Transfer History...")
        history_df = pd.read_csv("Pl_foreign_transfers.csv")
        
        # Map Columns to Model Schema
        # History DF cols: Name, Origin_Season, Origin_League, Origin_Team, Age, Pre_PL_G_A, Pre_PL_npxG, Pre_PL_xA, PL_Debut_Season, PL_Team, PL_Performance, Season_Control
        
        for _, row in history_df.iterrows():
            # Inferred metrics (PrgC/PrgP) are missing in the historical CSV. 
            # We must synthesize them based on position/stats again, or the model will fail.
            # Simplified estimation:
            prgC = row['Pre_PL_xA'] * 8.0 + np.random.normal(1, 0.5) 
            prgP = row['Pre_PL_npxG'] * 5.0 + np.random.normal(1, 0.5)
            
            # Determine Success Label based on ACTUAL PL Performance (Ground Truth)
            if row['PL_Performance'] >= 0.55: verdict = "Star"
            elif row['PL_Performance'] >= 0.30: verdict = "Effective"
            else: verdict = "Flop"
            
            real_players.append({
                "Name": row['Name'],
                "Origin_League": row['Origin_League'].split("-")[-1] if "-" in row['Origin_League'] else row['Origin_League'],
                "Age": np.random.randint(20, 29), # Age missing in Understat export, infer range
                "Transfer_Fee_M": round(row['Pre_PL_npxG'] * 40 + np.random.normal(10, 5), 1), # Historic fee proxy
                "G_A_per_90": round(row['Pre_PL_G_A'], 2),
                "npxG_per_90": round(row['Pre_PL_npxG'], 2),
                "xAG_per_90": round(row['Pre_PL_xA'], 2),
                "PrgC_per_90": round(max(0, prgC), 2),
                "PrgP_per_90": round(max(0, prgP), 2),
                "Buying_Club_Tier": "Mid-Table", # Default for history, can be improved
                "PL_Success": verdict,
                "Season_Control": row['Season_Control']
            })
            
    # --- 2. Synthetic Data Generation (For Missing Leagues) ---
    # We still need synthetic data for Liga NOS, Eredivisie, etc., as Understat doesn't cover them.
    np.random.seed(42)
    n_samples = 1500
    
    leagues = {
        "Liga NOS": 0.65,
        "Eredivisie": 0.60,
        "Brasileirao": 0.55,
        "Austria": 0.45
    }
    
    synthetic_data = []
    
    for _ in range(n_samples):
        league_name = np.random.choice(list(leagues.keys()))
        league_strength = leagues[league_name]
        
        talent = np.random.normal(0.5, 0.15)
        pos_type = np.random.choice([0, 1, 2, 3], p=[0.25, 0.30, 0.20, 0.25])
        
        if talent < 0.4: continue
        
        raw_production = talent / league_strength
        g_a = raw_production + np.random.normal(0, 0.1)
        
        if pos_type == 0: # Striker
            npxG = g_a * 0.75; xAG = g_a * 0.15; prgC = np.random.normal(1.5, 0.5); prgP = np.random.normal(1.0, 0.3)
        elif pos_type == 1: # Winger
            npxG = g_a * 0.40; xAG = g_a * 0.40; prgC = np.random.normal(4.5, 1.0); prgP = np.random.normal(2.5, 0.8)
        elif pos_type == 2: # #10
            npxG = g_a * 0.30; xAG = g_a * 0.60; prgC = np.random.normal(3.5, 0.8); prgP = np.random.normal(4.5, 1.2)
        else: # CM
            npxG = g_a * 0.15; xAG = g_a * 0.30; prgC = np.random.normal(2.5, 0.8); prgP = np.random.normal(6.0, 1.5)
            
        # Add noise
        npxG = np.clip(npxG + np.random.normal(0, 0.05), 0, 1.5)
        xAG = np.clip(xAG + np.random.normal(0, 0.05), 0, 1.0)
        prgC = np.clip(prgC + np.random.normal(0, 1.0), 0, 10.0)
        prgP = np.clip(prgP + np.random.normal(0, 1.0), 0, 10.0)
        
        finishing = np.random.normal(1.0, 0.1)
        final_g_a = (npxG * finishing) + (xAG * finishing)
        
        buying_club_tier = np.random.choice(["Elite", "Upper-Mid", "Mid-Table", "Relegation"], p=[0.2, 0.3, 0.3, 0.2])
        
        # Label Logic
        adaptability = np.random.normal(0, 0.1)
        club_boost = {"Elite": 0.15, "Upper-Mid": 0.05, "Mid-Table": 0.0, "Relegation": -0.1}
        pl_performance_score = talent + adaptability + club_boost[buying_club_tier]
        
        if pl_performance_score >= 0.65: verdict = "Star"
        elif pl_performance_score >= 0.45: verdict = "Effective"
        else: verdict = "Flop"
        
        age = np.random.randint(18, 30)
        hype = final_g_a * 30
        fee = hype + (30-age)*2 + np.random.normal(0, 5)
        if buying_club_tier == "Elite": fee *= 1.5
        fee = np.clip(fee, 5, 150)

        synthetic_data.append({
            "Name": f"Player_{np.random.randint(10000,99999)}",
            "Origin_League": league_name,
            "Age": age,
            "Transfer_Fee_M": round(fee, 2),
            "G_A_per_90": round(final_g_a, 2),
            "npxG_per_90": round(npxG, 2),
            "xAG_per_90": round(xAG, 2),
            "PrgC_per_90": round(prgC, 2),
            "PrgP_per_90": round(prgP, 2),
            "Buying_Club_Tier": buying_club_tier,
            "PL_Success": verdict,
            "Season_Control": 2023 # Default for synthetic current-day prospects
        })
        
    # Combine
    combined_data = real_players + synthetic_data
    df = pd.DataFrame(combined_data)
    
    # Fill missing PL_Success for Real Players using the same logic as synthetic
    # (In a real app, this would be the prediction target, but for training we need labels.
    #  Here we assume their 'stats' reflect their talent directly for simplicity, or we treat them as 'test' data)
    
    def auto_label(row):
        if row['PL_Success'] != "Unknown": return row['PL_Success']
        
        # Heuristic for Real Data labeling (mocking "future" success)
        # Elite PL players have high stats.
        score = row['G_A_per_90'] * 0.4 + row['npxG_per_90'] * 0.4 + row['xAG_per_90'] * 0.2
        if score > 0.6: return "Star"
        if score > 0.35: return "Effective"
        return "Flop"

    df['PL_Success'] = df.apply(auto_label, axis=1)
    
    return df

if __name__ == "__main__":
    print(get_data().head())