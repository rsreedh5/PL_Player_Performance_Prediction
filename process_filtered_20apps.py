import pandas as pd
import numpy as np

def build_filtered_dataset():
    print("Building '20+ Appearance' filtered dataset...")
    
    # 1. Load Raw Understat Data
    try:
        raw_df = pd.read_csv("understat_10_years.csv")
    except FileNotFoundError:
        print("Error: 'understat_10_years.csv' not found.")
        return

    # 2. Sort and Shift to find Transfers (Same logic as before)
    raw_df = raw_df.sort_values(by=['player', 'season'])
    transfers = []
    
    grouped = raw_df.groupby('player')
    
    for player, group in grouped:
        if len(group) < 2:
            continue
            
        # Shift to get Next Season data
        group['next_league'] = group['league'].shift(-1)
        group['next_season'] = group['season'].shift(-1)
        group['next_minutes'] = group['minutes'].shift(-1) # Crucial: Next season minutes
        group['next_G_A'] = (group['goals'].shift(-1) + group['assists'].shift(-1)) / (group['minutes'].shift(-1) / 90)
        
        # Current Stats
        group['90s'] = group['minutes'] / 90
        group['G_A_per_90'] = (group['goals'] + group['assists']) / group['90s']
        group['npxG_per_90'] = group['np_xg'] / group['90s']
        group['xA_per_90'] = group['xa'] / group['90s']
        
        # Filter Logic:
        # 1. Moving TO Premier League
        # 2. PLAYED > 1800 minutes (Approx 20 full games) OR > 20 Appearances?
        # Understat csv has 'matches'. Let's use 'matches'.
        group['next_matches'] = group['matches'].shift(-1)
        
        transfer_mask = (
            (group['league'] != 'ENG-Premier League') & 
            (group['next_league'] == 'ENG-Premier League') &
            (group['90s'] > 5.0) & # Decent sample size in origin
            (group['next_matches'] >= 20) # <--- NEW FILTER: 20+ Appearances in PL
        )
        
        player_transfers = group[transfer_mask]
        
        if not player_transfers.empty:
            transfers.append(player_transfers)

    if not transfers:
        print("No transfers found matching strict criteria.")
        return

    transfer_df = pd.concat(transfers)
    
    # 3. Enhance Features (League Quality + Finishing)
    transfer_df['Pre_PL_Expected_GA'] = transfer_df['npxG_per_90'] + transfer_df['xA_per_90']
    transfer_df['Finishing_Overperf'] = transfer_df['G_A_per_90'] - transfer_df['Pre_PL_Expected_GA']
    
    def get_league_weight(league):
        if "La Liga" in league or "Bundesliga" in league: return 1.0
        if "Serie A" in league: return 0.9
        if "Ligue 1" in league: return 0.8
        return 0.7 
        
    transfer_df['League_Quality_Score'] = transfer_df['league'].apply(get_league_weight)
    transfer_df['Season_Control'] = transfer_df['season'].astype(str).str[:2].astype(int) + 2000

    # 4. Save Final Ready-to-Train Dataset
    cols_to_keep = [
        'player', 'season', 'league', 'G_A_per_90', 'npxG_per_90', 'xA_per_90', 
        'Finishing_Overperf', 'League_Quality_Score', 'Season_Control', 'next_G_A'
    ]
    
    final_df = transfer_df[cols_to_keep].rename(columns={
        'player': 'Name',
        'season': 'Origin_Season',
        'league': 'Origin_League',
        'G_A_per_90': 'Pre_PL_G_A',
        'npxG_per_90': 'Pre_PL_npxG',
        'xA_per_90': 'Pre_PL_xA',
        'next_G_A': 'PL_Raw_GA'
    })
    
    final_df.to_csv("Pl_transfers_20apps.csv", index=False)
    print(f"\nSUCCESS: Created filtered dataset with {len(final_df)} players (20+ PL Apps).")

if __name__ == "__main__":
    build_filtered_dataset()
