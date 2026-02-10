"""
# Enrich Understat Free
Enriches the transfer dataset with Shots/90, KeyPasses/90, and League Averages using free Understat data.

## Usage
Run with:
    python enrich_understat_free.py
"""

import pandas as pd
import numpy as np
import aiohttp
import asyncio
from understat import Understat
import sys

# Constants
LEAGUE_MAPPING = {
    "ENG-Premier League": "EPL",
    "Premier League": "EPL",
    "ESP-La Liga": "La_liga",
    "La Liga": "La_liga",
    "GER-Bundesliga": "Bundesliga",
    "Bundesliga": "Bundesliga",
    "ITA-Serie A": "Serie_A",
    "Serie A": "Serie_A",
    "FRA-Ligue 1": "Ligue_1",
    "Ligue 1": "Ligue_1"
}

CACHE = {}

def clean_player_name(name):
    """Normalize player names for matching."""
    try:
        return name.strip().lower()
    except:
        return ""

async def fetch_understat_season(understat, league_name, season_year):
    """
    Fetches season data using understat library.
    """
    cache_key = (league_name, season_year)
    if cache_key in CACHE:
        return CACHE[cache_key]

    print(f"Fetching {league_name} {season_year} via Understat library...")
    try:
        data = await understat.get_league_players(league_name, season_year)
        CACHE[cache_key] = data
        return data
    except Exception as e:
        print(f"Error fetching {league_name} {season_year}: {e}")
        return None

def calculate_league_averages(players_data):
    """
    Calculates average xG, xA, Shots, KP per 90 for the league.
    Filter: Minutes > 450
    """
    if not players_data:
        return None
    
    filtered_stats = []
    
    for p in players_data:
        try:
            # Understat library returns dicts with strings, usually
            minutes = int(p['time'])
            if minutes > 450:
                n90 = minutes / 90.0
                filtered_stats.append({
                    'xG_p90': float(p['xG']) / n90,
                    'xA_p90': float(p['xA']) / n90,
                    'Shots_p90': float(p['shots']) / n90,
                    'KP_p90': float(p['key_passes']) / n90
                })
        except (ValueError, KeyError, ZeroDivisionError):
            continue
            
    if not filtered_stats:
        return None
        
    df_stats = pd.DataFrame(filtered_stats)
    return {
        'Origin_League_Avg_xG_p90': df_stats['xG_p90'].mean(),
        'Origin_League_Avg_xA_p90': df_stats['xA_p90'].mean(),
        'Origin_League_Avg_Shots_p90': df_stats['Shots_p90'].mean(),
        'Origin_League_Avg_KP_p90': df_stats['KP_p90'].mean()
    }

async def enrich_data():
    input_file = "Pl_transfers_20apps.csv"
    output_file = "Pl_transfers_20apps_enriched.csv"
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    print(f"Loaded {len(df)} rows from {input_file}")
    
    # Initialize new columns
    df['Pre_PL_Shots_p90'] = np.nan
    df['Pre_PL_KeyPasses_p90'] = np.nan
    df['Origin_League_Avg_xG_p90'] = np.nan
    df['Origin_League_Avg_xA_p90'] = np.nan
    df['Origin_League_Avg_Shots_p90'] = np.nan
    df['Origin_League_Avg_KP_p90'] = np.nan

    missing_players = []
    enriched_count = 0
    
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        # Prefetch needed seasons
        needed_seasons = df[['Origin_League', 'Season_Control']].drop_duplicates()
        
        for _, row in needed_seasons.iterrows():
            league_raw = row['Origin_League']
            season = int(row['Season_Control'])
            
            league_code = None
            for k, v in LEAGUE_MAPPING.items():
                if k in league_raw:
                    league_code = v
                    break
            
            if league_code:
                await fetch_understat_season(understat, league_code, season)
            else:
                print(f"Skipping unknown league format: {league_raw}")

        # Process rows
        for idx, row in df.iterrows():
            league_raw = row['Origin_League']
            season = int(row['Season_Control'])
            player_name = row['Name']
            
            league_code = None
            for k, v in LEAGUE_MAPPING.items():
                if k in league_raw:
                    league_code = v
                    break
            
            if not league_code:
                continue
                
            players_data = CACHE.get((league_code, season))
            if not players_data:
                continue
                
            # 1. League Averages
            avgs = calculate_league_averages(players_data)
            if avgs:
                df.at[idx, 'Origin_League_Avg_xG_p90'] = avgs['Origin_League_Avg_xG_p90']
                df.at[idx, 'Origin_League_Avg_xA_p90'] = avgs['Origin_League_Avg_xA_p90']
                df.at[idx, 'Origin_League_Avg_Shots_p90'] = avgs['Origin_League_Avg_Shots_p90']
                df.at[idx, 'Origin_League_Avg_KP_p90'] = avgs['Origin_League_Avg_KP_p90']
            
            # 2. Player Stats
            target_clean = clean_player_name(player_name)
            found_player = None
            
            # Exact match
            for p in players_data:
                if clean_player_name(p['player_name']) == target_clean:
                    found_player = p
                    break
            
            # Substring match
            if not found_player:
                for p in players_data:
                    p_clean = clean_player_name(p['player_name'])
                    if target_clean in p_clean or p_clean in target_clean:
                        found_player = p
                        break
            
            if found_player:
                try:
                    minutes = float(found_player['time'])
                    if minutes > 0:
                        n90 = minutes / 90.0
                        df.at[idx, 'Pre_PL_Shots_p90'] = float(found_player['shots']) / n90
                        df.at[idx, 'Pre_PL_KeyPasses_p90'] = float(found_player['key_passes']) / n90
                        enriched_count += 1
                except:
                    pass
            else:
                missing_players.append(f"{player_name} ({league_code} {season})")

    # Save
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("ENRICHMENT SUMMARY")
    print("="*50)
    print(f"Rows Enriched: {enriched_count}/{len(df)}")
    print(f"Missing Players: {len(missing_players)}")
    print("\nTop 10 Missing Players:")
    for m in missing_players[:10]:
        print(f" - {m}")
    
    print(f"\nSaved enriched dataset to {output_file}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(enrich_data())