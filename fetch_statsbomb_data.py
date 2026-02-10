from statsbombpy import sb
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def gather_data():
    print("Checking StatsBomb Free Data Availability...")
    
    # 1. Get Competitions
    try:
        comps = sb.competitions()
    except Exception as e:
        print(f"Error connecting to StatsBomb: {e}")
        return

    # 2. Filter for Top Leagues
    target_leagues = [
        "Premier League", 
        "La Liga", 
        "1. Bundesliga", 
        "Serie A", 
        "Ligue 1"
    ]
    
    # Also statsbomb sometimes names them differently, let's look for partial matches
    # e.g. "La Liga" might be "La Liga", "Bundesliga" might be "1. Bundesliga"
    
    available_data = []
    
    print("\n--- Searching for target leagues ---")
    for index, row in comps.iterrows():
        c_name = row['competition_name']
        c_country = row['country_name']
        season = row['season_name']
        c_id = row['competition_id']
        s_id = row['season_id']
        
        # Check if this competition is one of our targets
        is_target = False
        for target in target_leagues:
            if target in c_name:
                is_target = True
                break
        
        if is_target:
            print(f"Found: {c_name} ({c_country}) - {season} [ID: {c_id}/{s_id}]")
            available_data.append({
                "League": c_name,
                "Season": season,
                "Comp_ID": c_id,
                "Season_ID": s_id
            })

    if not available_data:
        print("\n[!] No free historical data found for the Top 5 Leagues in StatsBomb's public dataset.")
        print("StatsBomb free data typically focuses on specific tournaments (World Cup, Euros) or Messi's career.")
        print("It does NOT contain the complete last 10 years of the Premier League/Bundesliga for free.")
        return

    print(f"\n--- Found {len(available_data)} seasons of interest. Attempting to fetch player stats ---")
    
    all_player_stats = []
    
    for item in available_data:
        print(f"Processing {item['League']} {item['Season']}...")
        
        try:
            # StatsBomb data is event-based (matches -> events). 
            # We need to aggregate it to get "Player Season Stats".
            # This is heavy. We will try to fetch matches first.
            matches = sb.matches(competition_id=item['Comp_ID'], season_id=item['Season_ID'])
            
            # For demonstration and speed, we will process a subset or just list matches
            print(f"  > {len(matches)} matches available.")
            
            # Fetching event data for ALL matches in 10 years would take hours/days and gigabytes.
            # We will grab specific advanced metrics if pre-calculated, but SB doesn't usually pre-calc for free.
            # We will simulate the extraction of "Progressive Carries" from a sample match to show capability.
            
            if not matches.empty:
                # Pick one match to demo aggregation
                match_id = matches.iloc[0]['match_id']
                print(f"  > Pulling events for sample match ID: {match_id}")
                events = sb.events(match_id=match_id)
                
                # Check for Progressive Carries (Carries that move ball 10 yards closer to goal)
                if 'carry' in events.columns:
                    # Logic: end_location x - start_location x etc.
                    # Simplified: just count carries per player
                    carries = events[events['type'] == 'Carry']
                    print(f"  > Found {len(carries)} carries in this match.")
                    
        except Exception as e:
            print(f"  > Error processing season: {e}")

    print("\n--- Summary ---")
    print("StatsBombpy installed and checked.")
    print("As expected, full historical data for Top 5 leagues is likely restricted.")

if __name__ == "__main__":
    gather_data()
