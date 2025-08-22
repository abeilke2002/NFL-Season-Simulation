import pandas as pd
import nfl_data_py as nfl
from nfl_data_py import import_schedules
import numpy as np
from collections import defaultdict
import arviz as az
import pymc as pm
import joblib
import ssl
import os
import contextlib
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import t

def preprocess_data(years):
    '''
    Parameters:
        years (list of ints): Example: [2020, 2021, 2022]
    
    Returns:
        final_scores (DataFrame): Game-level data with pregame stats for game-level modeling
    '''

    # Load play-by-play data
    pbp = nfl.import_pbp_data(years)

    pbp = pbp[pbp['season_type'] == 'REG'].copy()

    # Create final scores table
    final_scores = (
        pbp.groupby("game_id")
        .agg({
            "home_team": "first",
            "away_team": "first",
            "home_score": "max",
            "away_score": "max",
            "game_date": "first"
        })
        .reset_index()
    )

    # Create win/loss column
    final_scores["home_win"] = (final_scores["home_score"] > final_scores["away_score"]).astype(int)

    # Make game_date datetime and extract year
    final_scores['game_date'] = pd.to_datetime(final_scores['game_date'])
    final_scores['year'] = final_scores['game_date'].dt.year

    # Aggregate team-level stats for each game
    team_game_stats = (
        pbp.groupby(['game_id', 'posteam'], as_index=False)
        .agg({
            'air_yards': 'sum',
            'passing_yards': 'sum',
            'rushing_yards': 'sum',
            'epa': 'sum',
            'pass_attempt': 'sum',
            'rush_attempt': 'sum',
            'interception': 'sum',
            'fumble_lost': 'sum'
        })
        .assign(plays=pbp.groupby(['game_id', 'posteam']).size().values)
        .assign(epa_per_play=lambda df: (df['epa'] / df['plays']).round(4))
        .rename(columns={'posteam': 'team'})
    )


    # Create long-format table of team-game appearances with points scored and allowed
    home = final_scores[['game_id', 'game_date', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'points', 'away_score': 'points_allowed'})
    away = final_scores[['game_id', 'game_date', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'points', 'home_score': 'points_allowed'})

    team_games = pd.concat([home, away], ignore_index=True)
    team_games['year'] = team_games['game_date'].dt.year
    team_games['season'] = team_games['game_date'].apply(
        lambda d: d.year if d >= pd.Timestamp(f"{d.year}-08-30") else d.year - 1
        )
    # Merge in aggregated play-by-play stats
    team_games = team_games.merge(team_game_stats, on=['game_id', 'team'], how='left')

    # Sort and compute pregame cumulative averages
    team_games = team_games.sort_values(['team', 'year', 'game_date'])

    stat_cols = ['points', 'points_allowed', 'air_yards', 'passing_yards', 'rushing_yards', 'epa_per_play',
             'pass_attempt', 'rush_attempt', 'interception', 'fumble_lost']

    for stat in stat_cols:
        team_games[f'{stat}'] = (
            team_games
            .groupby(['team', 'season'])[stat]
            .transform(lambda x: x.shift(1).expanding().mean().round(2))
        )

    # Final stat columns to keep
    stat_features = [f'{stat}' for stat in stat_cols]

    # Merge back into final_scores for home team
    final_scores = final_scores.merge(
        team_games[['game_id', 'team'] + stat_features],
        left_on=['game_id', 'home_team'],
        right_on=['game_id', 'team'],
        how='left'
    ).rename(columns={col: f'home_{col}' for col in stat_features}).drop(columns='team')

    # Merge back into final_scores for away team
    final_scores = final_scores.merge(
        team_games[['game_id', 'team'] + stat_features],
        left_on=['game_id', 'away_team'],
        right_on=['game_id', 'team'],
        how='left'
    ).rename(columns={col: f'away_{col}' for col in stat_features}).drop(columns='team')

    # Drop games with missing pregame data (first game of season per team)
    final_scores = final_scores.dropna().sort_values('game_date').reset_index(drop=True)

    final_scores['home_team_won'] = np.where(final_scores['home_score'] > final_scores['away_score'], 1, 0)
    final_scores['away_team_won'] = np.where(final_scores['home_score'] < final_scores['away_score'], 1, 0)

    team_wins = defaultdict(int)

    home_team_wins = []
    away_team_wins = []

    final_scores = final_scores.dropna().sort_values('game_date').reset_index(drop=True)
    final_scores['season'] = final_scores['game_date'].apply(
        lambda d: d.year if d >= pd.Timestamp(f"{d.year}-08-30") else d.year - 1
    )
    # Group by season
    for season, group in final_scores.groupby('season'):
        team_wins = {}  # reset win counter per season
        
        for _, row in group.iterrows():
            home = row['home_team']
            away = row['away_team']
            
            # Initialize win counters for new teams in this season
            if home not in team_wins:
                team_wins[home] = 0
            if away not in team_wins:
                team_wins[away] = 0
            
            # Record current wins before this game
            home_team_wins.append(team_wins[home])
            away_team_wins.append(team_wins[away])
            
            # Update win counters after game result
            if row['home_team_won'] == 1:
                team_wins[home] += 1
            elif row['away_team_won'] == 1:
                team_wins[away] += 1

    # Add results back to the DataFrame
    final_scores['home_team_wins'] = home_team_wins
    final_scores['away_team_wins'] = away_team_wins


    return final_scores

def bayes_modeling_process(years):
    ''' 
    This function is designed to get all 3 variables (outside of wins)
    into the dataset for bayesian modeling
    '''
    
    # Load play-by-play data
    pbp = nfl.import_pbp_data(years)

    # only regular season games
    pbp = pbp[pbp['season_type'] == 'REG'].copy()

    # Create final scores table
    final_scores = (
        pbp.groupby("game_id")
        .agg({
            "home_team": "first",
            "away_team": "first",
            "home_score": "max",
            "away_score": "max",
            "game_date": "first"
        })
        .reset_index()
    )
    # Get points allowed
    final_scores['home_points_allowed'] = final_scores['away_score']
    final_scores['away_points_allowed'] = final_scores['home_score']

    # Make game_date datetime and extract year
    final_scores['game_date'] = pd.to_datetime(final_scores['game_date'])
    final_scores['year'] = final_scores['game_date'].dt.year

    final_scores['season'] = final_scores['game_date'].apply(
        lambda d: d.year if d >= pd.Timestamp(f"{d.year}-08-30") else d.year - 1
    )

    ## Get team game stats

    team_game_stats = (
        pbp.groupby(['game_id', 'posteam'], as_index=False)
        .agg({
            'epa': 'sum',
        })
        .assign(plays=pbp.groupby(['game_id', 'posteam']).size().values)
        .assign(epa_per_play=lambda df: (df['epa'] / df['plays']).round(4))
        .rename(columns={'posteam': 'team'})
    )

        # Melt final_scores into long format (one row per team per game)
    home_df = final_scores.rename(columns={
        'home_team': 'team',
        'home_score': 'points_scored',
        'away_score': 'points_allowed',
    }).assign(home_away='home')[['game_id', 'team', 'points_scored', 'points_allowed', 'game_date', 'year', 'season', 'home_away']]

    away_df = final_scores.rename(columns={
        'away_team': 'team',
        'away_score': 'points_scored',
        'home_score': 'points_allowed',
    }).assign(home_away='away')[['game_id', 'team', 'points_scored', 'points_allowed', 'game_date', 'year', 'season', 'home_away']]

    long_final_scores = pd.concat([home_df, away_df], ignore_index=True)

    # Merge on game_id and team
    final_with_epa = pd.merge(
        long_final_scores,
        team_game_stats,
        on=['game_id', 'team'],
        how='left'
    )

    final_scores_with_epa = final_with_epa.pivot(index='game_id', columns='home_away')
    final_scores_with_epa.columns = ['_'.join(col).strip() for col in final_scores_with_epa.columns.values]

    # Reset index
    final_scores_with_epa = final_scores_with_epa.reset_index()

    final_scores_with_epa = final_scores_with_epa.rename(columns={
        'team_home': 'home_team',
        'points_scored_home': 'home_score',
        'points_allowed_home': 'home_points_allowed',
        'epa_home': 'home_epa',
        'team_away': 'away_team',
        'points_scored_away': 'away_score',
        'points_allowed_away': 'away_points_allowed',
        'epa_away': 'away_epa',
        'game_date_home': 'game_date',  # identical across both
        'year_home': 'year',
        'season_home': 'season',
    })
    
    return final_scores_with_epa

def generate_season_win_totals(df, year_predicting, simulations = 1000, val = 0.5):
    '''
    Parameters:
        df - after running through generate_pred_stats
        year_predicting - season you are generating wins for
        simulations - times simulating season, defaulted to 1000
        
    Returns:
        all_wins - dataframe with the teams projected win totals by mean and median in that season
    '''
    
    
    schedules = import_schedules([year_predicting])
    schedules = schedules[['home_team', 'away_team', 'week']]

    # no win diff
    # features = ['home_points', 'away_points', 'home_epa_per_play', 'away_epa_per_play', 
    #            'home_points_allowed', 'away_points_allowed', 'ppg_diff', 'epa_diff', 'win_diff',
    #              'ppga_diff','home_elo', 'away_elo']

    features = ['win_diff', 'ppg_diff', 'epa_diff', 'ppga_diff', 'home_elo', 'away_elo', 'exp_ppg_ppga_home', 'exp_ppg_ppga_away']
    home_feats = ['home_points', 'home_epa_per_play', 'away_points_allowed', 'home_elo']
    away_feats = ['away_points', 'away_epa_per_play', 'home_points_allowed', 'away_elo']

    weeks = np.sort(schedules['week'].unique())
    weeks = weeks[weeks < 19]

    teams = schedules['home_team'].unique()
    home_points_model = joblib.load('season_simulation/game_modeling/best_rf_home_points_model.pkl')
    away_points_model = joblib.load('season_simulation/game_modeling/best_rf_away_points_model.pkl')

    all_wins = pd.DataFrame(index=teams)
    model = joblib.load('season_simulation/game_modeling/best_rf_model.pkl')

    team_modeled_stats = df[['team', 'ppg_pred', 'ppg_pred_std', 'epapp_pred', 'epapp_pred_std', 'ppga_pred', 'ppga_pred_std']]
    home_stats = team_modeled_stats.rename(columns=lambda x: f'home_{x}' if x != 'team' else 'home_team')
    away_stats = team_modeled_stats.rename(columns=lambda x: f'away_{x}' if x != 'team' else 'away_team')
    schedules = schedules.merge(home_stats, on='home_team', how='left')
    schedules = schedules.merge(away_stats, on='away_team', how='left')

    sim_results = []

    for sim in range(simulations):
        # Re-generate random stats for this simulation
        # Make a fresh copy of schedules so you don’t overwrite original data
        sim_sched = schedules.copy()

        sim_sched['home_points'] = np.random.normal(
                loc=sim_sched['home_ppg_pred'],
                scale=sim_sched['home_ppg_pred_std']
            )
        sim_sched['home_epa_per_play'] = np.random.normal(
                loc=sim_sched['home_epapp_pred'],
                scale=sim_sched['home_epapp_pred_std']
            )
        sim_sched['home_points_allowed'] = np.random.normal(
                loc=sim_sched['home_ppga_pred'],
                scale=sim_sched['home_ppga_pred_std']
            )

        sim_sched['away_points'] = np.random.normal(
                loc=sim_sched['away_ppg_pred'],
                scale=sim_sched['away_ppg_pred_std']
            )
        sim_sched['away_epa_per_play'] = np.random.normal(
                loc=sim_sched['away_epapp_pred'],
                scale=sim_sched['away_epapp_pred_std']
            )
        sim_sched['away_points_allowed'] = np.random.normal(
                loc=sim_sched['away_ppga_pred'],
                scale=sim_sched['away_ppga_pred_std']
            )
        sim_sched['home_team_wins'] = 0
        sim_sched['away_team_wins'] = 0

        sim_sched['ppg_diff'] = sim_sched['home_points'] - sim_sched['away_points']
        sim_sched['epa_diff'] = sim_sched['home_epa_per_play'] - sim_sched['away_epa_per_play']
        sim_sched['win_diff'] = sim_sched['home_team_wins'] - sim_sched['away_team_wins']
        sim_sched['ppga_diff'] = sim_sched['home_points_allowed'] - sim_sched['away_points_allowed']

        ## Home Elo
        sim_sched['home_epa_score'] = (
            (sim_sched['home_epa_per_play'] - sim_sched['home_epa_per_play'].mean()) /
            sim_sched['home_epa_per_play'].std()
        )

        sim_sched['home_point_score'] = (
            (sim_sched['home_points'] - sim_sched['home_points_allowed'] - 
            (sim_sched['home_points'] - sim_sched['home_points_allowed']).mean()) /
            (sim_sched['home_points'] - sim_sched['home_points_allowed']).std()
        )

        sim_sched['home_elo'] = sim_sched['home_epa_score'] + (sim_sched['home_point_score'] * 1.5)

        ## Away Elo
        sim_sched['away_epa_score'] = (
            (sim_sched['away_epa_per_play'] - sim_sched['away_epa_per_play'].mean()) /
            sim_sched['away_epa_per_play'].std()
        )

        sim_sched['away_point_score'] = (
        (sim_sched['away_points'] - sim_sched['away_points_allowed'] - 
            (sim_sched['away_points'] - sim_sched['away_points_allowed']).mean()) /
            (sim_sched['away_points'] - sim_sched['away_points_allowed']).std()
        )

        sim_sched['away_elo'] = sim_sched['away_epa_score'] + (sim_sched['away_point_score'] * 1.5)

        sim_sched['exp_ppg_ppga_home'] = np.where(
        sim_sched['home_points_allowed'] == 0,
        np.nan,
        sim_sched['home_points'] / sim_sched['home_points_allowed']
        )

        sim_sched['exp_ppg_ppga_away'] = np.where(
        sim_sched['away_points_allowed'] == 0,
        np.nan,
        sim_sched['away_points'] / sim_sched['away_points_allowed']
        )

        team_wins = {team: 0 for team in teams}
        weeks = np.sort(sim_sched['week'].unique())
        weeks = weeks[weeks < 19]

        for week in weeks:
            df = sim_sched[sim_sched['week'] == week].copy()

            df['home_team_wins'] = df['home_team'].map(team_wins)
            df['away_team_wins'] = df['away_team'].map(team_wins)
            df['win_diff'] = df['home_team_wins'] - df['away_team_wins']

            # Amplified probabilities
            df['home_win_prob'] = model.predict_proba(df[features])[:, 1]
            df['home_pred_points'] = home_points_model.predict(df[home_feats])
            df['away_pred_points'] = away_points_model.predict(df[away_feats])

                # --- Hybrid Winner Selection ---
            probs = df['home_win_prob'].values
            rand = np.random.rand(len(df))
            confidence = np.abs(probs - val) * 10 

            deterministic = np.where(probs > val, df['home_team'], df['away_team'])
            opposite = np.where(probs > val, df['away_team'], df['home_team'])
            use_deterministic = rand < np.clip(confidence, 0, 1)

            df['winner'] = np.where(use_deterministic, deterministic, opposite)

            # df['random_draw'] = np.random.rand(len(df))
            # df['winner'] = np.where(df['random_draw'] < df['home_win_prob'], df['home_team'], df['away_team'])
            # df['winner'] = np.where(df['home_win_prob'] > val, df['home_team'], df['away_team'])
            # df['winner'] = np.where(df['home_pred_points'] > df['away_pred_points'], df['home_team'], df['away_team'])

            for _, row in df.iterrows():
                team_wins[row['winner']] += 1

        sim_results.append(pd.Series(team_wins, name=f'sim_{sim + 1}'))

    # Row-wise mean/median
    all_wins = pd.concat(sim_results, axis=1)

    # Add stats
    all_wins['row_mean'] = all_wins.mean(axis=1)
    all_wins['row_median'] = all_wins.median(axis=1)
    row_modes = all_wins.mode(axis=1)
    all_wins['row_mode'] = row_modes.iloc[:, 0]
    all_wins['season'] = year_predicting

    # Reset index
    all_wins = all_wins.reset_index(names='team')

    return all_wins

def get_regression_metrics(df):
    '''
    Parameters:
        df - with both years predicted stats and actual stats
    
    Returns:
        MSE, RMSE, MAE - metrics for each of the 3 predicted stats
    '''

    stats = ["ppg", "epapp", "ppga"]

    for stat in stats:
        mse = mean_squared_error(df[f"{stat}"], df[f"{stat}_pred"])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(df[f"{stat}"], df[f"{stat}_pred"])

        print(f"\n--- {stat.upper()} Regression Metrics ---")
        print(f"MSE:  {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE:  {mae:.3f}")

def get_team_wins(years, df):
    '''
    Parameters:
        years (list of ints): Example: [2020, 2021, 2022]
        df - a dataframe with the team and season predicted wins 
    
    Returns:
        team_wins (DataFrame): dataframe of both the teams actual and predicted wins for that season
    '''

    pbp = nfl.import_pbp_data(years)

    pbp = pbp[pbp['season_type'] == 'REG'].copy()

    pbp['game_date'] = pd.to_datetime(pbp['game_date'])

    # Group by game and collect final scores and metadata
    final_scores = (
        pbp.groupby("game_id")
        .agg({
            "home_team": "first",
            "away_team": "first",
            "home_score": "max",
            "away_score": "max",
            "game_date": "first"
        })
        .reset_index()
    )

    # Extract calendar year and NFL season (starting after Aug 30)
    final_scores['year'] = final_scores['game_date'].dt.year
    final_scores['season'] = final_scores['game_date'].apply(
        lambda d: d.year if d >= pd.Timestamp(f"{d.year}-08-30") else d.year - 1
    )

    # Determine winner
    final_scores['winner'] = final_scores.apply(
        lambda row: (
            row['home_team'] if row['home_score'] > row['away_score']
            else row['away_team'] if row['away_score'] > row['home_score']
            else None  # handle ties if needed
        ), axis=1
    )

    # Filter out ties (optional, depending on how you want to handle them)
    final_scores = final_scores[final_scores['winner'].notna()]

    # Count wins per team per season
    team_wins = (
        final_scores.groupby(['winner', 'season'])
        .size()
        .reset_index(name='wins')
        .rename(columns={'winner': 'team'})
    )

    team_wins = team_wins.sort_values(['season', 'wins'], ascending=[True, False])
    team_wins = team_wins.merge(df, on = ['team', 'season'])

    return team_wins

def get_breakout_features(df):
    '''
    Parameters:
        df - dataframe that is ready to get breakout features [is_new_coach, is_new_qb]
            - df is read in from csv file
    
    Returns:
        merged_df (DataFrame): df with features ready to train model for breakout/regress probs
    '''

    teams = df['team'].unique()

    # placeholder_rows = pd.DataFrame({
    #     'team' : teams,
    #     'season' : 2025,
    #     'wins' : np.nan,
    #     'is_breakout' : np.nan,
    #     'is_regress' : np.nan,
    # })

    # df = pd.concat([df, placeholder_rows], ignore_index=True).sort_values(['team', 'season'])

    nfl_team_abbr = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Oakland Raiders": "LV",            # historic name
    "Los Angeles Chargers": "LAC",
    "San Diego Chargers": "LAC",        # historic name
    "Los Angeles Rams": "LA",
    "St. Louis Rams": "LA",             # historic name
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
    "Washington Redskins": "WAS",       # historic name
    "Washington Football Team": "WAS"   # historic name
    }

    ssl._create_default_https_context = ssl._create_unverified_context

    years = list(range(2005, 2025))
    all_coaches_changes = []

    for year in years:
        try:
            url = f'https://en.wikipedia.org/wiki/{year}_NFL_season'
            tables = pd.read_html(url)

            # Try to find the table with 'Team' column
            coaches_table = None
            for table in tables:
                if isinstance(table.columns, pd.MultiIndex):
                    table.columns = [' '.join(col).strip() for col in table.columns.values]

                if 'Departing coach' in table.columns or 'Former Coach' in table.columns or 'Reason for leaving' in table.columns:
                    coaches_table = table
                    break

            if coaches_table is not None:
                coaches_table['season'] = year
                coaches_table['team'] = coaches_table['Team']
                all_coaches_changes.append(coaches_table[['team', 'season']])
            else:
                print(f"No suitable coach change table found for {year}")
        except Exception as e:
            print(f"Error for year {year}: {e}")

    # Combine all years into one DataFrame
    coach_changes_df = pd.concat(all_coaches_changes, ignore_index=True)

    coach_changes_df['team'] = coach_changes_df['team'].map(nfl_team_abbr)

    merged_df = df.merge(
    coach_changes_df[['team', 'season']],  # only need these two cols
    on=['team', 'season'],
    how='left',
    indicator=True  # adds a column to tell us if a match was found
    )

    merged_df['is_new_hc'] = (merged_df['_merge'] == 'both').astype(int)
    merged_df.drop(columns=['_merge'], inplace=True)

    # Add 2025 New Head Coaches Manually

    merged_df['is_new_hc'] = np.where(
    (merged_df['season'] == 2025) & (merged_df['team'].isin(['NE', 'CHI',
                                                    'LV', 'NYJ',
                                                    'NO', 'JAX',
                                                    'DAL'])),
    1,
    0
    )
    # Previous Wins

    merged_df['previous_wins'] = (
    merged_df
    .groupby('team')['wins']
    .shift(1)
    )

    merged_df['previous_500'] = np.where(merged_df['previous_wins'] >= 9, 1, 0)

    # New QB

    pbp = nfl.import_pbp_data(list(range(2005, 2025)))
    pbp = pbp[pbp['season_type'] == 'REG'].copy()

    week1 = pbp[(pbp['season'] >= 2000) & (pbp['week'] == 1)]

    week1 = week1.dropna(subset=['passer'])

    qb_starters = (
        week1.groupby(['season', 'posteam', 'passer'])
        .size()
        .reset_index(name='snaps')
    )

    starting_qbs = (
        qb_starters.sort_values('snaps', ascending=False)
        .drop_duplicates(subset=['season', 'posteam'])
        .rename(columns={'posteam': 'team', 'passer': 'qb'})
        .sort_values(['team', 'season'])
        .reset_index(drop=True)
    )

    starting_qbs['prev_qb'] = starting_qbs.groupby('team')['qb'].shift(1)
    starting_qbs['is_new_qb'] = (starting_qbs['qb'] != starting_qbs['prev_qb']).astype(int)
    starting_qbs = starting_qbs.dropna(subset = 'prev_qb')
    starting_qbs = starting_qbs[['season', 'team', 'is_new_qb']]

    qbs_rows = pd.DataFrame({
        'team' : teams,
        'season' : 2025,
        'is_new_qb' : 0
    })

    starting_qbs = pd.concat([starting_qbs, qbs_rows], ignore_index=True).sort_values(by = ['team', 'season'])

    starting_qbs['is_new_qb'] = np.where(
    (starting_qbs['season'] == 2025) & (starting_qbs['team'].isin(['ATL', 'CAR',
                                                                'CLE', 'LV',
                                                                'MIN', 'NO',
                                                                'NYG', 'NYJ',
                                                                'PIT', 'SEA',
                                                                'TEN'])),
    1,
    0
    )

    merged_df = merged_df.merge(
    starting_qbs,
    on=['team', 'season'],
    how='left'
    )

    # Pythag win percent

    final_scores = (
    pbp.groupby("game_id")
    .agg({
    "home_team": "first",
    "away_team": "first",
    "home_score": "max",
    "away_score": "max",
    "game_date": "first"
    })
    .reset_index()
    )

    home = final_scores[['game_id', 'game_date', 'home_team', 'home_score', 'away_score']].rename(
    columns={'home_team': 'team', 'home_score': 'points', 'away_score': 'points_allowed'})
    away = final_scores[['game_id', 'game_date', 'away_team', 'away_score', 'home_score']].rename(
    columns={'away_team': 'team', 'away_score': 'points', 'home_score': 'points_allowed'})

    team_games = pd.concat([home, away], ignore_index=True)
    team_games['game_date'] = pd.to_datetime(team_games['game_date'], errors='coerce')
    team_games['year'] = team_games['game_date'].dt.year
    team_games['season'] = team_games['game_date'].apply(
        lambda d: d.year if d >= pd.Timestamp(f"{d.year}-08-30") else d.year - 1
    )

    ps_pa = (
    team_games.groupby(['team', 'season'])
    .agg({
        'points' : 'sum',
        'points_allowed' : 'sum'
    })
    ).reset_index()

    ps_pa_rows = pd.DataFrame({
        'team' : teams,
        'season' : 2025,
        'points' : np.nan,
        'points_allowed' : np.nan
    })

    ps_pa = pd.concat([ps_pa, ps_pa_rows], ignore_index = True).sort_values(by = ['team', 'season'])

    ps_pa['exp_win_pct'] = (ps_pa['points'] ** 2.37) / ((ps_pa['points'] ** 2.37) + ps_pa['points_allowed'] ** 2.37)

    ps_pa['previous_pyth_win'] = (
    ps_pa
    .groupby('team')['exp_win_pct']
    .shift(1)
    )

    pyth_df = ps_pa[['team', 'season', 'previous_pyth_win']]

    merged_df = merged_df.merge(
    pyth_df,
    on=['team', 'season'],
    how='left'
    )

    return merged_df

def run_breakout_and_regress_cv_probs(df, holdout_season):
    '''
    Parameters:
        df - df with features ready to predict breakout/regress prob
        holdout_season - season that is being predicted on without training data seeing it
    
    Returns:
        df_with_probs (DataFrame): dataframe same as passed in with added breakout/regress probs
    '''

    features = ['is_new_hc', 'is_new_qb', 'previous_500', 'previous_pyth_win']

    # Split data into train and holdout
    df_train = df[df['season'] != holdout_season].copy().dropna()
    df_holdout = df[df['season'] == holdout_season].copy()

    # Prepare X and y
    X_train = df_train[features]

    # Models and CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, solver='liblinear')

    # --- Breakout model ---
    y_breakout = df_train['is_breakout']
    breakout_probs_cv = cross_val_predict(model, X_train, y_breakout, cv=cv, method='predict_proba')[:, 1]

    # Fit on full training data to predict holdout
    model.fit(X_train, y_breakout)
    breakout_probs_holdout = model.predict_proba(df_holdout[features])[:, 1]

    # --- Regress model ---
    y_regress = df_train['is_regress']
    regress_probs_cv = cross_val_predict(model, X_train, y_regress, cv=cv, method='predict_proba')[:, 1]

    # Fit on full training data to predict holdout
    model.fit(X_train, y_regress)
    regress_probs_holdout = model.predict_proba(df_holdout[features])[:, 1]

    # Attach CV probs to train set
    df_train = df_train.copy()
    df_train['breakout_prob'] = breakout_probs_cv
    df_train['regress_prob'] = regress_probs_cv

    # Attach holdout probs
    df_holdout = df_holdout.copy()
    df_holdout['breakout_prob'] = breakout_probs_holdout
    df_holdout['regress_prob'] = regress_probs_holdout

    # Combine back together if needed
    df_with_probs = pd.concat([df_train, df_holdout], ignore_index=True)

    return df_with_probs

def generate_predicted_passing_yards(pbp, holdout_season, n_splits=5):

    summary = (
        pbp.groupby(['posteam', 'season'], as_index=False)
        .agg({
            'pass_attempt': 'sum',
            'passing_yards': 'sum',
            'air_epa': 'sum',
            'complete_pass': 'sum'
        })
        .sort_values(['posteam', 'season'])
    )

    teams = summary['posteam'].unique()

    placeholder_rows = pd.DataFrame({
    'posteam': teams,
    'season': 2025,
    'pass_attempt': np.nan,
    'passing_yards': np.nan,
    'air_epa': np.nan,
    'complete_pass': np.nan
    })

    summary = pd.concat([summary, placeholder_rows], ignore_index=True).sort_values(['posteam', 'season'])

    summary['completion_perc'] = summary['complete_pass'] / summary['pass_attempt']

    for lag in [1, 2]:
        summary[f'passing_yards_lag{lag}'] = summary.groupby('posteam')['passing_yards'].shift(lag)
        summary[f'pass_attempt_lag{lag}'] = summary.groupby('posteam')['pass_attempt'].shift(lag)
        summary[f'completion_perc_lag{lag}'] = summary.groupby('posteam')['completion_perc'].shift(lag)

    summary = summary[~((summary['season'] <= 2024) & summary.isna().any(axis=1))]
    summary = summary[summary['posteam'] != '']
    summary = summary[summary['season'] >= 2002]

    features = [
        'passing_yards_lag1',
        'pass_attempt_lag1',
        'completion_perc_lag1',
        'passing_yards_lag2',
        'pass_attempt_lag2',
        'completion_perc_lag2'
    ]

    # Separate holdout season
    holdout_df = summary[summary['season'] == holdout_season]
    train_df = summary[summary['season'] < holdout_season].copy().dropna()

    # Scale features
    scaler = StandardScaler()
    X_train_all = train_df[features].values
    X_train_all_scaled = scaler.fit_transform(X_train_all)

    y_train_all = train_df['passing_yards'].values

    # 1. Cross-validation predictions on train_df
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))  # out-of-fold predictions

    for train_idx, val_idx in kf.split(X_train_all_scaled):
        X_tr, X_val = X_train_all_scaled[train_idx], X_train_all_scaled[val_idx]
        y_tr = y_train_all[train_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    train_df['pred_passing_yards'] = oof_preds
    train_df['team'] = train_df['posteam']

    # 2. Fit on all training data, predict holdout season
    model = LinearRegression()
    model.fit(X_train_all_scaled, y_train_all)

    X_holdout = scaler.transform(holdout_df[features].values)
    holdout_df = holdout_df.copy()
    holdout_df['pred_passing_yards'] = model.predict(X_holdout)
    holdout_df['team'] = holdout_df['posteam']

    # 3. Combine and return
    combined_df = pd.concat([train_df, holdout_df], ignore_index=True)

    return combined_df


    summary = (
        pbp.groupby(['posteam', 'season'], as_index=False)
        .agg({
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',
        })
    ).sort_values(['posteam', 'season'])

    summary['yards_per_attempt'] = summary['rushing_yards'] / summary['rush_attempt']

    # Create lag features
    for lag in [1, 2]:
        summary[f'rushing_yards_lag{lag}'] = summary.groupby('posteam')['rushing_yards'].shift(lag)
        summary[f'rush_attempt_lag{lag}'] = summary.groupby('posteam')['rush_attempt'].shift(lag)
        summary[f'yards_per_attempt_lag{lag}'] = summary.groupby('posteam')['yards_per_attempt'].shift(lag)

    summary = summary.dropna()

    features = [
        'rushing_yards_lag1',
        'rush_attempt_lag1',
        'yards_per_attempt_lag1',
        'rushing_yards_lag2',
        'rush_attempt_lag2',
        'yards_per_attempt_lag2'
    ]

    model_df = summary.dropna(subset=features + ['rushing_yards']).copy()

    # Split train and predict data
    train_df = model_df[model_df['season'] != season_predicting_for]
    predict_df = model_df[model_df['season'] == season_predicting_for]

    X_train = train_df[features].values
    y_train = train_df['rushing_yards'].values

    X_predict = predict_df[features].values

    # Optional: scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)

    predict_df = predict_df.copy()
    predict_df['team'] = predict_df['posteam']
    predict_df['pred_rushing_yards'] = lr.predict(X_predict_scaled)

    return predict_df

def generate_predicted_rushing_yards(pbp, holdout_season, n_splits=5):

    summary = (
        pbp.groupby(['posteam', 'season'], as_index=False)
        .agg({
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',
        })
        .sort_values(['posteam', 'season'])
    )
    
    teams = summary['posteam'].unique()

    placeholder_rows = pd.DataFrame({
    'posteam': teams,
    'season': 2025,
    'rush_attempt': np.nan,
    'rushing_yards': np.nan,
    })

    summary = pd.concat([summary, placeholder_rows], ignore_index=True).sort_values(['posteam', 'season'])

    summary['yards_per_attempt'] = summary['rushing_yards'] / summary['rush_attempt']

    for lag in [1, 2]:
        summary[f'rushing_yards_lag{lag}'] = summary.groupby('posteam')['rushing_yards'].shift(lag)
        summary[f'rush_attempt_lag{lag}'] = summary.groupby('posteam')['rush_attempt'].shift(lag)
        summary[f'yards_per_attempt_lag{lag}'] = summary.groupby('posteam')['yards_per_attempt'].shift(lag)

    summary = summary[~((summary['season'] <= 2024) & summary.isna().any(axis=1))]
    summary = summary[summary['posteam'] != '']
    summary = summary[summary['season'] >= 2002]

    features = [
        'rushing_yards_lag1',
        'rush_attempt_lag1',
        'yards_per_attempt_lag1',
        'rushing_yards_lag2',
        'rush_attempt_lag2',
        'yards_per_attempt_lag2'
    ]

    # Separate holdout season
    holdout_df = summary[summary['season'] == holdout_season]
    train_df = summary[summary['season'] < holdout_season].copy().dropna()

    # Scale features
    scaler = StandardScaler()
    X_train_all = train_df[features].values
    X_train_all_scaled = scaler.fit_transform(X_train_all)

    y_train_all = train_df['rushing_yards'].values

    # 1. Cross-validation predictions on train_df
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))  # out-of-fold predictions

    for train_idx, val_idx in kf.split(X_train_all_scaled):
        X_tr, X_val = X_train_all_scaled[train_idx], X_train_all_scaled[val_idx]
        y_tr = y_train_all[train_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    train_df['pred_rushing_yards'] = oof_preds
    train_df['team'] = train_df['posteam']

    # 2. Fit on all training data, predict holdout season
    model = LinearRegression()
    model.fit(X_train_all_scaled, y_train_all)

    X_holdout = scaler.transform(holdout_df[features].values)
    holdout_df = holdout_df.copy()
    holdout_df['pred_rushing_yards'] = model.predict(X_holdout)
    holdout_df['team'] = holdout_df['posteam']

    # 3. Combine and return
    combined_df = pd.concat([train_df, holdout_df], ignore_index=True)

    return combined_df



    summary = (
        pbp.groupby(['defteam', 'season'], as_index=False)
        .agg({
            'rush_attempt': 'sum',
            'rushing_yards': 'sum',
        })
    ).sort_values(['defteam', 'season'])

    teams = summary['defteam'].unique()

    placeholder_rows = pd.DataFrame({
    'defteam': teams,
    'season': 2025,
    'rush_attempt': np.nan,
    'rushing_yards': np.nan
    })

    summary = pd.concat([summary, placeholder_rows], ignore_index=True).sort_values(['defteam', 'season'])

    summary['yards_per_attempt'] = summary['rushing_yards'] / summary['rush_attempt']

    for lag in [1, 2]:
        summary[f'rushing_yards_lag{lag}'] = summary.groupby('defteam')['rushing_yards'].shift(lag)
        summary[f'rush_attempt_lag{lag}'] = summary.groupby('defteam')['rush_attempt'].shift(lag)
        summary[f'yards_per_attempt_lag{lag}'] = summary.groupby('defteam')['yards_per_attempt'].shift(lag)

    summary = summary[~((summary['season'] <= 2024) & summary.isna().any(axis=1))]
    summary = summary[summary['defteam'] != '']
    summary = summary[summary['season'] >= 2002]

    features = [
        'rushing_yards_lag1',
        'rush_attempt_lag1',
        'yards_per_attempt_lag1',
        'rushing_yards_lag2',
        'rush_attempt_lag2',
        'yards_per_attempt_lag2'
    ]

    # Separate holdout season
    holdout_df = summary[summary['season'] == holdout_season]
    train_df = summary[summary['season'] < holdout_season].copy().dropna()

    # Scale features
    scaler = StandardScaler()
    X_train_all = train_df[features].values
    X_train_all_scaled = scaler.fit_transform(X_train_all)

    y_train_all = train_df['rushing_yards'].values

    # 1. Cross-validation predictions on train_df
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))  # out-of-fold predictions

    for train_idx, val_idx in kf.split(X_train_all_scaled):
        X_tr, X_val = X_train_all_scaled[train_idx], X_train_all_scaled[val_idx]
        y_tr = y_train_all[train_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    train_df['pred_rushing_yards_against'] = oof_preds
    train_df['team'] = train_df['defteam']

    # 2. Fit on all training data, predict holdout season
    model = LinearRegression()
    model.fit(X_train_all_scaled, y_train_all)

    X_holdout = scaler.transform(holdout_df[features].values)
    holdout_df = holdout_df.copy()
    holdout_df['pred_rushing_yards_against'] = model.predict(X_holdout)
    holdout_df['team'] = holdout_df['defteam']

    # 3. Combine and return
    combined_df = pd.concat([train_df, holdout_df], ignore_index=True)

    return combined_df

def generate_predicted_qb_epa(pbp, holdout_season, n_splits=5):

    summary = (
        pbp.groupby(['posteam', 'season'], as_index=False)
        .agg({
            'qb_epa': 'mean',
            'air_epa': 'mean',
            'yac_epa': 'mean',
            'qb_scramble' : 'sum'
        })
    ).sort_values(['posteam', 'season'])

    teams = summary['posteam'].unique()

    placeholder_rows = pd.DataFrame({
    'posteam': teams,
    'season': 2025,
    'qb_epa': np.nan,
    'air_epa': np.nan,
    'yac_epa': np.nan,
    'qb_scramble' : np.nan
    })

    summary = pd.concat([summary, placeholder_rows], ignore_index=True).sort_values(['posteam', 'season'])

    for lag in [1, 2]:
        summary[f'qb_epa_lag{lag}'] = summary.groupby('posteam')['qb_epa'].shift(lag)
        summary[f'air_epa_lag{lag}'] = summary.groupby('posteam')['air_epa'].shift(lag)
        summary[f'yac_epa_lag{lag}'] = summary.groupby('posteam')['yac_epa'].shift(lag)
        summary[f'qb_scrambles_lag{lag}'] = summary.groupby('posteam')['qb_scramble'].shift(lag)

    summary = summary[~((summary['season'] <= 2024) & summary.isna().any(axis=1))]
    summary = summary[summary['posteam'] != '']
    summary = summary[summary['season'] >= 2002]

    features = [
        'qb_epa_lag1',
        'air_epa_lag1',
        'yac_epa_lag1',
        'qb_scrambles_lag1',
        'qb_epa_lag2',
        'yac_epa_lag2',
        'air_epa_lag2',
        'qb_scrambles_lag2'
    ]

    # Separate holdout season
    holdout_df = summary[summary['season'] == holdout_season]
    train_df = summary[summary['season'] < holdout_season].copy().dropna()

    # Scale features
    scaler = StandardScaler()
    X_train_all = train_df[features].values
    X_train_all_scaled = scaler.fit_transform(X_train_all)

    y_train_all = train_df['qb_epa'].values

    # 1. Cross-validation predictions on train_df
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_df))  # out-of-fold predictions

    for train_idx, val_idx in kf.split(X_train_all_scaled):
        X_tr, X_val = X_train_all_scaled[train_idx], X_train_all_scaled[val_idx]
        y_tr = y_train_all[train_idx]

        model = LinearRegression()
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    train_df['pred_qb_epa'] = oof_preds
    train_df['team'] = train_df['posteam']

    # 2. Fit on all training data, predict holdout season
    model = LinearRegression()
    model.fit(X_train_all_scaled, y_train_all)

    X_holdout = scaler.transform(holdout_df[features].values)
    holdout_df = holdout_df.copy()
    holdout_df['pred_qb_epa'] = model.predict(X_holdout)
    holdout_df['team'] = holdout_df['posteam']

    # 3. Combine and return
    combined_df = pd.concat([train_df, holdout_df], ignore_index=True)

    return combined_df

def generate_preds_stats(df, 
                         season_generated_for,
                         breakout_weight = 0.1,
                         regress_weight = 0.5,
                         degrees_of_freedom = 9):
    '''
    Parameters:
        df - dataframe that is passed in after running bayes_modeling_years
        season_generated_for - season you want to generate predicted stats for 
    
    Returns:
        apply (DataFrame): 
    '''
    pbp = nfl.import_pbp_data(list(range(2000,2025)))
    pbp = pbp[pbp['season_type'] == 'REG']

    os.chdir("/Users/aidanbeilke/Desktop/Football Projects")
    home = (
        df.groupby(['home_team', 'season'])
        .agg(home_score=('home_score', 'sum'),
             home_points_allowed=('home_points_allowed', 'sum'),
             home_epa=('epa_per_play_home', 'sum'))
        .reset_index()
        .rename(columns={'home_team': 'team', 'home_score': 'points_home'})
    )

    away = (
        df.groupby(['away_team', 'season'])
        .agg(away_score=('away_score', 'sum'),
             away_points_allowed=('away_points_allowed', 'sum'),
             away_epa=('epa_per_play_away', 'sum'))
        .reset_index()
        .rename(columns={'away_team': 'team', 'away_score': 'points_away'})
    )

    all_points = pd.merge(away, home, on=['team', 'season'])

    all_points['ppg'] = (all_points['points_away'] + all_points['points_home']) / 17
    all_points['epapp'] = (all_points['home_epa'] + all_points['away_epa']) / 17
    all_points['ppga'] = (all_points['away_points_allowed'] + all_points['home_points_allowed']) / 17

    all_points = all_points[['team', 'season', 'ppg', 'epapp', 'ppga']]
    all_points = all_points.sort_values(['team', 'season'])

    # If predicting for a season not in the dataset, add empty rows to allow rolling computation
    if season_generated_for not in all_points['season'].values:
        teams = all_points['team'].unique()
        new_rows = pd.DataFrame({
            'team': teams,
            'season': season_generated_for,
        })
        all_points = pd.concat([all_points, new_rows], ignore_index=True).sort_values(['team', 'season'])

    def weighted_rolling(series):
        weights = np.array([0.6, 0.2, 0.2])
        result = []
        for i in range(len(series)):
            if i < 3:
                result.append(np.nan)
            else:
                window = series.iloc[i-3:i]
                weighted_avg = np.dot(window[::-1], weights)  # reverse to make latest first
                result.append(weighted_avg)
        return pd.Series(result, index=series.index)

    all_points['ppg_rolling3'] = all_points.groupby('team')['ppg'].transform(weighted_rolling)
    all_points['epapp_rolling3'] = all_points.groupby('team')['epapp'].transform(weighted_rolling)
    all_points['ppga_rolling3'] = all_points.groupby('team')['ppga'].transform(weighted_rolling)

    all_points['ppg_prev'] = all_points.groupby('team')['ppg'].shift(1)
    all_points['epapp_prev'] = all_points.groupby('team')['epapp'].shift(1)
    all_points['ppga_prev'] = all_points.groupby('team')['ppga'].shift(1)
    breakout_df = pd.read_csv("season_simulation/breakout_regress.csv")

    teams = breakout_df['team'].unique()
    placeholder_rows = pd.DataFrame({
        'team' : teams,
        'season' : 2025,
        'wins' : np.nan,
        'is_breakout' : np.nan,
        'is_regress' : np.nan,
    })

    breakout_df = pd.concat([breakout_df, placeholder_rows], ignore_index=True).sort_values(['team', 'season'])
    all_points = all_points.merge(breakout_df, on = ['team', 'season'])

    modeled_df = get_breakout_features(all_points)
    modeled_df = run_breakout_and_regress_cv_probs(modeled_df, season_generated_for)

    def run_model(target, features, df_train, df_apply):
        df_train = df_train.dropna(subset=features)
        X_raw = df_train[features].values
        y = df_train[target].values

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        with pm.Model() as model:
            intercept = pm.Normal("Intercept", mu=0, sigma=10)
            coefs = pm.Normal("coefs", mu=0, sigma=15, shape=X.shape[1])
            sigma = pm.HalfCauchy("sigma", beta=25)

            mu = intercept + pm.math.dot(X, coefs)
            y_obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                idata = pm.sample(
                    3000,
                    tune=1000,
                    random_seed=42,
                    target_accept=0.95,
                    max_treedepth=30,
                    progressbar=False
                )

        # Scale df_apply features with the same scaler
        X_new = scaler.transform(df_apply[features].values)

        coefs_vals = idata.posterior['coefs'].stack(draws=("chain", "draw")).values
        intercepts = idata.posterior['Intercept'].stack(draws=("chain", "draw")).values
        sigma_vals = idata.posterior['sigma'].stack(draws=("chain", "draw")).values

        mu_preds = intercepts + np.dot(X_new, coefs_vals)
        pred_samples = mu_preds + t.rvs(degrees_of_freedom, loc=0, scale=sigma_vals, size=mu_preds.shape)

        coefs_summary = az.summary(idata, var_names=["coefs"])
        importances_df = pd.DataFrame({
            "feature": features,
            "mean": coefs_summary["mean"].values,
            "std": coefs_summary["sd"].values,
            "hdi_3%": coefs_summary["hdi_3%"].values,
            "hdi_97%": coefs_summary["hdi_97%"].values,
        }).sort_values(by="mean", key=abs, ascending=False)

        return {
            'mean': pred_samples.mean(axis=1),
            'std': pred_samples.std(axis=1),
            'idata': idata,
            'importances': importances_df
        }

    # Passing Yards Model 
    pred_passing_yards = generate_predicted_passing_yards(pbp, 
                                                        holdout_season=season_generated_for)
    
    pred_rushing_yards = generate_predicted_rushing_yards(pbp,
                                                          holdout_season = season_generated_for)
    
    pred_qb_epa = generate_predicted_qb_epa(pbp,
                                            holdout_season=season_generated_for)
    
    modeled_passing_yards = pred_passing_yards[['team','season','pred_passing_yards']]
    modeled_rushing_yards = pred_rushing_yards[['team', 'season', 'pred_rushing_yards']]
    modeled_qb_epa = pred_qb_epa[['team', 'season', 'pred_qb_epa']]

    modeled_df = modeled_df.merge(modeled_passing_yards, on = ['team', 'season'])
    modeled_df['adjusted_passing_yards'] = (modeled_df['pred_passing_yards'] * np.exp(breakout_weight * modeled_df['breakout_prob']) *np.exp(-regress_weight * modeled_df['regress_prob']))

    modeled_df = modeled_df.merge(modeled_rushing_yards, on = ['team', 'season'])
    modeled_df['adjusted_rushing_yards'] = (modeled_df['pred_rushing_yards'] * np.exp(breakout_weight * modeled_df['breakout_prob']) *np.exp(-regress_weight * modeled_df['regress_prob']))

    modeled_df = modeled_df.merge(modeled_qb_epa, on = ['team', 'season'])
    modeled_df['adjusted_qb_epa'] = (modeled_df['pred_qb_epa'] * np.exp(breakout_weight * modeled_df['breakout_prob']) *np.exp(-regress_weight * modeled_df['regress_prob']))

    model_data = modeled_df[modeled_df['season'] <= season_generated_for - 3]
    model_data = model_data[model_data['season'] != season_generated_for]
    apply = modeled_df[modeled_df['season'] == season_generated_for].copy()
    
    # --- PPG MODEL ---
    ppg_features = ['adjusted_qb_epa', 'adjusted_rushing_yards', 'adjusted_passing_yards']
    ppg_results = run_model('ppg', ppg_features, model_data.dropna(subset=ppg_features + ['ppg']), apply)
    apply['ppg_pred'] = ppg_results['mean']
    apply['ppg_pred_std'] = ppg_results['std']
    ppg_importances = ppg_results['importances']

    # --- EPA MODEL ---
    epa_features = ['epapp_rolling3','adjusted_passing_yards', 'pred_rushing_yards', 'breakout_prob', 'regress_prob']
    epa_results = run_model('epapp', epa_features, model_data.dropna(subset=epa_features + ['epapp']), apply)
    apply['epapp_pred'] = epa_results['mean']
    apply['epapp_pred_std'] = epa_results['std']
    epa_importances = epa_results['importances']

    # --- POINTS ALLOWED MODEL ---
    ppga_features = ['ppga_rolling3', 'breakout_prob', 'regress_prob']
    ppga_results = run_model('ppga', ppga_features, model_data.dropna(subset=ppga_features + ['ppga']), apply)
    apply['ppga_pred'] = ppga_results['mean']
    apply['ppga_pred_std'] = ppga_results['std']
    ppga_importances = ppga_results['importances']

    return apply, {
        'ppg_importances': ppg_importances,
        'epapp_importances': epa_importances,
        'ppga_importances': ppga_importances
    }

def run_functions(year_predicting, sims):
    pbp = bayes_modeling_process(list(range(2000, 2025)))
    apply_df, importances_dict = generate_preds_stats(pbp, year_predicting)
    win_totals = generate_season_win_totals(apply_df, year_predicting=year_predicting, simulations=sims)
    all = get_team_wins([year_predicting], win_totals)
    if year_predicting != 2025:
        rmse = np.sqrt(mean_squared_error(all['wins'], all['row_median']))

    espn_bet_totals_2024 = {
    'ARI': 6.5,  # Arizona Cardinals
    'ATL': 9.5,  # Atlanta Falcons
    'BAL': 11.5,  # Baltimore Ravens
    'BUF': 10.5,  # Buffalo Bills
    'CAR': 4.5,  # Carolina Panthers
    'CHI': 8.5,  # Chicago Bears
    'CIN': 10.5,  # Cincinnati Bengals
    'CLE': 8.5,  # Cleveland Browns
    'DAL': 10.5,  # Dallas Cowboys
    'DEN': 5.5,  # Denver Broncos
    'DET': 10.5,  # Detroit Lions
    'GB': 9.5,   # Green Bay Packers
    'HOU': 9.5,  # Houston Texans
    'IND': 8.5,  # Indianapolis Colts
    'JAX': 8.5,  # Jacksonville Jaguars
    'KC': 11.5,   # Kansas City Chiefs
    'LV': 6.5,   # Las Vegas Raiders
    'LAC': 8.5,  # Los Angeles Chargers
    'LA': 8.5,  # Los Angeles Rams
    'MIA': 9.5,  # Miami Dolphins
    'MIN': 6.5,  # Minnesota Vikings
    'NE': 4.5,   # New England Patriots
    'NO': 7.5,   # New Orleans Saints
    'NYG': 6.5,  # New York Giants
    'NYJ': 9.5,  # New York Jets
    'PHI': 10.5,  # Philadelphia Eagles
    'PIT': 8.5,  # Pittsburgh Steelers
    'SF': 11.5,   # San Francisco 49ers
    'SEA': 7.5,  # Seattle Seahawks
    'TB': 7.5,   # Tampa Bay Buccaneers
    'TEN': 6.5,  # Tennessee Titans
    'WAS': 6.5   # Washington Commanders
    }

    caeser_totals_2024 = {
        'ARI': 7,  # Arizona Cardinals
        'ATL': 9.5,  # Atlanta Falcons
        'BAL': 10.5,  # Baltimore Ravens
        'BUF': 10.5,  # Buffalo Bills
        'CAR': 5.5,  # Carolina Panthers
        'CHI': 9,  # Chicago Bears
        'CIN': 10.5,  # Cincinnati Bengals
        'CLE': 8.5,  # Cleveland Browns
        'DAL': 9.5,  # Dallas Cowboys
        'DEN': 6,  # Denver Broncos
        'DET': 10.5,  # Detroit Lions
        'GB': 9.5,   # Green Bay Packers
        'HOU': 9.5,  # Houston Texans
        'IND': 8.5,  # Indianapolis Colts
        'JAX': 8.5,  # Jacksonville Jaguars
        'KC': 11.5,   # Kansas City Chiefs
        'LV': 6.5,   # Las Vegas Raiders
        'LAC': 9,  # Los Angeles Chargers
        'LA': 8.5,  # Los Angeles Rams
        'MIA': 9.5,  # Miami Dolphins
        'MIN': 7,  # Minnesota Vikings
        'NE': 4.5,   # New England Patriots
        'NO': 7.5,   # New Orleans Saints
        'NYG': 6.5,  # New York Giants
        'NYJ': 10,  # New York Jets
        'PHI': 10.5,  # Philadelphia Eagles
        'PIT': 8.5,  # Pittsburgh Steelers
        'SF': 11,   # San Francisco 49ers
        'SEA': 8,  # Seattle Seahawks
        'TB': 8.5,   # Tampa Bay Buccaneers
        'TEN': 6.5,  # Tennessee Titans
        'WAS': 6.5   # Washington Commanders
        }

    draft_king_totals_2024 = {
        'ARI': 6.5,  # Arizona Cardinals
        'ATL': 10.5,  # Atlanta Falcons
        'BAL': 11.5,  # Baltimore Ravens
        'BUF': 10.5,  # Buffalo Bills
        'CAR': 4.5,  # Carolina Panthers
        'CHI': 8.5,  # Chicago Bears
        'CIN': 10.5,  # Cincinnati Bengals
        'CLE': 8.5,  # Cleveland Browns
        'DAL': 10.5,  # Dallas Cowboys
        'DEN': 5.5,  # Denver Broncos
        'DET': 10.5,  # Detroit Lions
        'GB': 10.5,   # Green Bay Packers
        'HOU': 9.5,  # Houston Texans
        'IND': 8.5,  # Indianapolis Colts
        'JAX': 8.5,  # Jacksonville Jaguars
        'KC': 11.5,   # Kansas City Chiefs
        'LV': 6.5,   # Las Vegas Raiders
        'LAC': 8.5,  # Los Angeles Chargers
        'LA': 8.5,  # Los Angeles Rams
        'MIA': 10.5,  # Miami Dolphins
        'MIN': 6.5,  # Minnesota Vikings
        'NE': 4.5,   # New England Patriots
        'NO': 7.5,   # New Orleans Saints
        'NYG': 6.5,  # New York Giants
        'NYJ': 9.5,  # New York Jets
        'PHI': 10.5,  # Philadelphia Eagles
        'PIT': 8.5,  # Pittsburgh Steelers
        'SF': 11.5,   # San Francisco 49ers
        'SEA': 7.5,  # Seattle Seahawks
        'TB': 8.5,   # Tampa Bay Buccaneers
        'TEN': 5.5,  # Tennessee Titans
        'WAS': 7.5   # Washington Commanders
        }
    
    if year_predicting == 2024:

        all['espn_total'] = all['team'].map(espn_bet_totals_2024)
        all['caeser_total'] = all['team'].map(caeser_totals_2024)
        all['draft_kings_total'] = all['team'].map(draft_king_totals_2024)

        rmse2 = np.sqrt(mean_squared_error(all['wins'], all['espn_total']))
        rmse3 = np.sqrt(mean_squared_error(all['wins'], all['caeser_total']))
        rmse4 = np.sqrt(mean_squared_error(all['wins'], all['draft_kings_total']))

    rmse = np.sqrt(mean_squared_error(all['wins'], all['row_median']))

    return {
        "pbp": pbp,
        "apply_df": apply_df,
        "win_totals": win_totals,
        "team_wins": all,
        "rmse_model": rmse,
        "rmse_espn": rmse2,
        "rmse_caesars": rmse3,
        "rmse_dk": rmse4
    }

def tune_best_df_param(apply_df, pbp, year_predicting):
    '''
    Parameters:
        aplly_df - df as a result of generate_pred stats
        pbp - df generated from nfl data import pbp for passing yards and rushing yards functions
        year predicting - value for season trying to predict
            - must be less than 2025
    
    Returns:
        rmses (list): list of rmse values corresponding with different df values
    '''
 

    all_rmses = []

    for df in range(2, 25):
        apply_df, importances_dict = generate_preds_stats(pbp, 
                                                        year_predicting,
                                                        degrees_of_freedom=df)

        rmses = []  # <-- Reset here for each degree of freedom

        for i in range(10):
            win_totals = generate_season_win_totals(apply_df, year_predicting=year_predicting, simulations=100)
            all = get_team_wins([year_predicting], win_totals)

            rmse = np.sqrt(mean_squared_error(all['wins'], all['row_median']))

            rmses.append({'run': i, 'rmse': rmse})
        
        rmses_df = pd.DataFrame(rmses)
        temp_rmse = rmses_df['rmse'].mean()

        print(f"Degrees of Freedom -{df}- complete")
        all_rmses.append({'dfs': df, 'rmse': temp_rmse})

    return all_rmses

def get_division(team_abbr):
    division_map = {
        # AFC East
        'BUF': 'AFC East', 'MIA': 'AFC East', 'NE': 'AFC East', 'NYJ': 'AFC East',
        
        # AFC North
        'BAL': 'AFC North', 'CIN': 'AFC North', 'CLE': 'AFC North', 'PIT': 'AFC North',
        
        # AFC South
        'HOU': 'AFC South', 'IND': 'AFC South', 'JAX': 'AFC South', 'TEN': 'AFC South',
        
        # AFC West
        'DEN': 'AFC West', 'KC': 'AFC West', 'LV': 'AFC West', 'LAC': 'AFC West',

        # NFC East
        'DAL': 'NFC East', 'NYG': 'NFC East', 'PHI': 'NFC East', 'WAS': 'NFC East',
        
        # NFC North
        'CHI': 'NFC North', 'DET': 'NFC North', 'GB': 'NFC North', 'MIN': 'NFC North',
        
        # NFC South
        'ATL': 'NFC South', 'CAR': 'NFC South', 'NO': 'NFC South', 'TB': 'NFC South',
        
        # NFC West
        'ARI': 'NFC West', 'LA': 'NFC West', 'SF': 'NFC West', 'SEA': 'NFC West',
    }
    
    return division_map.get(team_abbr.upper(), 'Unknown')
