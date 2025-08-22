import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
import importlib
import os
from season_simulation.functions import preprocess_data, bayes_modeling_process, generate_season_win_totals, get_team_wins, get_breakout_features, run_breakout_and_regress_cv_probs, generate_predicted_passing_yards, generate_predicted_qb_epa, generate_predicted_rushing_yards, generate_preds_stats, get_division, run_functions

if __name__ == "__main__":

    year_predicting = 2024
    sims = 1000

    pbp = bayes_modeling_process(list(range(2000, 2025)))
    print("getting pred stats")
    apply_df, importances_dict = generate_preds_stats(pbp, year_predicting)
    print("getting season totals")
    win_totals = generate_season_win_totals(apply_df, year_predicting=year_predicting, simulations=sims)
    print("getting actual wins")
    all = get_team_wins([year_predicting], win_totals)

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

    all['espn_total'] = all['team'].map(espn_bet_totals_2024)
    all['caeser_total'] = all['team'].map(caeser_totals_2024)
    all['draft_kings_total'] = all['team'].map(draft_king_totals_2024)

    rmse2 = np.sqrt(mean_squared_error(all['wins'], all['espn_total']))
    rmse3 = np.sqrt(mean_squared_error(all['wins'], all['caeser_total']))
    rmse4 = np.sqrt(mean_squared_error(all['wins'], all['draft_kings_total']))

    rmse = np.sqrt(mean_squared_error(all['wins'], all['row_median']))

    print(rmse)
    print(rmse2)
    print(rmse3)
    print(rmse4)