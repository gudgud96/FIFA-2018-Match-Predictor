'''
Pre-process raw data from results.csv which contains all international match results.
Introduce attributes such as winning counts, goal difference, etc. for prediction task later.
Author: gudgud96
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

team_names = ['Australia', 'Iran', 'Japan', 'Korea Republic', 
            'Saudi Arabia', 'Egypt', 'Morocco', 'Nigeria', 
            'Senegal', 'Tunisia', 'Costa Rica', 'Mexico', 
            'Panama', 'Argentina', 'Brazil', 'Colombia', 
            'Peru', 'Uruguay', 'Belgium', 'Croatia', 
            'Denmark', 'England', 'France', 'Germany', 
            'Iceland', 'Poland', 'Portugal', 'Russia', 
            'Serbia', 'Spain', 'Sweden', 'Switzerland']

history = pd.read_csv('results.csv')
rankings = pd.read_csv('rankings.csv')

# sanitize to find results only related to the 24 teams taking part
history = history[(history['home_team'].isin(team_names)) | (history['away_team'].isin(team_names))]
history = history.reset_index()

# get all team names in sanitized dataset - this is to prevent KeyError problem
all_teams = team_names.copy()
for i in range(len(history)):
    if history['home_team'][i] not in all_teams:
        all_teams.append(history['home_team'][i]) 
    if history['away_team'][i] not in all_teams:
        all_teams.append(history['away_team'][i])

# add winning classifier - home loses 0, wins 1, draws 2 
home_wins = []
for i in range(len(history['home_score'])):
    if history['home_score'][i] < history['away_score'][i]:
        home_wins.append(0)
    elif  history['home_score'][i] > history['away_score'][i]:
        home_wins.append(1)
    else:
        home_wins.append(2)

history['home_wins'] = home_wins 

# add rankings - rankings should be up to the year the match is competed
home_team_ranking = []
away_team_ranking = []
for i in tqdm(range(len(history))):     # this takes around 5mins, so I added tqdm wrapper to keep track
    comparing_year = 1993 if history['year'][i] < 1993 else history['year'][i]
    home_rank_df = rankings[(rankings['country_full'] == history['home_team'][i]) & (rankings['year'] == comparing_year)]
    away_rank_df = rankings[(rankings['country_full'] == history['away_team'][i]) & (rankings['year'] == comparing_year)]
    home_rank = home_rank_df.reset_index()['rank'][0] if len(home_rank_df) != 0 else np.NaN
    home_team_ranking.append(home_rank)
    away_rank = away_rank_df.reset_index()['rank'][0] if len(away_rank_df) != 0 else np.NaN
    away_team_ranking.append(away_rank)

history['home_team_ranking'] = home_team_ranking
history['away_team_ranking'] = away_team_ranking

# create some caches for attributes
winning_count_cache = dict.fromkeys(all_teams, 0)
drawing_count_cache = dict.fromkeys(all_teams, 0)
num_of_matches_cache = dict.fromkeys(all_teams, 0)
goal_diff_cache = dict.fromkeys(all_teams, 0)

home_winning_count = []
home_drawing_count = []
away_winning_count = []
away_drawing_count = []
home_win_away = []
away_win_home = []
home_draw_away = []
home_mean_goal_diff = []
away_mean_goal_diff = []

# this takes around 13mins, thank god we have tqdm <3
for i in tqdm(range(len(history))):
    home_team = history['home_team'][i]
    away_team = history['away_team'][i]
    
    home_winning_count.append(0 if num_of_matches_cache[home_team] == 0  else winning_count_cache[home_team] / num_of_matches_cache[home_team])
    home_drawing_count.append(0 if num_of_matches_cache[home_team] == 0  else drawing_count_cache[home_team] / num_of_matches_cache[home_team])
    home_mean_goal_diff.append(0 if num_of_matches_cache[home_team] == 0 else goal_diff_cache[home_team] / num_of_matches_cache[home_team])

    away_winning_count.append(0 if num_of_matches_cache[away_team] == 0 else winning_count_cache[away_team] / num_of_matches_cache[away_team])
    away_drawing_count.append(0 if num_of_matches_cache[away_team] == 0 else drawing_count_cache[away_team] / num_of_matches_cache[away_team])
    away_mean_goal_diff.append(0 if num_of_matches_cache[away_team] == 0 else goal_diff_cache[away_team] / num_of_matches_cache[away_team])

    home_win_away_history = history[(history.index < i) & (
                            ((history['home_team'] == home_team) & (history['away_team'] == away_team) & (history['home_wins'] == 1)) |
                            ((history['home_team'] == away_team) & (history['away_team'] == home_team) & (history['home_wins'] == 0)))]
    away_win_home_history = history[(history.index < i) & (
                            ((history['home_team'] == home_team) & (history['away_team'] == away_team) & (history['home_wins'] == 0)) |
                            ((history['home_team'] == away_team) & (history['away_team'] == home_team) & (history['home_wins'] == 1)))]
    home_draw_away_history = history[(history.index < i) & (
                            ((history['home_team'] == home_team) & (history['away_team'] == away_team) & (history['home_wins'] == 2)) |
                            ((history['home_team'] == away_team) & (history['away_team'] == home_team) & (history['home_wins'] == 2)))]

    home_win_away.append(len(home_win_away_history))
    away_win_home.append(len(away_win_home_history))
    home_draw_away.append(len(home_draw_away_history))

    # update each cache
    if history['tournament'][i] != "FIFA":
        num_of_matches_cache[home_team] += 1
        num_of_matches_cache[away_team] += 1

        if history['home_wins'][i] == 1:
            winning_count_cache[home_team] += 1
        elif history['home_wins'][i] == 0:
            winning_count_cache[away_team] += 1
        else:
            drawing_count_cache[home_team] += 1
            drawing_count_cache[away_team] += 1
        
        goal_diff = history['home_score'][i] - history['away_score'][i]  
        goal_diff_cache[home_team] += goal_diff
        goal_diff_cache[away_team] -= goal_diff

history['home_winning_count'] = home_winning_count
history['home_drawing_count'] = home_drawing_count
history['away_winning_count'] = away_winning_count
history['away_drawing_count'] = away_drawing_count
history['home_win_away'] = home_win_away
history['away_win_home'] = away_win_home
history['home_draw_away'] = home_draw_away
history['home_mean_goal_diff'] = home_mean_goal_diff
history['away_mean_goal_diff'] = away_mean_goal_diff

# output processed results - attributes used are in this column
column_names = ['home_team', 'away_team', 'home_score', 'away_score',
                'home_team_ranking', 'away_team_ranking'
                'home_winning_count', 'home_drawing_count', 'away_winning_count', 'away_drawing_count',
                'home_win_away', 'away_win_home', 'home_draw_away', 
                'home_mean_goal_diff', 'away_mean_goal_diff', 'neutral', 'home_wins']

history.to_csv('results_processed.csv')