#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("all_matches.csv")
data.head()


# In[2]:


tournaments = pd.read_csv("all_tournaments.csv")
tournaments.head()
tourneys = tournaments[tournaments.masters >= 100]


# In[3]:


#getting only the tournaments we need
tourneys = tournaments[tournaments.masters >= 100]
tourneys = tourneys[tourneys.year>=2000]
tourneys


# In[4]:


#making a new dataset with tourney_id and year
tourneys_id = [tourneys.year,tourneys.tournament] 
tourneys_id = pd.concat(tourneys_id,axis=1)
tourneys_id


# In[5]:


#selecting only the matches from the tournaments previously selected
data = data[data.year >= 2000]
data = data[data.masters >= 100]


# In[6]:


#merging tourney_id and year 

#data
#type(data.tournament.loc[3])
#type(tourneys_id.tournament.loc[6])
t_years = tourneys_id.year
t_names = tourneys_id.tournament
t_info = t_names + t_years.astype(str)
t_info = t_info.reset_index(drop=True)
t_info


# In[7]:


#creating final dataframe for storing matches
final_df = pd.DataFrame(index = range(0,352512),columns = [#general info
                                   'start_date',
                                   'end_date',
                                   'location',
                                   'court_surface',
                                   'prize_money', #not relevant
                                   'currency', #not relevant
                                   'year',
                                   'player_id',
                                   'player_name', #not relevant, we only need id
                                   'opponent_id',
                                   'opponent_name', #not relevant, we only need id
                                   'tournament', 
                                   'round', 
                                   'num_sets',
                                   'doubles', #doubles need to be excluded
                                   'masters',
                                   'round_num',
                                   'duration',
                                   'total_points',
                                   #player 1 info, all relevant
                                   'sets_won_1',
                                   'games_won_1',
                                   'games_against_1',
                                   'tiebreaks_won_1',
                                   'tiebreaks_total',
                                   'serve_rating_1',
                                   'aces_1', 
                                   'double_faults_1', 
                                   'first_serve_made_1',
                                   'first_serve_attempted_1',
                                   'first_serve_points_made_1',
                                   'first_serve_points_attempted_1',
                                   'second_serve_points_made_1',
                                   'second_serve_points_attempted_1',
                                   'break_points_saved_1',
                                   'break_points_against_1', 
                                   'service_games_won_1', 
                                   'return_rating_1',
                                   'first_serve_return_points_made_1', 
                                   'first_serve_return_points_attempted_1',
                                   'second_serve_return_points_made_1',
                                   'second_serve_return_points_attempted_1',
                                   'break_points_made_1',
                                   'break_points_attempted_1',
                                   'return_games_played_1',
                                   'service_points_won_1',
                                   'service_points_attempted_1',
                                   'return_points_won_1',
                                   'return_points_attempted_1',
                                   'total_points_won_1',
                                   'player_1_victory',
                                   'retirement_1',
                                   'seed', 
                                   'won_first_set_1',
                                   'nation_1',
                                   #player 2 info, all relevant
                                   'sets_won_2',
                                   'games_won_2',
                                   'games_against_2',
                                   'tiebreaks_won_2',
                                   'serve_rating_2',
                                   'aces_2', 
                                   'double_faults_2', 
                                   'first_serve_made_2',
                                   'first_serve_attempted_2',
                                   'first_serve_points_made_2',
                                   'first_serve_points_attempted_2',
                                   'second_serve_points_made_2',
                                   'second_serve_points_attempted_2',
                                   'break_points_saved_2',
                                   'break_points_against_2', 
                                   'service_games_won_2', 
                                   'return_rating_2',
                                   'first_serve_return_points_made_2', 
                                   'first_serve_return_points_attempted_2',
                                   'second_serve_return_points_made_2',
                                   'second_serve_return_points_attempted_2',
                                   'break_points_made_2',
                                   'break_points_attempted_2',
                                   'return_games_played_2',
                                   'service_points_won_2',
                                   'service_points_attempted_2',
                                   'return_points_won_2',
                                   'return_points_attempted_2',
                                   'total_points_won_2',
                                   'player_2_victory',
                                   'retirement_2',
                                   'won_first_set_2',                                   
                                   'nation_2'])
    
final_df.fillna(0)


# In[8]:


#creating a dictionary of dataframes, every df is a torunament
tourney_dfs = {}

#remove doubles
data = data.loc[data["doubles"]=='f']
data = data.reset_index(drop=True)
i=0
#creating individual dataframes for every tournament and storing them into a dictionary tourneys_dfs
for  year,tournament in zip(t_years,t_names) :
    tourney_dfs[t_info[i]] = data.loc[(data["year"]==year) & (data["tournament"]==tournament)]
    i+=1
    


# In[9]:


data.columns


# In[11]:


#test
test = tourney_dfs['prague_challenger2015']
test


# In[12]:


#test
test.iloc[4].player_id


# In[13]:


final_df.columns


# In[32]:


#test
tourney_dfs['nottingham2015']


# In[33]:


data


# In[10]:


data = data.reset_index(drop=True)


# In[11]:


data


# In[12]:


#getting number of total rows in tourney_dfs
k=0
for key,value in tourney_dfs.items():
    j=len(value)
    k = k + j
    
print(k)


# In[13]:


#creating a final dataframe for every tournament and storing it in a dictionary
final_dfs = {}
i=0
for  key,value in tourney_dfs.items() :
    final_dfs[key] = pd.DataFrame(index = range(0,len(value)),columns = [#general info
                                   'start_date',
                                   'end_date',
                                   'location',
                                   'court_surface',
                                   'prize_money', #not relevant
                                   'currency', #not relevant
                                   'year',
                                   'player_id',
                                   'player_name', #not relevant, we only need id
                                   'opponent_id',
                                   'opponent_name', #not relevant, we only need id
                                   'tournament', 
                                   'round', 
                                   'num_sets',
                                   'doubles', #doubles need to be excluded
                                   'masters',
                                   'round_num',
                                   'duration',
                                   'total_points',
                                   #player 1 info, all relevant
                                   'sets_won_1',
                                   'games_won_1',
                                   'games_against_1',
                                   'tiebreaks_won_1',
                                   'tiebreaks_total',
                                   'serve_rating_1',
                                   'aces_1', 
                                   'double_faults_1', 
                                   'first_serve_made_1',
                                   'first_serve_attempted_1',
                                   'first_serve_points_made_1',
                                   'first_serve_points_attempted_1',
                                   'second_serve_points_made_1',
                                   'second_serve_points_attempted_1',
                                   'break_points_saved_1',
                                   'break_points_against_1', 
                                   'service_games_won_1', 
                                   'return_rating_1',
                                   'first_serve_return_points_made_1', 
                                   'first_serve_return_points_attempted_1',
                                   'second_serve_return_points_made_1',
                                   'second_serve_return_points_attempted_1',
                                   'break_points_made_1',
                                   'break_points_attempted_1',
                                   'return_games_played_1',
                                   'service_points_won_1',
                                   'service_points_attempted_1',
                                   'return_points_won_1',
                                   'return_points_attempted_1',
                                   'total_points_won_1',
                                   'player_1_victory',
                                   'retirement_1',
                                   'seed', 
                                   'won_first_set_1',
                                   'nation_1',
                                   #player 2 info, all relevant
                                   'sets_won_2',
                                   'games_won_2',
                                   'games_against_2',
                                   'tiebreaks_won_2',
                                   'serve_rating_2',
                                   'aces_2', 
                                   'double_faults_2', 
                                   'first_serve_made_2',
                                   'first_serve_attempted_2',
                                   'first_serve_points_made_2',
                                   'first_serve_points_attempted_2',
                                   'second_serve_points_made_2',
                                   'second_serve_points_attempted_2',
                                   'break_points_saved_2',
                                   'break_points_against_2', 
                                   'service_games_won_2', 
                                   'return_rating_2',
                                   'first_serve_return_points_made_2', 
                                   'first_serve_return_points_attempted_2',
                                   'second_serve_return_points_made_2',
                                   'second_serve_return_points_attempted_2',
                                   'break_points_made_2',
                                   'break_points_attempted_2',
                                   'return_games_played_2',
                                   'service_points_won_2',
                                   'service_points_attempted_2',
                                   'return_points_won_2',
                                   'return_points_attempted_2',
                                   'total_points_won_2',
                                   'player_2_victory',
                                   'retirement_2',
                                   'won_first_set_2',                                   
                                   'nation_2'])
    final_dfs[key].fillna(0)
    i+=1


# In[42]:


final_dfs


# In[14]:


#populating dataframes with matches data
k=0
for key,value in tourney_dfs.items():
    
    for i in range(len(value)):
        #add first match
        final_dfs[key].iloc[i].start_date = value.iloc[i].start_date
        final_dfs[key].iloc[i].end_date = value.iloc[i].end_date
        final_dfs[key].iloc[i].location = value.iloc[i].location
        final_dfs[key].iloc[i].court_surface = value.iloc[i].court_surface
        final_dfs[key].iloc[i].prize_money = value.iloc[i].prize_money
        final_dfs[key].iloc[i].currency = value.iloc[i].currency
        final_dfs[key].iloc[i].year = value.iloc[i].year
        final_dfs[key].iloc[i].player_id = value.iloc[i].player_id
        final_dfs[key].iloc[i].player_name = value.iloc[i].player_name
        final_dfs[key].iloc[i].opponent_id = value.iloc[i].opponent_id
        final_dfs[key].iloc[i].opponent_name = value.iloc[i].opponent_name
        final_dfs[key].iloc[i].tournament = value.iloc[i].tournament
        final_dfs[key].iloc[i].round = value.iloc[i].round
        final_dfs[key].iloc[i].num_sets = value.iloc[i].num_sets
        final_dfs[key].iloc[i].doubles = value.iloc[i].doubles
        final_dfs[key].iloc[i].masters = value.iloc[i].masters
        final_dfs[key].iloc[i].round_num = value.iloc[i].round_num
        final_dfs[key].iloc[i].duration = value.iloc[i].duration
        final_dfs[key].iloc[i].total_points = value.iloc[i].total_points
        final_dfs[key].iloc[i].sets_won_1 = value.iloc[i].sets_won
        final_dfs[key].iloc[i].games_won_1 = value.iloc[i].games_won
        final_dfs[key].iloc[i].games_against_1 = value.iloc[i].games_against
        final_dfs[key].iloc[i].tiebreaks_won_1 = value.iloc[i].tiebreaks_won
        final_dfs[key].iloc[i].tiebreaks_total = value.iloc[i].tiebreaks_total
        final_dfs[key].iloc[i].serve_rating_1 = value.iloc[i].serve_rating
        final_dfs[key].iloc[i].aces_1 = value.iloc[i].aces
        final_dfs[key].iloc[i].double_faults_1 = value.iloc[i].double_faults
        final_dfs[key].iloc[i].first_serve_made_1 = value.iloc[i].first_serve_made
        final_dfs[key].iloc[i].first_serve_attempted_1 = value.iloc[i].first_serve_attempted
        final_dfs[key].iloc[i].first_serve_points_made_1 = value.iloc[i].first_serve_points_made
        final_dfs[key].iloc[i].first_serve_points_attempted_1 = value.iloc[i].first_serve_points_attempted
        final_dfs[key].iloc[i].second_serve_points_made_1 = value.iloc[i].second_serve_points_made
        final_dfs[key].iloc[i].second_serve_points_attempted_1 = value.iloc[i].second_serve_points_attempted
        final_dfs[key].iloc[i].break_points_saved_1 = value.iloc[i].break_points_saved
        final_dfs[key].iloc[i].break_points_against_1 = value.iloc[i].break_points_against
        final_dfs[key].iloc[i].service_games_won_1 = value.iloc[i].service_games_won
        final_dfs[key].iloc[i].return_rating_1 = value.iloc[i].return_rating
        final_dfs[key].iloc[i].first_serve_return_points_made_1 = value.iloc[i].first_serve_return_points_made
        final_dfs[key].iloc[i].first_serve_return_points_attempted_1 = value.iloc[i].first_serve_return_points_attempted
        final_dfs[key].iloc[i].second_serve_return_points_made_1 = value.iloc[i].second_serve_return_points_made
        final_dfs[key].iloc[i].second_serve_return_points_attempted_1 = value.iloc[i].second_serve_return_points_attempted
        final_dfs[key].iloc[i].break_points_made_1 = value.iloc[i].break_points_made
        final_dfs[key].iloc[i].break_points_attempted_1 = value.iloc[i].break_points_attempted
        final_dfs[key].iloc[i].return_games_played_1 = value.iloc[i].return_games_played
        final_dfs[key].iloc[i].service_points_won_1 = value.iloc[i].service_points_won
        final_dfs[key].iloc[i].service_points_attempted_1 = value.iloc[i].service_points_attempted
        final_dfs[key].iloc[i].return_points_won_1 = value.iloc[i].return_points_won
        final_dfs[key].iloc[i].return_points_attempted_1 = value.iloc[i].return_points_attempted
        final_dfs[key].iloc[i].total_points_won_1 = value.iloc[i].total_points_won
        final_dfs[key].iloc[i].player_1_victory = value.iloc[i].player_victory
        final_dfs[key].iloc[i].retirement_1 = value.iloc[i].retirement
        final_dfs[key].iloc[i].seed = value.iloc[i].seed
        final_dfs[key].iloc[i].won_first_set_1 = value.iloc[i].won_first_set
        final_dfs[key].iloc[i].nation_1 = value.iloc[i].nation


        #search for matching match
        match_temp = value.loc[(value['opponent_id'] == value.iloc[i]['player_id']) & (value['player_id'] == value.iloc[i]['opponent_id'])]
        #merge
        final_dfs[key].iloc[i].sets_won_2                       = match_temp['sets_won'].values[0]
        final_dfs[key].iloc[i].games_won_2                      = match_temp['games_won'].values[0]
        final_dfs[key].iloc[i].games_against_2                  = match_temp['games_against'].values[0]
        final_dfs[key].iloc[i].tiebreaks_won_2                  = match_temp['tiebreaks_won'].values[0]
        final_dfs[key].iloc[i].serve_rating_2                   = match_temp['serve_rating'].values[0]
        final_dfs[key].iloc[i].aces_2                           = match_temp['aces'].values[0]
        final_dfs[key].iloc[i].double_faults_2                  = match_temp['double_faults'].values[0]
        final_dfs[key].iloc[i].first_serve_made_2               = match_temp['first_serve_made'].values[0]
        final_dfs[key].iloc[i].first_serve_attempted_2          = match_temp['first_serve_attempted'].values[0]
        final_dfs[key].iloc[i].first_serve_points_made_2        = match_temp['first_serve_points_made'].values[0]
        final_dfs[key].iloc[i].first_serve_points_attempted_2   = match_temp['first_serve_points_attempted'].values[0]
        final_dfs[key].iloc[i].second_serve_points_made_2       = match_temp['second_serve_points_made'].values[0]
        final_dfs[key].iloc[i].second_serve_points_attempted_2  = match_temp['second_serve_points_attempted'].values[0]
        final_dfs[key].iloc[i].break_points_saved_2             = match_temp['break_points_saved'].values[0]
        final_dfs[key].iloc[i].break_points_against_2           = match_temp['break_points_against'].values[0]
        final_dfs[key].iloc[i].service_games_won_2              = match_temp['service_games_won'].values[0]
        final_dfs[key].iloc[i].return_rating_2                  = match_temp['return_rating'].values[0]
        final_dfs[key].iloc[i].first_serve_return_points_made_2 = match_temp['first_serve_return_points_made'].values[0]
        final_dfs[key].iloc[i].first_serve_return_points_attempted_2 = match_temp['first_serve_return_points_attempted'].values[0]
        final_dfs[key].iloc[i].second_serve_return_points_made_2 = match_temp['second_serve_return_points_made'].values[0]
        final_dfs[key].iloc[i].second_serve_return_points_attempted_2 = match_temp['second_serve_return_points_attempted'].values[0]
        final_dfs[key].iloc[i].break_points_made_2              = match_temp['break_points_made'].values[0]
        final_dfs[key].iloc[i].break_points_attempted_2         = match_temp['break_points_attempted'].values[0]
        final_dfs[key].iloc[i].return_games_played_2            = match_temp['return_games_played'].values[0]
        final_dfs[key].iloc[i].service_points_won_2             = match_temp['service_points_won'].values[0]
        final_dfs[key].iloc[i].service_points_attempted_2       = match_temp['service_points_attempted'].values[0]
        final_dfs[key].iloc[i].return_points_won_2              = match_temp['return_points_won'].values[0]
        final_dfs[key].iloc[i].return_points_attempted_2        = match_temp['return_points_attempted'].values[0]
        final_dfs[key].iloc[i].total_points_won_2               = match_temp['total_points_won'].values[0]
        final_dfs[key].iloc[i].player_2_victory                 = match_temp['player_victory'].values[0]
        final_dfs[key].iloc[i].retirement_2                     = match_temp['retirement'].values[0]
        final_dfs[key].iloc[i].won_first_set_2                  = match_temp['won_first_set'].values[0]
        final_dfs[key].iloc[i].nation_2                         = match_temp['nation'].values[0]        
                
        #clear temporary match storage
        #match_temp = match_temp[0:0]
    k+=1
    print("checkpoint : " + str(k))

        #lastly drop duplicate matches 
        


# In[15]:


print(len(tourney_dfs.items()))
print(len(t_info))
#the values are different because some tournaments inside the Tournaments.csv are not present inside the All_Matches.csv
#some tournaments are missing


# In[50]:


#test save
final_dfs['kosice_challenger2012'].to_csv('kosice_challenger2012.csv')


# In[16]:


#getting list of dataframe names
frames=[]
for key,value in final_dfs.items():
    frames.append(value)


# In[53]:


#getting final dataframe
final_df = pd.concat(frames)


# In[54]:


#printing final dataframe
final_df


# In[56]:


#saving final 
final_df.to_csv('final_kaggle_dataset.csv')


# In[57]:


#saving final
final_df.to_pickle('final_kaggle_dataset.pkl')


# In[18]:


#saving each tournament dataframe to a specific .csv file
for key,value in final_dfs.items():
    csv_name = "Tournaments_Data" + "\ " + str(key) + ".csv"
    value.to_csv(csv_name)
    csv_name=''


# In[ ]:


#Next steps:

#1 see if matches mentioned two times are needed
#2 add betting odds
#remove unnecessary columns
#...

