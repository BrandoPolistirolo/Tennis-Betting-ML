#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
matches_df = pd.read_csv('final_df1.csv')


# In[2]:


matches_df


# In[3]:


#head to head balance
#transform player_win : t -> 1, f -> 0
matches_df['player_1_v'] = 0
matches_df['player_2_v'] = 0
for i in range(0,len(matches_df)):
    if matches_df.iloc[i]['player_1_victory'] == 't':
        matches_df.at[i,'player_1_v'] = 1
    if matches_df.iloc[i]['player_1_victory'] == 'f':
        matches_df.at[i,'player_1_v'] = 0
    if matches_df.iloc[i]['player_2_victory'] == 't':
        matches_df.at[i,'player_2_v'] = 1
    if matches_df.iloc[i]['player_2_victory'] == 'f':
        matches_df.at[i,'player_2_v'] = 0


# In[ ]:


#let's check the frequency of player_1_v and player_2_v (we do this because one of these will be our target variable)
#we want the target to be more or less balanced, in order to avoid bias in our model 


# In[5]:


import seaborn as sns
sns.displot(matches_df['player_1_v'],aspect=3)


# In[6]:


sns.displot(matches_df['player_2_v'],aspect=3)


# In[ ]:


#data is balanced so we have one less worry


# In[4]:


#get player id's
players = []
for item in matches_df['player_id'].unique():
    players.append(item)
for item in matches_df['opponent_id'].unique():
    if item not in players:
        players.append(item)
players


# In[19]:



def head_to_head(player,opponent,data):
    temp = data.loc[((data['player_id']==player)&(data['opponent_id']==opponent))
        |((data['player_id']==opponent)&(data['opponent_id']==player))]
    if len(temp) == 0:
        return
    balance = []
    reference_player = temp.iloc[0]['player_id']
    balance.append(0)
    for i in range(0,len(temp)):
        if data.iloc[i]['player_id'] == reference_player:
if data.iloc[i]['player_1_v'] == 1:
    balance.append(1)
if data.iloc[i]['player_1_v'] == 0:
    balance.append(-1)
        if data.iloc[i]['opponent_id'] == reference_player:
if data.iloc[i]['player_1_v'] == 0:
    balance.append(1)
if data.iloc[i]['player_1_v'] == 1:
    balance.append(-1)
    #now set hth for reference player
    #first game
    data.at[temp.index[0],'hth_1'] = balance[0]
    data.at[temp.index[0],'hth_2'] = balance[0]
    #games after first
    for i in range(1,len(temp)):
        if temp.iloc[i]['player_id'] == reference_player:
data.at[temp.index[i],'hth_1'] = sum(balance[:i+1])
data.at[temp.index[i],'hth_2'] = -sum(balance[:i+1])
        if temp.iloc[i]['opponent_id'] == reference_player:
data.at[temp.index[i],'hth_2'] = sum(balance[:i+1])
data.at[temp.index[i],'hth_1'] = -sum(balance[:i+1])
    return      
     


# In[21]:


#create list of players
hth_list = []
for player in players:
    for opponent in players:
        if opponent == player:
            continue
        string = player + ':' + opponent
        hth_list.append(string)


# In[22]:


len(hth_list)


# In[26]:


hth_series = pd.Series(hth_list)
hth_series = hth_series.unique()


# In[29]:


matches_df['hth_1'] = 0
matches_df['hth_2'] = 0
i = 0
#this way it would take approximately 32 days to complete, find a different solution
for player in players:
    for opponent in players:
        if player == opponent:
            continue
        head_to_head(player,opponent,matches_df)
        i = i + 1
        print('iteration number : ',i,' of 94410372')


# In[6]:


#differnt solution
#for every match look in the past for head to head encounters
#compute balance and assign

def get_balance(row,player,opponent,data):
    temp = data.iloc[:row-1]
    temp = temp.loc[((temp['player_id']==player)&(temp['opponent_id']==opponent))
                    |((temp['player_id']==opponent)&(temp['opponent_id']==player))]
    if len(temp)==0:
        return 0
    balance = 0
    for i in range(0,len(temp)):
        if temp.iloc[i]['player_id'] == player:
            if temp.iloc[i]['player_1_v'] == 1:
                balance = balance + 1
            if temp.iloc[i]['player_1_v'] == 0:
                balance = balance - 1
        if temp.iloc[i]['player_id'] == opponent:
            if temp.iloc[i]['player_1_v'] == 1:
                balance = balance - 1
            if temp.iloc[i]['player_1_v'] == 0:
                balance = balance + 1
    return balance
    


# In[11]:


#function check
matches_df.loc[((matches_df['player_id']=='roger-federer')&(matches_df['opponent_id']=='carlos-moya'))
                    |((matches_df['player_id']=='carlos-moya')&(matches_df['opponent_id']=='roger-federer'))]


# In[8]:


get_balance(40000,'roger-federer','carlos-moya',matches_df)


# In[ ]:


#the function works


# In[9]:


#compute and assign head to head balances for every row of data
matches_df['hth_1'] = 0
matches_df['hth_2'] = 0
for i in range(0,len(matches_df)):
    temp_player = matches_df.iloc[i]['player_id']
    temp_opponent = matches_df.iloc[i]['opponent_id']
    bal = get_balance(i,temp_player,temp_opponent,matches_df)
    matches_df.at[i,'hth_1'] = bal
    matches_df.at[i,'hth_2'] = -bal
    print('iteration number : ',i,' of ',len(matches_df))


# In[10]:


matches_df


# In[12]:


#save
matches_df.to_csv('final_df1.csv')


# In[ ]:




