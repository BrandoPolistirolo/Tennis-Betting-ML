#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
matches_df = pd.read_csv('final_df.csv',parse_dates = [5,6],infer_datetime_format=True)


# In[ ]:


#we need a formula to calculate the alpha (decay factor for the exponential moving average)
#we want it to be based on the amount of games played by a player
# let's see the distribution of total games played by players 


# In[2]:


#get player id's
players = []
for item in matches_df['player_id'].unique():
    players.append(item)
for item in matches_df['opponent_id'].unique():
    if item not in players:
        players.append(item)
players


# In[3]:


len(players)


# In[148]:


players_games = dict()
for player in players:
    temp = matches_df.loc[(matches_df['player_id']==player)|(matches_df['opponent_id']==player)]
    players_games[player] = len(temp)


# In[102]:


#average numbers of games played 
sum(players_games.values())/len(players_games)


# In[109]:


import seaborn as sns
sns.displot(data = players_games.values(),kind='kde',aspect=2.5)


# In[111]:


#we can see that very few players have a total number of games played above 200
#our alpha will be defined as alpha = 2/(N+1)


# In[154]:


matches_df


# In[91]:


#FUNCTION

#Function to get exp moving average
# N ---> elements to put into moving average(variable , last games in a 6-week period)
# alpha --> decay factor = 2/(N+1)
#function that computes and sets ema for specific column of data
import numpy as np
def exp_moving_average(data,player,column1,column2,dest_col1,dest_col2):
    #get relevant data
    temp = data.loc[(data['player_id']==player) | (data['opponent_id']==player)].copy()
    temp.reset_index(inplace=True)
    aa = dict()
    index = temp.index
    orig_index = temp['index']
    #print(index)
    #print(orig_index)
    for i in range(0,len(temp)):
        if temp.iloc[i]['player_id'] == player:
            if np.isnan(temp.iloc[i][column1])==True:
                continue
            aa[str(orig_index[i]) + 'a'] = temp.iloc[i][column1]
        if temp.iloc[i]['opponent_id'] == player:
            if np.isnan(temp.iloc[i][column2])==True:
                continue
            aa[str(orig_index[i]) + 'b'] = temp.iloc[i][column2]
            
    #print(counter1)
    #print(counter2)
    ema = dict()
    wcount = 0
    wsum = 0
    alpha = 2/(len(aa) + 1)
    factor = 1 - alpha
    bb = list(aa.values())
    cc = list(aa.keys())
    for j in range(1,len(aa)):
        wsum = bb[j-1] + factor*wsum
        #print(wsum)
        wcount = 1 + factor*wcount
        #print(wcount)
        ema[cc[j]] = wsum/wcount
    #print(ema)
    for key,value in ema.items():
        if key[len(key)-1] == 'a':
            data.at[int(key[:len(key)-1]),dest_col1] = value
        if key[len(key)-1] == 'b':
            data.at[int(key[:len(key)-1]),dest_col2] = value  
    del temp
    return 


# In[87]:


tryout = exp_moving_average(matches_df,'roger-federer','service_pca_1','service_pca_2','service_pca_1_ema','service_pca_2_ema')


# In[20]:


list(tryout.opponent_id)


# In[205]:


federer = matches_df.loc[(matches_df['player_id']=='roger-federer') | (matches_df['opponent_id']=='roger-federer')]


# In[206]:


federer.to_csv('federer.csv')


# In[43]:


list(federer.service_pca_2)


# In[91]:


list(matches_df.columns)


# In[207]:


#compute emas and assign them to our df


matches_df['service_pca_1_ema'] = None
matches_df['service_pca_2_ema'] = None
matches_df['return_pca_1_ema'] = None
matches_df['return_pca_2_ema'] = None
i=0
j=0
for plyr in players:
    exp_moving_average(matches_df,plyr,'service_pca_1','service_pca_2','service_pca_1_ema','service_pca_2_ema')
    i=i+1
    print('serve pca iterations : ',i)
for plyr in players:
    exp_moving_average(matches_df,plyr,'return_pca_1','return_pca_2','return_pca_1_ema','return_pca_2_ema')
    j=j+1
    print('return pca iterations : ',j)


# In[93]:


matches_df


# In[211]:


#FUNCTION


#fill na values of moving averages with the last known moving average for every player
def get_last_ema(player,col1,col2,data,row):
    #find first ema of given cols
    temp = data.iloc[:row]
    temp = temp.loc[(temp['player_id']==player)|(temp['opponent_id']==player)].copy()
    #print(temp)
    row1=0
    row2=0
    for i in range(len(temp)-1,0,-1):
        if temp.iloc[i]['player_id'] == player:
            if pd.isnull(temp.iloc[i][col1]):
                continue
            row1 = i
            break
        if temp.iloc[i]['opponent_id'] == player:
            if pd.isnull(temp.iloc[i][col2]):
                continue
            row2 = i
            break
    #print(row1,row2)
    if row1 != 0 :
        last_ema = temp.iloc[row1][col1]
        rowf = row1
    if row2 != 0 :
        last_ema = temp.iloc[row2][col2]
        rowf = row2
    del temp
    if (row1 == 0) & (row2 == 0):
        return 0,0
    return last_ema,rowf

def get_nans(player,data,col1,col2):
    temp = data.loc[(data['player_id']==player)|(data['opponent_id']==player)].copy()
    temp.reset_index(inplace=True)
    nanrows1 = []
    nanrows2 = []
    for i in range(0,len(temp)):
        if temp.iloc[i]['player_id'] == player:
            if pd.isnull(temp.iloc[i][col1]):
                nanrows1.append(temp.iloc[i]['index'])
        if temp.iloc[i]['opponent_id'] == player:
            if pd.isnull(temp.iloc[i][col2]):
                nanrows2.append(temp.iloc[i]['index'])
    #fill nans
    for item in nanrows1:
        last_ema,rowf = get_last_ema(player,col1,col2,data,item)
        if (last_ema!=0) & (rowf!=0):
            data.at[item,col1] = last_ema
    
    for item in nanrows2:
        last_ema,rowf = get_last_ema(player,col1,col2,data,item)
        if (last_ema!=0) & (rowf!=0):
            data.at[item,col2] = last_ema
    return


# In[201]:


get_nans('roger-federer',matches_df,'service_pca_1_ema','service_pca_2_ema')


# In[204]:


federer.loc[(federer['service_pca_1_ema'].isnull()) & (federer['player_id']=='roger-federer')]


# In[212]:


#fill nas for all players
i=0
for player in players:
    get_nans(player,matches_df,'service_pca_1_ema','service_pca_2_ema')
    get_nans(player,matches_df,'return_pca_1_ema','return_pca_2_ema')
    i=i+1
    print('iteration number :',i,' of ',len(players))


# In[213]:


matches_df


# In[214]:


matches_df.to_csv('final_df1.csv')


# In[220]:


#check the results on a excel file, we pick a random player (in this case carlos moya) and check
carlosmoya = matches_df.loc[(matches_df['player_id']=='carlos-moya')|(matches_df['opponent_id']=='carlos-moya')]


# In[221]:


carlosmoya.to_csv('carlos-moya.csv')


# In[ ]:


#moving averages calculations and na fill checks out


# In[227]:


matches_df.dropna(subset=['service_pca_1_ema','service_pca_2_ema','return_pca_1_ema','return_pca_2_ema'])


# In[ ]:


#we have a total of 132504 matches at our disposal for training and testing, which should be enough 

