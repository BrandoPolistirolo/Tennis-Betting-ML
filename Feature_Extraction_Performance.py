#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
matches_df = pd.read_csv('final_df1.csv')


# In[49]:


matches_df


# In[41]:


matches_df.loc[matches_df['total_points_won_2'].isnull()]


# In[43]:


#performance
#compute L_ij = (s_i - s_j) + 1/6*(g_i - g_j) + 1/24*(x_i - x_j)

matches_df['L_12'] = 0
matches_df['L_21'] = 0

for i in range(0,len(matches_df)):
    if pd.isnull(matches_df.iloc[i]['total_points_won_1']):
        l = matches_df.iloc[i]['sets_won_1'] - matches_df.iloc[i]['sets_won_2'] + matches_df.iloc[i]['games_won_1']/6 -matches_df.iloc[i]['games_won_2']/6
    if pd.isnull(matches_df.iloc[i]['total_points_won_2']):
        l = matches_df.iloc[i]['sets_won_1'] - matches_df.iloc[i]['sets_won_2'] + matches_df.iloc[i]['games_won_1']/6 -matches_df.iloc[i]['games_won_2']/6
    else:
        l = matches_df.iloc[i]['sets_won_1'] - matches_df.iloc[i]['sets_won_2'] + matches_df.iloc[i]['games_won_1']/6 -matches_df.iloc[i]['games_won_2']/6 + matches_df.iloc[i]['total_points_won_1']/24 - matches_df.iloc[i]['total_points_won_2']/24
    
    matches_df.at[i,'L_12'] = l
    
    l = None
    print('iteration number : ',i,' of ',len(matches_df))


# In[44]:


#or in alternative ( much faster)
matches_df['l_ij'] = matches_df['sets_won_1'] - matches_df['sets_won_2'] + matches_df['games_won_1']/6 - matches_df['games_won_2']/6 + matches_df['total_points_won_1']/24 - matches_df['total_points_won_2']/24
#check if the two results coincide
matches_df['L_12'] == matches_df['l_ij']
#they do not we are going to use l_ij 


# In[50]:


#compute L_21
matches_df['l_ji'] = - matches_df['l_ij']


# In[51]:


matches_df.loc[matches_df['l_ij'].isnull()]


# In[52]:


for i in range(0,len(matches_df)):
    if pd.isnull(matches_df.iloc[i]['l_ij']):
        matches_df.at[i,'l_ij'] = matches_df.iloc[i]['L_12']
        matches_df.at[i,'l_ji'] = matches_df.iloc[i]['L_21']


# In[ ]:





# In[47]:


#compute delta_elo_ij = elo_i - elo_j
matches_df['delta_elo'] = matches_df['elo_1'] - matches_df['elo_2']


# In[57]:


#scale L_ij and delta_elo to (-1,1)

matches_df['elo_scaled'] = 0

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
matches_df['delta_elo_scaled'] = scaler.fit_transform(np.array(matches_df['delta_elo']).reshape(-1,1))
matches_df['l_ij_scaled'] = scaler.fit_transform(np.array(matches_df['l_ij']).reshape(-1,1))
matches_df['l_ji_scaled'] = scaler.fit_transform(np.array(matches_df['l_ji']).reshape(-1,1))


# In[58]:


matches_df


# In[69]:


#define and apply 'chessboard' classifier

def classify_performance(l,elo):
    performance= None
    if elo<=-0.8:
        if l>=0.8:
            performance =10
        if l>=0.2 and l<0.8:
            performance = 9
        if l>=0 and l<0.2:
            performance = 8
        if l>=-0.4 and l <0:
            performance = -1
        if -0.6<=l<-0.4:
            performance = -2
        if -0.8<=l<-0.6:
            performance = -3
        if -1<=l<-0.8:
            performance = -4
    if elo>-0.8 and elo<=-0.6:
        if l>= 0.4:
            performance = 9
        if l>=0:
            performance = 8
        if l<0 and l>=-0.2:
            performance = -1
        if l<-0.2 and l >= -0.4:
            performance = -2
        if l<-0.4 and l>=-0.6:
            performance = -3
        if l<-0.6 and l>=-0.8:
            performance = -4
        if l<-0.8 and l>=-1:
            performance = -5
    if elo>-0.6 and elo<=-0.4:
        if l>=0.6:
            performance = 9
        if l<0.6 and l>=0.2:
            performance = 8
        if l>=0 and l<0.2:
            performance =7
        if l<0 and l >=-0.2:
            performance = -2
        if l<-0.2 and l>=-0.4:
            performance = -3
        if l<-0.4 and l>=-0.6:
            performance = -4
        if l<-0.6 and l>=-0.8:
            performance = -5
        if l<-0.8 and l>=-1:
            performance = -6
    if elo>-0.4 and elo <=-0.2:
        if l>=0.8:
            performance = 9
        if l>=0.4 and l<0.8:
            performance = 8
        if l>=0.2 and l<0.4:
            performance = 7
        if l>=0 and l<0.2:
            performance = 6
        if l<0 and l>=-0.2:
            performance = -3
        if l<-0.2 and l>=-0.4:
            performance = -4
        if l<-0.4 and l>=-0.6:
            performance = -5
        if l<-0.6 and l>=-0.8:
            performance = -6
        if l<-0.8:
            performance = -7
    if elo>-0.2 and elo<=0:
        if l>=0.6:
            performance = 8
        if l>=0.4 and l < 0.6:
            performance = 7
        if l>=0.2 and l<0.4:
            performance = 6
        if l>=0 and l<0.2:
            performance = 5
        if l<0 and l>=-0.2:
            performance = -4
        if l<-0.2 and l>=-0.4:
            performance = -5
        if l<-0.4 and l>=-0.6:
            performance = -6
        if l<-0.6 and l>=-0.8:
            performance = -7
        if l<-0.8:
            performance = -8
    if elo>0 and elo<=0.2:
        if l>=0.8:
            performance = 8
        if l>=0.6 and l<0.8:
            performance = 7
        if l>=0.4 and l<0.6:
            performance = 6
        if l>=0.2 and l<0.4:
            performance = 5
        if l>=0 and l<0.2:
            performance = 4
        if l<0 and l>=-0.2:
            performance = - 5
        if l<-0.2 and l>=-0.4:
            performance = - 6
        if l<-0.4 and l >=-0.6:
            performance = - 7
        if l<-0.6:
            performance = -8
    if elo>0.2 and elo<=0.4:
        if l>=0.8:
            performance = 7
        if l>=0.6 and l<0.8:
            performance = 6
        if l>=0.4 and l<0.6:
            performance = 5
        if l>=0.2 and l<0.4:
            performance = 4
        if l>=0 and l<0.2:
            performance = 3
        if l<0 and l>=-0.2:
            performance = - 6
        if l<-0.2 and l>=-0.4:
            performance = - 7
        if l<-0.4 and l>=-0.8:
            performance = - 8
        if l<-0.8:
            performance = - 9
    if elo>0.4 and elo<=0.6:
        if l>=0.8:
            performance = 6
        if l>=0.6 and l<0.8:
            performance = 5
        if l>=0.4 and l<0.6:
            performance = 4
        if l>=0.2 and l<0.4:
            performance = 3
        if l>=0 and l<0.2:
            performance = 2
        if l<0 and l>=-0.2:
            performance = - 7
        if l<-0.2 and l>=-0.6:
            performance = - 8
        if l<-0.6:
            performance = - 9
    if elo>0.6 and elo<=0.8:
        if l>=0.8:
            performance = 5
        if l>=0.6 and l<0.8:
            performance = 4
        if l>=0.4 and l<0.6:
            performance = 3
        if l>=0.2 and l<0.4:
            performance = 2
        if l>=0 and l<0.2:
            performance = 1
        if l<0 and l>=-0.4:
            performance = - 8
        if l<-0.4:
            performance = - 9
    if elo>0.8 and elo<=1:
        if l>=0.8:
            performance = 4
        if l>=0.6 and l<0.8:
            performance = 3
        if l>=0.4 and l<0.6:
            performance = 2
        if l>=0.2 and l<0.4:
            performance = 1
        if l>=0 and l<0.2:
            performance = 1
        if l<0 and l>=-0.2:
            performance = - 8
        if l<-0.2 and l>=-0.8:
            performance = - 9
        if l<-0.8:
            performance = - 10
            
    return performance


# In[63]:


#check if classifier works
ls = np.arange(-1,1,0.1)
elos = np.arange(-1,1,0.1)
ls


# In[65]:


for l in ls:
    for elo in elos:
        perf = classify_performance(l,elo)
        print(perf,' ',l,' ',elo)


# In[66]:


matches_df.loc[matches_df['l_ij_scaled'].isnull()]


# In[71]:


#classifier works, we can apply it to our df
matches_df['performance_1'] = 0
matches_df['performance_2'] = 0
matches_df['delta_elo_scaled_2'] = -matches_df['delta_elo_scaled']
for i in range(0,len(matches_df)):
    if matches_df.iloc[i]['performance_1'] != 0 :
        continue
    matches_df.at[i,'performance_1'] = classify_performance(matches_df.iloc[i]['l_ij_scaled'],matches_df.iloc[i]['delta_elo_scaled'])
    matches_df.at[i,'performance_2'] = classify_performance(matches_df.iloc[i]['l_ji_scaled'],matches_df.iloc[i]['delta_elo_scaled_2'])
    print('iteration number : ',i,' of ',len(matches_df))


# In[72]:


matches_df


# In[73]:


#get player id's
players = []
for item in matches_df['player_id'].unique():
    players.append(item)
for item in matches_df['opponent_id'].unique():
    if item not in players:
        players.append(item)
players


# In[74]:


#we import the exponential moving average function from previous work(principal components) and modify it to suit our needs
#Function to get exp moving average
# N ---> elements to put into moving average(variable )
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


# In[76]:


matches_df['performance_1_ema'] = None
matches_df['performance_2_ema'] = None
i=0
for plyr in players:
    exp_moving_average(matches_df,plyr,'performance_1','performance_2','performance_1_ema','performance_2_ema')
    i=i+1
    print('iterations : ',i,' of ',len(players))


# In[77]:


#check the results on a excel file, we pick a random player (in this case carlos moya) and check
carlosmoya = matches_df.loc[(matches_df['player_id']=='carlos-moya')|(matches_df['opponent_id']=='carlos-moya')]
carlosmoya.to_csv('carlos-moya.csv')
federer = matches_df.loc[(matches_df['player_id']=='roger-federer') | (matches_df['opponent_id']=='roger-federer')]
federer.to_csv('federer.csv')


# In[78]:


#save
matches_df.to_csv('final_df1.csv')


# In[ ]:




