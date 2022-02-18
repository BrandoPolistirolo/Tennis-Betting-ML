#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
matches_df = pd.read_csv('final_df1.csv')
players = pd.read_csv('players_data.csv',infer_datetime_format=True,parse_dates=['dob'])


# In[35]:


matches_df


# In[12]:


players


# In[52]:


#get player id's from matches dataframe
players_df = []
for item in matches_df['player_id'].unique():
    players_df.append(item)
for item in matches_df['opponent_id'].unique():
    if item not in players_df:
        players_df.append(item)
players_df


# In[53]:


players_concat = pd.DataFrame()
indexes = []
for player in players_df:
    if len(players.loc[players['player_id']==player])!=0:
        k = players.loc[players['player_id']==player].index
        indexes.append(k[0])
    


# In[54]:


indexes


# In[55]:


len(indexes)


# In[56]:


len(players_df)


# In[64]:


players_concat = players.iloc[indexes].copy()


# In[75]:


players_concat


# In[74]:


players_concat.drop(['level_0','index','Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1'],axis=1,inplace=True)


# In[69]:


players_concat['dob1'] = None
players_concat.reset_index(inplace=True)


# In[76]:


for i in range(0,len(players_concat)):
    temp = players_concat.iloc[i]['dob']
    players_concat.at[i,'dob1'] = temp[:4] +'-'+temp[4:6]


# In[89]:


players_concat


# In[79]:


#convert to datetime
players_concat['date of birth'] =  pd.to_datetime(players_concat['dob1'], format='%Y-%m')


# In[81]:


players_concat['date of birth']


# In[82]:


players_concat.to_csv('players-age.csv')


# In[83]:


#now get age difference and add them to matches dataframe
matches_df = pd.read_csv('final_df1.csv',infer_datetime_format=True,parse_dates=['start_date','end_date'])


# In[84]:


matches_df['start_date']


# In[104]:


k = matches_df.iloc[9]['start_date'] - players_concat.iloc[5]['date of birth']


# In[105]:


k


# In[106]:


k.days


# In[108]:


matches_df['player age'] = None
matches_df['opponent age'] = None

for i in range(0,len(matches_df)):
    player = matches_df.iloc[i]['player_id']
    opponent = matches_df.iloc[i]['opponent_id']
    temp1 = players_concat.loc[players_concat['player_id']==player].index
    temp2 = players_concat.loc[players_concat['player_id']==opponent].index
    if len(temp1) != 0:
        l = matches_df.iloc[i]['start_date'] - players_concat.iloc[temp1[0]]['date of birth']
        matches_df.at[i,'player age'] = l.days
    if len(temp2) != 0:
        p = matches_df.iloc[i]['start_date'] - players_concat.iloc[temp2[0]]['date of birth']
        matches_df.at[i,'opponent age'] = p.days
    player = None
    opponent = None
    temp1 = None
    temp2 = None
    l = None
    p = None
    print('iteration : ',i,' of ',len(matches_df))


# In[109]:


matches_df


# In[110]:


#save
matches_df.to_csv('final_df1.csv')


# In[ ]:




