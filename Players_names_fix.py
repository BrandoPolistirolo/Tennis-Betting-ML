#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load matches dataset
import pandas as pd
matches_dataset_final = pd.read_csv('final_kaggle_dataset.csv')


# In[2]:


#load players data
players_dataset = pd.read_csv('players_data.csv')


# In[3]:


#get unique player id from matches
players_from_matches = matches_dataset_final['player_id'].unique()


# In[4]:


len(players_from_matches)


# In[5]:


len(players_dataset)


# In[6]:


players_list = players_dataset['player_id'].tolist()


# In[7]:


players_list


# In[8]:


players_from_matches


# In[9]:


diff = list(set(players_list)-set(players_from_matches))


# In[10]:


diff


# In[16]:


len(diff)


# In[27]:


#remove players that are too old (born before 1960)
for i in range(0,len(players_dataset)-1):
    yob = players_dataset.iloc[i]['dob']
    yob = str(yob)
    yob = yob[:4]
    yob = int(yob)
    if yob <= 1960 : 
        players_dataset = players_dataset.drop(players_dataset.index[i])


# In[22]:


type(players_dataset.iloc[7]['dob'])


# In[28]:


players_dataset


# In[30]:


#save data
players_dataset.to_csv('players_data.csv')


# In[13]:


plyrs = pd.DataFrame(data=players_from_matches,columns=['player_id'])
plyrs


# In[29]:


#find players we need in players_data
needed_players = pd.DataFrame()
for i in range(len(plyrs)):
    row = players_dataset.loc[plyrs.iloc[i]['player_id'] == players_dataset['player_id'] ]
    needed_players = needed_players.append(row,ignore_index=True)
    


# In[19]:


players_dataset.loc[plyrs.iloc[0]['player_id'] == players_dataset['player_id'] ]


# In[30]:


needed_players


# In[35]:


#merge with remaining
merged_players_data = pd.merge(needed_players,plyrs,on='player_id',how='outer')


# In[36]:


merged_players_data


# In[37]:


merged_players_data.to_csv('players_data_1.csv')


# In[ ]:




