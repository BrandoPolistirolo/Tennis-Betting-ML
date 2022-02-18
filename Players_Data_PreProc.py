#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("all_players.csv")
data.head()


# In[2]:


players_data = pd.read_csv("atp_players.csv",header=0,names=['id','name','surname','hand','dob','country'])


# In[3]:


players_data.head()


# In[4]:


len(data)


# In[ ]:


underscore = "_"
for i in range(0,len(data)):
    if underscore in data['player_id'][i]  :
        data['player_id'][i] = "No"


# In[6]:


data = data.loc[data['player_id'] != "No"]


# In[7]:


cleaned_players_data = data


# In[8]:


players_data['player_id'] = " "
players_data['player_id'] = players_data['name'].replace(" ","-") + "-" + players_data['surname'].replace(' ','-')
players_data['player_id'] = players_data['player_id'].str.lower()


# In[9]:


players_data['player_id']


# In[10]:


players_data


# In[11]:


cleaned_players_data


# In[12]:


final_players = cleaned_players_data.merge(players_data,left_on='player_id',right_on='player_id',how='outer')


# In[13]:


final_players = final_players.loc[(final_players['dob'].isna()) == False]


# In[14]:


final_players


# In[15]:


#reset index
final_players.reset_index()


# In[19]:


final_players = final_players.reset_index()


# In[20]:


final_players


# In[21]:


#drop useless col
final_players=final_players.drop(['country_x','id','name','surname'],axis=1)


# In[24]:


final_players = final_players.drop(['index'],axis=1)


# In[25]:


final_players


# In[26]:


final_players.to_csv('players_data.csv')
final_players.to_pickle('players_data.pkl')


# In[1]:


#load players data
import pandas as pd
players_data = pd.read_csv(r'C:\Users\PC\Desktop\Kaggle_Tennis\players_data.csv')
players_data


# In[8]:


for i in range(0,len(players_data)):
    text = str(players_data.iloc[i]['player_id'])
    text = text.replace(" ","-")
    players_data.replace(to_replace = str(players_data.iloc[i]['player_id']),value = str(text),inplace = True)
    
    


# In[9]:


players_data


# In[11]:


players_data.to_csv('players_data.csv')


# In[ ]:




