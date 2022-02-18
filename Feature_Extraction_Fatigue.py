#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
matches_df = pd.read_csv('final_df1.csv',infer_datetime_format=True,parse_dates=[1,2])


# In[18]:


#to model fatigue we use the duration of game and the number of sets played. time frame is also taken into account
#we consider only last games in a certain period of time


# In[22]:


matches_df


# In[11]:


list(matches_df.columns)


# In[60]:


#parse duration, we convert the format hours:minutes:seconds to hours(with decimal value representing minutes)
import numpy as np

matches_df['duration_hours'] = 0
#change nan values first
index = matches_df.loc[pd.isnull(matches_df['duration']) == True].index
for j in index:
    matches_df.at[j,'duration']=0
    
for i in range(0,len(matches_df)):
    if matches_df.iloc[i]['duration'] == 0:
        matches_df.at[i,'duration_hours'] = 0
        continue
    temp = matches_df.iloc[i]['duration']
    temp = temp.split(':')
    hours = float(temp[0])
    minutes = float(temp[1])
    matches_df.at[i,'duration_hours'] = hours + minutes/60
    temp = None
    hours = None
    minutes = None
    print('iteration number : ',i,' of ',len(matches_df))


# In[61]:


matches_df['duration_hours']


# In[ ]:




