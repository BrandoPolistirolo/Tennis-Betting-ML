#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir(r'C:\Users\PC\Desktop\Kaggle_Tennis\Tournaments_Data')
cwd = os.getcwd()
files = os.listdir(cwd)


# In[2]:


#remove double matches function

def remove_doubles(data):
    indexes = []
    for i in range(0,len(data)):
        temp = data.index[ (data['player_id'] == data.iloc[i]['opponent_id']) & (data['opponent_id'] == data.iloc[i]['player_id']).tolist()]
        indexes.append((i,int(temp[0])))     
    
    for (a,b) in indexes:
        if (b,a) in indexes:
            indexes.remove((b,a))
    index_new = []
    for item in indexes:
        index_new.append(item[1])
    data = data.drop(index=index_new)
    data = data.reset_index(drop=True)
        
    return data


# In[118]:


#test
import pandas as pd
test_set = pd.read_csv(' miami2010.csv')


# In[119]:


test_set = remove_doubles(test_set)


# In[120]:


test_set.loc[ (test_set['player_id'] == test_set.iloc[1]['opponent_id']) & (test_set['opponent_id'] == test_set.iloc[1]['player_id'])]


# In[121]:


test_set['player_id'] + '   ' +test_set['opponent_id']


# In[122]:


test_set.to_csv('test_miami2010.csv')


# In[4]:


import pandas as pd
for file in files:
    temp_dat = pd.read_csv(str(file))
    temp_dat = remove_doubles(temp_dat)
    temp_dat.to_csv(str(file))


# In[ ]:




