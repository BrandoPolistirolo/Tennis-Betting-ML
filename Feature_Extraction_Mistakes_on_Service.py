#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
matches_df = pd.read_csv('final_df1.csv')


# In[2]:


list(matches_df.columns)


# In[3]:


matches_df['first_serve_errors_1'] = None
matches_df['second_serve_errors_1'] = None
matches_df['first_serve_errors_2'] = None
matches_df['second_serve_errors_2'] = None


# In[9]:


for i in range(0,len(matches_df)):
    matches_df.at[i,'first_serve_errors_1'] = (1 - matches_df.iloc[i]['first_serve_made_1']/matches_df.iloc[i]['first_serve_attempted_1'])
    matches_df.at[i,'second_serve_errors_1'] = matches_df.iloc[i]['double_faults_1']/(matches_df.iloc[i]['first_serve_attempted_1']-matches_df.iloc[i]['first_serve_made_1'])
    matches_df.at[i,'first_serve_errors_2'] = (1 - matches_df.iloc[i]['first_serve_made_2']/matches_df.iloc[i]['first_serve_attempted_2'])
    matches_df.at[i,'second_serve_errors_2'] = matches_df.iloc[i]['double_faults_2']/(matches_df.iloc[i]['first_serve_attempted_2']-matches_df.iloc[i]['first_serve_made_2'])
    print('iteration : ',i,' of ',len(matches_df))


# In[14]:


#checkpoint
matches_df.to_csv('final_df2.csv')


# In[1]:


import pandas as pd
matches_df = pd.read_csv('final_df2.csv')


# In[4]:


matches_df


# In[2]:


error_pca_1 = matches_df[['first_serve_errors_1','second_serve_errors_1']]
error_pca_1


# In[3]:


error_pca_2 = matches_df[['first_serve_errors_2','second_serve_errors_2']]
error_pca_2


# In[5]:


error_pca_1.dropna()


# In[7]:


#NA's save index 
index_1 = error_pca_1.dropna().index
index_2 = error_pca_2.dropna().index
index_1


# In[26]:


#perform pca on serve errors


#SOLVE NA's problem 


from sklearn.preprocessing import StandardScaler
x = error_pca_1.dropna()
x = x.values
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca_errors_1 = PCA(n_components=1)
princomp_errors1 = pca_errors_1.fit_transform(x)
pca_errors_1.explained_variance_ratio_


# In[27]:


from sklearn.preprocessing import StandardScaler
x = error_pca_2.dropna()
x = x.values
x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca_errors_2 = PCA(n_components=1)
princomp_errors2 = pca_errors_2.fit_transform(x)
pca_errors_2.explained_variance_ratio_


# In[28]:


#add pca's to matches_df in correct index position
matches_df['error_pca_1'] = None
matches_df['error_pca_2'] = None
k=0
for i in index_1:
    matches_df.at[i,'error_pca_1'] = princomp_errors1[k][0]
    k = k + 1

g = 0
for j in index_2:
    matches_df.at[j,'error_pca_2'] = princomp_errors2[g][0]
    g = g + 1


# In[29]:


matches_df


# In[42]:


#exponential moving average (same function used for other features)
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
            if (temp.iloc[i][column1] is None)==True:
                continue
            aa[str(orig_index[i]) + 'a'] = temp.iloc[i][column1]
        if temp.iloc[i]['opponent_id'] == player:
            if (temp.iloc[i][column2] is None)==True:
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


# In[20]:


#get player id's
players = []
for item in matches_df['player_id'].unique():
    players.append(item)
for item in matches_df['opponent_id'].unique():
    if item not in players:
        players.append(item)
players


# In[43]:


matches_df['errors_pca_1_ema'] = None
matches_df['errors_pca_2_ema'] = None

i=0
for plyr in players:
    exp_moving_average(matches_df,plyr,'error_pca_1','error_pca_2','errors_pca_1_ema','errors_pca_2_ema')
    i=i+1
    print('iterations : ',i,' of ',len(players))


# In[44]:


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


# In[45]:


i=0
for player in players:
    get_nans(player,matches_df,'errors_pca_1_ema','errors_pca_2_ema')
    i=i+1
    print('iteration number :',i,' of ',len(players))


# In[46]:


#save 
matches_df.to_csv('final_df2.csv')


# In[ ]:




