#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
matches_df = pd.read_csv('final_df.csv')
matches_df


# In[3]:


list(matches_df.columns)


# In[4]:


#Prepare data for principal component analysis for Serve Rating for player 1
import numpy as np
col = ['aces_1', 'first_serve_made_1',
       'first_serve_attempted_1', 'first_serve_points_made_1',
       'first_serve_points_attempted_1', 'second_serve_points_made_1','games_won_1','games_against_1',
       'second_serve_points_attempted_1','service_games_won_1','service_points_won_1', 'service_points_attempted_1','return_games_played_2']
service_pca = matches_df[col].copy()
service_pca.dropna(inplace=True)
#remove rows with 0's in return games played 2
indx = service_pca.loc[service_pca['return_games_played_2']==0].index
service_pca.drop(index=indx,inplace=True)


# In[5]:



service_pca['0'] = service_pca['first_serve_made_1']/service_pca['first_serve_attempted_1']
service_pca['1'] = service_pca['service_points_won_1']/service_pca['service_points_attempted_1']
service_pca['2'] = service_pca['first_serve_points_made_1']/service_pca['first_serve_points_attempted_1']
service_pca['3'] = service_pca['second_serve_points_made_1']/service_pca['second_serve_points_attempted_1']
service_pca['4'] = service_pca['aces_1']/service_pca['service_points_attempted_1']
service_pca['5'] = (service_pca['service_games_won_1'])/(service_pca['return_games_played_2'])
service_pca


# In[6]:


service_pca.drop(col,axis=1,inplace=True)
service_pca.dropna(inplace=True)
from sklearn.preprocessing import StandardScaler
x = service_pca
x = x.values
x = StandardScaler().fit_transform(x) #data is standardized, this could be avoided because our data is composed of ratios
#but the pca results are better if standardized(in the sense that the principal component has values between approximately defined
#boundaries -10 and 10, while without standardization the outcome was mostly unbounded)
#standardization, since our data is made up of ratios,
#should have little to no impact on PCA dimensionality reduction
from sklearn.decomposition import PCA
pca_service = PCA(n_components=1)
princomp_service1 = pca_service.fit_transform(x)
pca_service.explained_variance_ratio_ #we can see here how much of the variance is explained by the component


# In[8]:


service1 = service_pca.copy()
service1['pc'] = princomp_service1
service1.sort_values('pc')
service1
#we can see that the  principal component has an inverse relationship with good service stats


# In[9]:


#return pca for player 1
colss = ['first_serve_return_points_made_1',
 'first_serve_return_points_attempted_1',
 'second_serve_return_points_made_1',
 'second_serve_return_points_attempted_1',
 'break_points_made_1',
 'break_points_attempted_1',
 'return_games_played_1','return_points_won_1',
 'return_points_attempted_1','games_won_1','service_games_won_1',
        'double_faults_2','first_serve_attempted_2','first_serve_made_2']
rpca = matches_df[colss].copy()
indx1 = rpca.loc[rpca['return_games_played_1']==0].index
rpca.drop(index=indx1,inplace=True)
indx2 = rpca.loc[rpca['first_serve_return_points_attempted_1']==0].index
rpca.drop(index=indx2,inplace=True)
rpca.dropna(inplace=True)
rpca


# In[10]:


#when computing return statistics we must take out the opponent's double faults, this is because when a double fault happens
#the player has no contribution whatsoever with it , it is only the opponent's mistake. so the player should not be rewarded
# when a double fault happens because he gained a point without doing anything.the same goes for first serve errors (made by the opponent)

rpca['0'] = (rpca['first_serve_return_points_made_1'] - (rpca['first_serve_attempted_2'] - rpca['first_serve_made_2']))/rpca['first_serve_return_points_attempted_1']
rpca['1'] = (rpca['second_serve_return_points_made_1']-rpca['double_faults_2'])/rpca['second_serve_return_points_attempted_1']
rpca['2'] = rpca['break_points_made_1']/rpca['break_points_attempted_1']
rpca['3'] = (rpca['return_points_won_1']-rpca['double_faults_2'])/rpca['return_points_attempted_1']
rpca['4'] = (rpca['games_won_1'] - rpca['service_games_won_1'])/(rpca['return_games_played_1'])
rpca.drop(colss,inplace=True,axis=1)
rpca.dropna(inplace=True)


# In[11]:


from sklearn.preprocessing import StandardScaler
x2 = rpca
x2 = x2.values
x2 = StandardScaler().fit_transform(x2)
from sklearn.decomposition import PCA
pca_return = PCA(n_components=1)
princomp_return1 = pca_return.fit_transform(x2)
pca_return.explained_variance_ratio_


# In[12]:


return1 = rpca.copy()
return1['pc'] = princomp_return1
return1.sort_values('pc')
return1


# In[13]:


#compute pca_service and pca_return for player 2
col2 = ['aces_2', 'first_serve_made_2',
       'first_serve_attempted_2', 'first_serve_points_made_2',
       'first_serve_points_attempted_2', 'second_serve_points_made_2','games_won_2','games_against_2',
       'second_serve_points_attempted_2','service_games_won_2','service_points_won_2', 'service_points_attempted_2','return_games_played_1']
service_pca2 = matches_df[col2].copy()
service_pca2.dropna(inplace=True)
service_pca2['0'] = service_pca2['first_serve_made_2']/service_pca2['first_serve_attempted_2']
service_pca2['1'] = service_pca2['service_points_won_2']/service_pca2['service_points_attempted_2']
service_pca2['2'] = service_pca2['first_serve_points_made_2']/service_pca2['first_serve_points_attempted_2']
service_pca2['3'] = service_pca2['second_serve_points_made_2']/service_pca2['second_serve_points_attempted_2']
service_pca2['4'] = service_pca2['aces_2']/service_pca2['service_points_attempted_2']
service_pca2['return_games_played_1'] = service_pca2['return_games_played_1'].replace(0,np.nan)
service_pca2['5'] = (service_pca2['service_games_won_2'])/(service_pca2['return_games_played_1'])
service_pca2
service_pca2.drop(col2,axis=1,inplace=True)
service_pca2.dropna(inplace=True)

from sklearn.preprocessing import StandardScaler
x3 = service_pca2
x3 = x3.values
x3 = StandardScaler().fit_transform(x3)
from sklearn.decomposition import PCA
pca_service2 = PCA(n_components=1)
princomp_service2 = pca_service2.fit_transform(x3)
pca_service2.explained_variance_ratio_


# In[14]:


service2 = service_pca2.copy()
service2['pc'] = princomp_service2
service2.sort_values('pc')
service2


# In[15]:


#return pca for player 2
colret = ['first_serve_return_points_made_2',
 'first_serve_return_points_attempted_2',
 'second_serve_return_points_made_2',
 'second_serve_return_points_attempted_2',
 'break_points_made_2',
 'break_points_attempted_2',
 'return_games_played_2','return_points_won_2',
 'return_points_attempted_2','games_won_2','service_games_won_2',
        'double_faults_1','first_serve_attempted_1','first_serve_made_1']
rpca2 = matches_df[colret].copy()
rpca2.dropna(inplace=True)
rpca2['first_serve_return_points_attempted_2'] = rpca2['first_serve_return_points_attempted_2'].replace(0,np.nan)
rpca2['0'] = (rpca2['first_serve_return_points_made_2'] - (rpca2['first_serve_attempted_1'] - rpca2['first_serve_made_1']))/rpca2['first_serve_return_points_attempted_2']
rpca2['1'] = (rpca2['second_serve_return_points_made_2']-rpca2['double_faults_1'])/rpca2['second_serve_return_points_attempted_2']
rpca2['2'] = rpca2['break_points_made_2']/rpca2['break_points_attempted_2']
rpca2['return_games_played_2'] = rpca2['return_games_played_2'].replace(0,np.nan)
rpca2['3'] = (rpca2['return_points_won_2']-rpca2['double_faults_1'])/rpca2['return_points_attempted_2']
rpca2['4'] = (rpca2['games_won_2'] - rpca2['service_games_won_2'])/(rpca2['return_games_played_2'])
rpca2.dropna(inplace=True)

rpca2.drop(colret,inplace=True,axis=1)
from sklearn.preprocessing import StandardScaler
x4 = rpca2
x4 = x4.values
x4 = StandardScaler().fit_transform(x4)
from sklearn.decomposition import PCA
pca_return2 = PCA(n_components=1)
princomp_return2 = pca_return2.fit_transform(x4)
pca_return2.explained_variance_ratio_


# In[16]:


#we can see that our princomps explain in average about 45% to 50% of the variance (45% for service and 50% for return)
#this is not bad, considering that in any way by synthetizing statistics into one indicator one would encounter
# a loss in explained variance. the goal though is to create an efficient pca for prediction, so explained variance , although acceptable,
#should not be a concern.


# In[17]:


return2 = rpca2.copy()
return2['pc'] = princomp_return2
return2.sort_values('pc')
return2


# In[18]:


#add them to matches_df in the correct index position
matches_df['service_pca_1'] = None
matches_df['service_pca_2'] = None
matches_df['return_pca_1'] = None
matches_df['return_pca_2'] = None


# In[19]:


service1.reset_index(inplace=True)
service2.reset_index(inplace=True)
return1.reset_index(inplace=True)
return2.reset_index(inplace=True)


# In[23]:


service1.sort_values('pc')


# In[27]:


service2.sort_values('pc')


# In[28]:


return1.sort_values('pc')


# In[22]:


return2.sort_values('pc')


# In[23]:


#adding the princomps to matches_df in the correct index position
for i in range(0,len(service1)):
    k = service1.iloc[i]['index']
    matches_df.at[k,'service_pca_1'] = service1.iloc[i]['pc']  
    k = None
for i in range(0,len(service2)):
    k = service2.iloc[i]['index']
    matches_df.at[k,'service_pca_2'] = service2.iloc[i]['pc']  
    k = None
for i in range(0,len(return1)):
    k = return1.iloc[i]['index']
    matches_df.at[k,'return_pca_1'] = return1.iloc[i]['pc']  
    k = None
for i in range(0,len(return2)):
    k = return2.iloc[i]['index']
    matches_df.at[k,'return_pca_2'] = return2.iloc[i]['pc'] 
    k = None


# In[24]:


matches_df


# In[32]:


#so we can say that the following relationships exists between the data and the princomps
# Good Service ---> high service pca
# Bad Service ---> negative service pca
# Good Return ---> high return pca
# Bad Return ---> negative return pca
# Service 1 PCA ---> [-10,10]
# Service 2 PCA ---> [-10,10]
# Return 1 PCA ---> [-6,10] (taking out the last three outliers that had princomp>10)
# Return 2 PCA ---> [-6,10] (also here taking out three or two outliers that where > 10)


# In[28]:


#save matches_df dataset
matches_df.to_csv('final_df.csv')


# In[ ]:


#KERNEL PCA and FINE TUNING

import pandas as pd
matches_df = pd.read_csv('final_df.csv')
matches_df

