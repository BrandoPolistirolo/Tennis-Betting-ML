#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('final_df.csv')


# In[3]:


df


# In[4]:


df.columns


# In[5]:


import numpy as np
col = ['aces_1', 'first_serve_made_1','double_faults_1',
       'first_serve_attempted_1', 'first_serve_points_made_1',
       'first_serve_points_attempted_1', 'second_serve_points_made_1','games_won_1','games_against_1',
       'second_serve_points_attempted_1','service_games_won_1','service_points_won_1', 'service_points_attempted_1','return_games_played_2'
      ,'player_1_victory']
service_pca = df[col].copy()
service_pca.dropna(inplace=True)
#remove rows with 0's in return games played 2
indx = service_pca.loc[service_pca['return_games_played_2']==0].index
service_pca.drop(index=indx,inplace=True)


# In[6]:


service_pca


# In[7]:


service_pca['0'] = service_pca['first_serve_made_1']/service_pca['first_serve_attempted_1']
service_pca['1'] = service_pca['service_points_won_1']/service_pca['service_points_attempted_1']
service_pca['2'] = service_pca['first_serve_points_made_1']/service_pca['first_serve_points_attempted_1']
service_pca['3'] = service_pca['second_serve_points_made_1']/service_pca['second_serve_points_attempted_1']
service_pca['4'] = service_pca['aces_1']/service_pca['service_points_attempted_1']
service_pca['5'] = service_pca['double_faults_1']/(service_pca['first_serve_attempted_1'] + service_pca['second_serve_points_attempted_1'])
service_pca['6'] = (service_pca['service_games_won_1'])/(service_pca['return_games_played_2'])
service_pca


# In[8]:


#drop unnecessary columns
col = col[0:len(col)-1]
service_pca.drop(col,axis=1,inplace=True)
service_pca.dropna(inplace=True)
service_pca


# In[14]:


#service pca for player 2
col2 = ['aces_2', 'first_serve_made_2','double_faults_2',
       'first_serve_attempted_2', 'first_serve_points_made_2',
       'first_serve_points_attempted_2', 'second_serve_points_made_2','games_won_2','games_against_2',
       'second_serve_points_attempted_2','service_games_won_2','service_points_won_2', 'service_points_attempted_2','return_games_played_1']
service_pca2 = df[col2].copy()
service_pca2.dropna(inplace=True)
service_pca2['0'] = service_pca2['first_serve_made_2']/service_pca2['first_serve_attempted_2']
service_pca2['1'] = service_pca2['service_points_won_2']/service_pca2['service_points_attempted_2']
service_pca2['2'] = service_pca2['first_serve_points_made_2']/service_pca2['first_serve_points_attempted_2']
service_pca2['3'] = service_pca2['second_serve_points_made_2']/service_pca2['second_serve_points_attempted_2']
service_pca2['4'] = service_pca2['aces_2']/service_pca2['service_points_attempted_2']
service_pca2['return_games_played_1'] = service_pca2['return_games_played_1'].replace(0,np.nan)
service_pca2['5'] = service_pca2['double_faults_2']/(service_pca2['first_serve_attempted_2'] + service_pca2['second_serve_points_attempted_2'])
service_pca2['6'] = (service_pca2['service_games_won_2'])/(service_pca2['return_games_played_1'])

service_pca2.drop(col2,axis=1,inplace=True)
service_pca2.dropna(inplace=True)
service_pca2


# In[16]:


#plot 
for i in range(0,7):
    for j in range(0,7):
        if i==j:
            continue
        service_pca.plot.scatter(x=str(i),y=str(j))


# In[ ]:


import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.decomposition import KernelPCA

#try kernel pca
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
gammas = [0.01,0.05,0.5,1,3,5,10] #for rbf, poly and sigmoid kernels
degrees = [2,3,4,5] #for poly kernel

def cv_and_training(dataset1,dataset2,train_size,):
    #data merge, use dataframe with minor index size
    data = 
    #define classifier
    clf = SGDClassifier(loss="log", fit_intercept=False,max_iter=1000,shuffle=False,verbose=1,early_stopping=True
                        ,n_iter_no_change=25,tol=0.000001)
    param_dist = {'alpha':10.0**-np.arange(1,7)}
    rsh = RandomizedSearchCV(estimator=clf, param_distributions=param_dist,return_train_score=True
                             ,n_iter=6,cv=KFold(30),verbose=1)
    rsh.fit(x_train, y_train)
    
    return (rsh.best_score_,rsh.best_estimator_)

for item in kernels:
    if item in ['rbf','poly','sigmoid']:
        for gamma_pam in gammas:
            
            if item=='poly':
                for degree in degrees:
                    #####
            kpca = KernelPCA(kernel=item, gamma=gamma_pam)
            #fit 
            X_kpca = kpca.fit_transform(service_pca['0','1','2','3','4','5','6'])
            Y_kpca = kpca.fit_transform(service_pca2)
            X_kpca['target'] = service_pca['player_1_victory']
            final_kpca = 
    


# In[16]:


from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2,kernel='rbf', gamma=10,copy_X=False)
X_kpca = kpca.fit_transform(service_pca2[:10000])


# In[29]:


Y_kpca = kpca.transform(service_pca2[10000:])


# In[ ]:


#ABANDONED : not computationally possible to do with kernelPCA
#maybe streaming KernelPCA could work

