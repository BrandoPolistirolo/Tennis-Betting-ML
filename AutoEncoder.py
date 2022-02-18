#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('final_df.csv')


# In[2]:


df.columns


# In[3]:


#get statistics for player 1
col1 = ['sets_won_1','games_won_1', 'games_against_1', 'tiebreaks_won_1',
      'aces_1', 'double_faults_1', 'first_serve_made_1',
       'first_serve_attempted_1', 'first_serve_points_made_1',
       'first_serve_points_attempted_1', 'second_serve_points_made_1',
       'second_serve_points_attempted_1', 'break_points_saved_1',
       'break_points_against_1', 'service_games_won_1',
      'first_serve_return_points_made_1',
       'first_serve_return_points_attempted_1',
       'second_serve_return_points_made_1',
       'second_serve_return_points_attempted_1', 'break_points_made_1',
       'break_points_attempted_1', 'return_games_played_1',
       'service_points_won_1', 'service_points_attempted_1',
       'return_points_won_1', 'return_points_attempted_1',
       'total_points_won_1','player_1_victory']
player_1_reduction = df[col1].copy()


# In[4]:


# statistics for player 2
col2 = ['sets_won_2','games_won_2', 'games_against_2', 'tiebreaks_won_2',
      'aces_2', 'double_faults_2', 'first_serve_made_2',
       'first_serve_attempted_2', 'first_serve_points_made_2',
       'first_serve_points_attempted_2', 'second_serve_points_made_2',
       'second_serve_points_attempted_2', 'break_points_saved_2',
       'break_points_against_2', 'service_games_won_2',
      'first_serve_return_points_made_2',
       'first_serve_return_points_attempted_2',
       'second_serve_return_points_made_2',
       'second_serve_return_points_attempted_2', 'break_points_made_2',
       'break_points_attempted_2', 'return_games_played_2',
       'service_points_won_2', 'service_points_attempted_2',
       'return_points_won_2', 'return_points_attempted_2',
       'total_points_won_2']
player_2_reduction = df[col2].copy()


# In[5]:


player_1_reduction.dropna(inplace=True)
player_2_reduction.dropna(inplace=True)


# In[6]:


player_1_reduction


# In[7]:


player_2_reduction


# In[8]:


#we want 3 components
#standardize
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#fit scaler
#keep index!!!
x1 = scaler.fit_transform(player_1_reduction.drop('player_1_victory',axis=1)[:int(len(player_1_reduction)*7/10)])
y1 = scaler.transform(player_1_reduction.drop('player_1_victory',axis=1)[int(len(player_1_reduction)*7/10):])
x2 = scaler.fit_transform(player_2_reduction[:int(len(player_2_reduction)*7/10)])
y2 = scaler.transform(player_2_reduction[int(len(player_2_reduction)*7/10):])


# In[9]:


xtrain1 = pd.DataFrame(x1,index=player_1_reduction[:int(len(player_1_reduction)*7/10)].index)
xtest1 = pd.DataFrame(y1,index=player_1_reduction[int(len(player_1_reduction)*7/10):].index)

xtrain2 = pd.DataFrame(x2,index=player_2_reduction[:int(len(player_2_reduction)*7/10)].index)
xtest2 = pd.DataFrame(y2,index=player_2_reduction[int(len(player_2_reduction)*7/10):].index)


# In[10]:


xtest1


# In[16]:


#first autoencoder
import keras
from keras import layers

input_dat = keras.Input(shape=(27,))
encoded = layers.Dense(27, activation='relu')(input_dat)
encoded = layers.Dense(16, activation='relu')(encoded)
encoded = layers.Dense(2, activation='relu')(encoded)
encoder = keras.Model(input_dat,encoded)
decoded = layers.Dense(2, activation='relu')(encoded)
decoded = layers.Dense(16, activation='relu')(decoded)
decoded = layers.Dense(27, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_dat, decoded)
autoencoder.compile(optimizer='adam', loss='mae')

autoencoder.fit(xtrain1, xtrain1,
                epochs=100,
                batch_size = 256,
                shuffle=True,
                validation_data=(xtest1, xtest1))


# In[ ]:


#fit on second player
autoencoder.fit(xtrain2, xtrain2,
                epochs=100,
                batch_size = 256,
                shuffle=True,
                validation_data=(xtest2, xtest2))


# In[43]:


pred = encoder.predict(xtest)


# In[44]:


pred


# In[45]:


dat = pd.DataFrame(pred)


# In[46]:


dat


# In[11]:


#second autoencoder
import keras
from keras import layers

input_dat1 = keras.Input(shape=(27,))
encoded1 = layers.Dense(27, activation='relu')(input_dat1)

encoded1 = layers.Dense(21,activation = 'relu')(encoded1)

encoded1 = layers.Dense(15, activation='relu')(encoded1)

encoded1 = layers.Dense(9,activation = 'relu')(encoded1)

encoded1 = layers.Dense(2, activation='relu')(encoded1)

encoder1 = keras.Model(input_dat1,encoded1)

decoded1 = layers.Dense(2, activation='relu')(encoded1)

encoded1 = layers.Dense(9,activation = 'relu')(decoded1)

encoded1 = layers.Dense(15, activation='relu')(decoded1)

encoded1 = layers.Dense(21,activation = 'relu')(decoded1)

decoded1 = layers.Dense(27, activation='sigmoid')(decoded1)

autoencoder1 = keras.Model(input_dat1, decoded1)
autoencoder1.compile(optimizer='adam', loss='mse')


autoencoder1.fit(xtrain1, xtrain1,
                epochs=150,
                batch_size = 512,
                shuffle=True,
                validation_data=(xtest1, xtest1))


# In[44]:


#fit on player 2
# not needed if fit on player 1
autoencoder1.fit(xtrain2, xtrain2,
                epochs=100,
                batch_size = 256,
                shuffle=True,
                validation_data=(xtest2, xtest2))


# In[13]:


#merge test and train for player 1
scaled_plyr1 = pd.concat([xtrain1,xtest1])
scaled_plyr2 = pd.concat([xtrain2,xtest2])


# In[14]:


#predict on player 1
pred_plyr1 = encoder1.predict(scaled_plyr1)
pred_plyr1


# In[16]:


plyr1_encoded = pd.DataFrame(pred_plyr1)


# In[23]:


plyr1_encoded[1].max()


# In[24]:


plyr2_encoded = pd.DataFrame(encoder1.predict(scaled_plyr2))


# In[25]:


plyr2_encoded


# In[30]:


a = pd.merge(plyr1_encoded,pd.DataFrame(player_1_reduction['player_1_victory']),left_index=True,right_index=True)


# In[32]:


b = pd.merge(plyr2_encoded,a,left_index=True,right_index=True)


# In[39]:


b


# In[37]:


for i in range(0,len(b)):
    if b.iloc[i]['player_1_victory'] == 't':
        b.at[i,'player_1_victory'] = 1
    else:
        b.at[i,'player_1_victory'] = 0


# In[38]:


b


# In[ ]:


import numpy as np
from sklearn.linear_model import SGDClassifier
#define classifier
clf = SGDClassifier(loss="log", fit_intercept=False,max_iter=1000,shuffle=False,verbose=1,early_stopping=True
                    ,n_iter_no_change=25,tol=0.000001)



from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

param_dist = {'alpha':10.0**-np.arange(1,7)}
rsh = RandomizedSearchCV(estimator=clf, param_distributions=param_dist,return_train_score=True
                             ,n_iter=6,cv=KFold(30),verbose=1)

#split data
x_train = 
x_test = 
y_train =
y_test = 

rsh.fit(x_train, y_train)

