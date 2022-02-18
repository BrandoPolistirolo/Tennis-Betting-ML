#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
matches_df = pd.read_csv('final_df2.csv')


# In[2]:


matches_df


# In[3]:


#select columns that we need
list(matches_df.columns)


# In[4]:


df = matches_df[[
    'player_1_v',
    'court_surface',
    'start_date',
    'player_id',
    'opponent_id',
    'elo_1_surface',
    'elo_2_surface',
    'elo_1',
    'elo_2',
    'service_pca_1_ema',
    'service_pca_2_ema',
    'return_pca_1_ema',
    'return_pca_2_ema',
    'hth_1',
    'hth_2',
    'performance_1_ema',
    'performance_2_ema',
    'player age',
    'opponent age',
    'errors_pca_1_ema',
    'errors_pca_2_ema']].copy()


# In[5]:


df


# In[6]:


df['delta_elo_surface'] = df['elo_1_surface'] - df['elo_2_surface']
df['delta_elo'] = df['elo_1'] - df['elo_2']
df['delta_service'] = df['service_pca_1_ema'] - df['service_pca_2_ema']
df['delta_return'] = df['return_pca_1_ema'] - df['return_pca_2_ema']
df['delta_performance'] = df['performance_1_ema'] - df['performance_2_ema']
df['delta_service_mistakes'] = df['errors_pca_1_ema'] - df['errors_pca_2_ema']
df['delta_age'] = df['player age'] - df['opponent age']


# In[7]:


df


# In[8]:


#clean the data for NA values
df.dropna(inplace = True,subset=['delta_elo_surface','delta_elo','delta_service','delta_return','delta_performance','delta_service_mistakes','delta_age','hth_1'])


# In[9]:


df


# In[10]:


#we have a total of 125938 matches we are going to split training/test ina a 70/30 percent ratio
#take out unnecessary columns
df = df[['player_1_v',
        'delta_elo_surface',
         'delta_elo',
         'delta_service',
         'delta_return',
         'delta_performance',
         'delta_service_mistakes',
         'delta_age',
         'hth_1']]


# In[11]:


df
#we have a total of 8 features


# In[12]:


df.to_csv('final_df3.csv')


# In[25]:


#plot

df.plot(kind='density',subplots=True,figsize=(10,10))


# In[26]:


#divide the dateset in training and test
x_train = df.iloc[0:88156][['delta_elo_surface',
         'delta_elo',
         'delta_service',
         'delta_return',
         'delta_performance',
         'delta_service_mistakes',
         'delta_age',
         'hth_1']]
x_test = df.iloc[88156:][['delta_elo_surface',
         'delta_elo',
         'delta_service',
         'delta_return',
         'delta_performance',
         'delta_service_mistakes',
         'delta_age',
         'hth_1']]
y_train = df.iloc[0:88156]['player_1_v']
y_test = df.iloc[88156:]['player_1_v']


# In[27]:


x_train


# In[28]:


x_test


# In[29]:


#perform scaling and trasformations on data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train.values)  # Don't cheat - fit only on training data
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)  # apply same transformation to test data


# In[30]:


x_train


# In[31]:


x_test


# In[32]:


df_trasformed = pd.DataFrame(x_train)
df_trasformed.columns = ['delta_elo_surface',
         'delta_elo',
         'delta_service',
         'delta_return',
         'delta_performance',
         'delta_service_mistakes',
         'delta_age',
         'hth_1']


# In[34]:


df_trasformed.plot(kind='density',subplots=True,figsize=(10,10))


# In[66]:


conda install -c conda-forge scikit-learn


# In[33]:


#now we can do training
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
rsh.fit(x_train, y_train)


# In[34]:


rsh.best_estimator_


# In[35]:


best = rsh.best_estimator_


# In[39]:


#learning curve
from sklearn.model_selection import learning_curve
a,b,c =  learning_curve(best, x_train, y_train, 
                   scoring="accuracy", cv=KFold(30))


# In[56]:


import matplotlib.pyplot as plt
plt.plot(a, c.mean(1), 'o-', color="r",
         label="SGD Logistic")
plt.xlabel("Train Size")
plt.ylabel('Accuracy')
plt.title('Learning curves')


# In[43]:


#learning curve
from sklearn.model_selection import learning_curve
d,e,f =  learning_curve(best, x_train, y_train, 
                   scoring="neg_log_loss", cv=KFold(30))


# In[55]:


plt.plot(d, -f.mean(1), 'o-', color="r",
         label="SGD Logistic - log loss")
plt.xlabel("Train Size")
plt.ylabel('Log Loss')
plt.title('Learning curves')


# In[36]:


best.score(x_train,y_train)


# In[37]:


rsh.cv_results_


# In[38]:


from sklearn.metrics import classification_report
y_pred = best.predict(x_test)
print(classification_report(y_test, y_pred))


# In[166]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, best.predict(x_test))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
disp = plot_precision_recall_curve(best, x_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))


# In[164]:


average_precision


# In[65]:


from sklearn.metrics import plot_roc_curve
sgd_disp = plot_roc_curve(best, x_test, y_test)
plt.show()


# In[39]:


prob = best.predict_proba(x_test)


# In[40]:


prob


# In[41]:


prob.shape


# In[42]:


params = best.get_params()


# In[43]:


params


# In[107]:


coef = best.coef_


# In[108]:


coef
labels = ['delta_elo_surface','delta_elo','delta_service','delta_return',
          'delta_performance','delta_service_mistakes','delta_age','hth_1']


# In[112]:


coef = [ 0.33239367,  0.54154889,  0.12990599,  0.02189188, -0.10584747,
        -0.08630714,  0.01451121,  0.04684119]


# In[85]:


y_axis = labels
x_axis = [ 0.33239367,  0.54154889,  0.12990599,  0.02189188, -0.10584747,
        -0.08630714,  0.01451121,  0.04684119]

plt.barh(y_axis,x_axis)
plt.title('Features Weights')
plt.ylabel('Features')
plt.xlabel('Weights')
plt.show()


# In[113]:


#interpretability
#let's start by getting the log odds
import math
oddsratio = []
for i in coef:
    oddsratio.append(math.exp(i))
    
oddsratio


# In[114]:


y_axis = labels
x_axis = oddsratio
plt.barh(y_axis,x_axis)
plt.title('Odds Ratios')
plt.ylabel('Features')
plt.xlabel('OR')
plt.show()


# In[115]:


pip install shap


# In[119]:


#SHAP
import shap


explainer = shap.Explainer(best, x_train, feature_names=labels)
shap_values = explainer(x_test)


# In[120]:


shap.plots.beeswarm(shap_values)


# In[ ]:





# In[44]:


#let's make a betting simulation using odds data
df_bet = matches_df[[
    'player_1_v',
    'court_surface',
    'start_date',
    'player_id',
    'opponent_id',
    'elo_1_surface',
    'elo_2_surface',
    'elo_1',
    'elo_2',
    'service_pca_1_ema',
    'service_pca_2_ema',
    'return_pca_1_ema',
    'return_pca_2_ema',
    'hth_1',
    'hth_2',
    'performance_1_ema',
    'performance_2_ema',
    'player age',
    'opponent age',
    'errors_pca_1_ema',
    'errors_pca_2_ema']].copy()

df_bet['delta_elo_surface'] = df_bet['elo_1_surface'] - df_bet['elo_2_surface']
df_bet['delta_elo'] = df_bet['elo_1'] - df_bet['elo_2']
df_bet['delta_service'] = df_bet['service_pca_1_ema'] - df_bet['service_pca_2_ema']
df_bet['delta_return'] = df_bet['return_pca_1_ema'] - df_bet['return_pca_2_ema']
df_bet['delta_performance'] = df_bet['performance_1_ema'] - df_bet['performance_2_ema']
df_bet['delta_service_mistakes'] = df_bet['errors_pca_1_ema'] - df_bet['errors_pca_2_ema']
df_bet['delta_age'] = df_bet['player age'] - df_bet['opponent age']


df_bet.dropna(inplace = True,subset=['delta_elo_surface','delta_elo','delta_service','delta_return','delta_performance','delta_service_mistakes','delta_age','hth_1'])

bet = df_bet.iloc[88156:][['player_id','opponent_id',
    'start_date','player_1_v'
    ]]


# In[45]:


bet


# In[46]:


bet['proba_0'] = prob[:,0]
bet['proba_1'] = prob[:,1]


# In[47]:


bet


# In[48]:


#import odds
odds = pd.read_csv('betting_moneyline.csv',parse_dates=[1],infer_datetime_format=True)


# In[49]:


odds


# In[50]:


odds = odds.loc[odds['start_date']>='2015-04-06'].copy()


# In[51]:


odds


# In[52]:


odds.rename(columns = {'team1':'player_id','team2':'opponent_id'}, inplace = True)


# In[53]:


odds


# In[54]:


bet = bet.merge(odds,on=['start_date','player_id','opponent_id'])


# In[60]:


bet


# In[56]:


#take out doubles in bet
bet.drop_duplicates(subset=['player_id','opponent_id','start_date'],inplace=True)


# In[59]:


#now we can simulate with a bankroll of 1000 usd
bet['quota_1'] = 1/bet['odds1']
bet['quota_2'] = 1/bet['odds2']


# In[64]:


bet['proba_0'].plot(kind='density')


# In[65]:


bet['proba_1'].plot(kind='density')


# In[94]:


budget = 1000
#simple backtest, choose stake for every bet and see results,we only bet on matches that have a predicted
#probability ( estimated by our model) higher than p, also the bet is placed only if p > odds given by the bookmaker
#the staked is computed with the Kelly criterion.

# 0 ---> player 1 victory = False so opponent wins
# 1 ---> player 1 victory = True so player wins
#stake is a number between 1 and 0 , the percentage of the budget you bet on each game, 
#then for every bet the kelly criterion computes the optimal betting quantity
#the kelly criterion only bets a certain amount if it believes that the bettor has an edge
#stake should be a number between 0 and 0.2 for realistical backtesting

def bet_backtest(budget,stake,data,p):
    bank = []
    bank.append(budget)
    bets = []
    bets.append(0)
    stake1 = stake*budget
    k = 0
    bets_won = 0
    bets_lost = 0
    for i in range(0,len(data)):
        if (data.iloc[i]['proba_0'] > p) & (data.iloc[i]['proba_0'] > data.iloc[i]['odds2']):
            net_odds = data.iloc[i]['odds2']
            stake_usd = stake1*((data.iloc[i]['proba_0'])*(net_odds + 1) - 1)
            stake_usd = stake_usd/net_odds
            #bet on opponent
            if data.iloc[i]['player_1_v'] == 0:
                #win bet
                bank.append(bank[k] + stake_usd*data.iloc[i]['quota_2'] - stake_usd)
                bets_won = bets_won + 1
            if data.iloc[i]['player_1_v'] == 1:
                #lose bet
                bank.append(bank[k] - stake_usd)
                bets_lost = bets_lost + 1 
            k = k + 1 
            bets.append(k)
        if (data.iloc[i]['proba_1'] > p) & (data.iloc[i]['proba_1'] > data.iloc[i]['odds1']):
            net_odds = data.iloc[i]['odds1']
            stake_usd = stake*((data.iloc[i]['proba_1'])*(net_odds + 1) - 1)
            stake_usd = stake_usd/net_odds
            #bet on player
            if data.iloc[i]['player_1_v'] == 0:
                #lose bet
                bank.append(bank[k] - stake_usd)
                bets_lost = bets_lost + 1 

            if data.iloc[i]['player_1_v'] == 1:
                #win bet
                bank.append(bank[k] + stake_usd*data.iloc[i]['quota_1'] - stake_usd)
                bets_won = bets_won + 1

            k = k + 1
            bets.append(k)
    print(bets_won,bets_lost)       
    return bets,bank


# In[95]:


#do backtesting 
#we only bet on games that have a predicted probability higher than p=0.9
bets,bank = bet_backtest(1000,0.1,bet,0.9)
bets


# In[96]:


bank


# In[105]:


#plot results
import matplotlib.pyplot as plt
# plot a line chart
plt.plot(bets, bank)
# set axis titles
plt.xlabel("Bets")
plt.ylabel("Budget")
# set chart title
plt.title("Backtest with Stake - 0.1")
plt.show()


# In[103]:


bets1,bank1 = bet_backtest(1000,0.05,bet,0.9)


# In[104]:


# plot a line chart
plt.plot(bets1, bank1)
# set axis titles
plt.xlabel("Bets")
plt.ylabel("Budget")
# set chart title
plt.title("Backtest with Stake - 0.05")
plt.show()


# In[106]:





# In[ ]:




