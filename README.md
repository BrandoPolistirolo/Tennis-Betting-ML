# Tennis-Betting-ML
Machine Learning model(specifically log-regression with stochastic gradient descent) for tennis matches prediction. Achieves a 66% accuracy rate on approximately 125000 matches (only singles)
# Abstract
In this project we built a machine learning algorithm for tennis match predictions. We used a logistic model with Stochastic Gradient Descent. 
# Summary
This assignment was divided in four parts. 1) Data Processing, 2) Features engineering and extraction 3) Training and validation 4) Interpretability  

# Introduction
We have chosen to predict outcome of tennis matches because it is a sport that implies one winner for every game so there isn't tie result, there are just two possible outcomes and just two players for each match: we take advantages by the structure of scoring.\\This system in tennis has a hierarchical structure, with a match being composed of sets, which in turn are composed of games, which are composed of individual points. By assuming that points are independently and identically distributed (iid), the expressions only need the probabilities of the two players winning a point on their serve.\\Another reason to choose tennis is because the statistics are available and complete. 

# Tennis Betting 
There are two main categories of tennis betting: pre-game and in-game. We will focus only on the pre-game bets. Bets on tennis matches can be placed either with bookmakers or on betting exchanges. Traditional bookmakers set odds for the different outcomes of a match and a bettor competes against the bookmakers. In the case of betting exchanges costumers can bet against odds set by other costumers.The exchange matches the customers’ bets to earn a risk-free profit by charging a commission on each bet matched. Betting odds represents the return a bettor receives from correctly predicting the outcome of an event. Those provide an implied probability of the outcome of a match, the bookmakers' estimate of the true probability. For example for odds X/Y for a player winning a match, the implied probability p of winning is <img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;p=&space;\frac{Y}{Y&plus;X}&space;\end{equation}\\" title="\begin{equation} p= \frac{Y}{Y+X} \end{equation}\\" /> \
Given the betting odds and a predicted probability of a match outcome, different strategies will result in a different return on investment (profit or loss resulting from a bet).

# The Dataset

The Dataset was taken from Kaggle, the creator is Evan Hallmark. It is a very large dataset that contains a lot of useful data, for example betting odds of various kind. The data was scraped by the creator of the dataset from the official ATP World Tour website, since we cannot see the scraping program we do not have a clear idea of how reliable the data is. Web Scraping is often a faulty process, and is very rarely void of errors. We chose this dataset mainly because by looking around online we found that the best source of tennis statistics is without a doubt the official ATP website. We found that this dataset was the most complete among the ones that scraped the ATP website. The following table is a full account of the columns contained in the dataset.

![1](https://user-images.githubusercontent.com/87206202/154717792-7c7edd93-2e9f-440e-815b-92c9019792a9.PNG)
![2](https://user-images.githubusercontent.com/87206202/154717908-41929f5f-35e1-4fc8-8a13-047b79d09940.PNG)\
![3](https://user-images.githubusercontent.com/87206202/154717901-cf964e13-b107-4adb-b58e-2d9b0bb04e46.PNG)\

# Data Processing
The original dataset needed cleaning, the following operations were performed :

    - we only selected matches that happened after the year 2000
    - the original dataset sampled every match twice, in other words every match was split into two rows : 
    one row regarding player i and another row regarding player j. the difficulty was that only the statistics for player i were present in every row.  The rows had to be merged in order to obtain a row for every match that had statistics and data for both players. To do this data was split by tournament, on top of the  separate tournament-specific dataframes the matches were merged. This was done in order to avoid confusion between matches that had the same players involved but in different  tournaments or years.
    - players data was cleaned, the result was a dataframe of only the players that took part in the games that we were interested in. This was done because we needed the date of birth of every player.
    - players Name fix : the player and opponent id's had to be fixed, some player names had a space instead of a '-' between first , middle or last name.

# Features

## Elo Ratings
Elo Ratings were computed for all relevant matches in the dataset. The method used is Elo 538, which is a variant of the original Elo rating system commonly used in chess.
 For a match at time t between pi and pj with Elo ratings Ei(t) and Ej(t), pi is forecasted to win with probability: \
 <img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;\large&space;\widehat{\pi}_{ij}(t)&space;=&space;(1&space;&plus;&space;10^{\frac{E_{j}(t)&space;-&space;E_{i}(t)}{400}})^{-1}&space;\end{equation}" title="\begin{equation} \large \widehat{\pi}_{ij}(t) = (1 + 10^{\frac{E_{j}(t) - E_{i}(t)}{400}})^{-1} \end{equation}" />\
 pi’s rating for the following match t + 1 is then updated accordingly: \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;\large&space;E_{i}(t&plus;1)&space;=&space;E_{i}(t)&space;&plus;&space;K_{it}*(W_{i}(t)-\widehat{\pi}_{ij}(t))\end{equation}" title="\begin{equation} \large E_{i}(t+1) = E_{i}(t) + K_{it}*(W_{i}(t)-\widehat{\pi}_{ij}(t))\end{equation}" />\
Wi(t) is an indicator for whether pi won the given match, while Kit is the learning rate for pi at
time t. According to FiveThirtyEight’s analysts, Elo ratings perform optimally when allowing
Kit to decay slowly over time. With mi(t) representing pi’s career matches
played at time t we update our learning rate: \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}\large&space;K_i_t=\frac{250}{(5&plus;m_i(t))^.^4}\end{equation}" title="\begin{equation}\large K_i_t=\frac{250}{(5+m_i(t))^.^4}\end{equation}" /> \
This variant updates a player’s Elo most quickly when we have no information about a player
and makes smaller changes as mi(t) accumulates. To apply this Elo rating method to our
dataset, we initalize each player’s Elo rating at Ei(0) = 1500 and match history mi(0) = 0.
Then, we iterate through all tour-level matches from 1968-2017 in chronological order, storing
Ei(t), Ej (t) for each match and updating each player’s Elo accordingly. 

## Exponential Moving Average
Some of the features were created using an exponential moving average. In particular: Service rating, return rating and performance.
The idea behind this was to find a method that would average the features values but would also give more importance to the last instances and less importance to the more distant ones. For example, as a player progresses through his career his service rating will change according to the matches he plays and how well he plays them, by using the EMA one bad match does not influence the rating significantly over time. Essentially we want to track the player performance without giving too much weight to games played years before. 
the EMA is commonly used in finance to track the price of assets, and to reduce volatility when trying to analyze a specific trend. 
In our case the prices were the values of features for every player, we computed the EMA's with an iterative method. Of course for every match only the matches played by pi before the match in question were used to compute the EMA. 
The method is described as : \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;\large&space;EMA_{t}&space;=&space;\frac{p_{1}&plus;(1-\alpha)*p_{2}&plus;(1-\alpha)^{2}*p_{3}\cdots}{1&plus;(1-\alpha)&plus;(1-\alpha)^{2}\cdots}=\frac{WeightedSum_n}{WeightedCount_n}\end{equation}" title="\begin{equation} \large EMA_{t} = \frac{p_{1}+(1-\alpha)*p_{2}+(1-\alpha)^{2}*p_{3}\cdots}{1+(1-\alpha)+(1-\alpha)^{2}\cdots}=\frac{WeightedSum_n}{WeightedCount_n}\end{equation}" />  \
we assume : \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;\large&space;WeightesSum_0=WeightedCount_0=0\end{equation}" title="\begin{equation} \large WeightesSum_0=WeightedCount_0=0\end{equation}" /> \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;&space;\large&space;WeightedSum_n&space;=&space;p_n&space;&plus;&space;(1-\alpha)*WeightedSum_{n-1}\end{equation}" title="\begin{equation} \large WeightedSum_n = p_n + (1-\alpha)*WeightedSum_{n-1}\end{equation}" /> \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;&space;\large&space;WeightedCount_n&space;=&space;1&space;&plus;&space;(1-\alpha)*WeightedCount_{n-1}&space;\end{equation}&space;&space;" title="\begin{equation} \large WeightedCount_n = 1 + (1-\alpha)*WeightedCount_{n-1} \end{equation} " />  
Alpha is chosen using the rule : \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;\large&space;\alpha&space;=&space;\frac{2}{N&plus;1}\end{equation}" title="\begin{equation} \large \alpha = \frac{2}{N+1}\end{equation}" />\
where N is the number of matches played by a certain player

## Service Rating 
Considering the matches' statistics, we describe 6 indices for the Service: \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;1)&space;\large&space;\frac{Aces}{TotServices}\end{equation}\begin{equation}&space;&space;2)&space;\large&space;\frac{FirstServicesMade}{FirtServices}\end{equation}\begin{equation}&space;&space;3)&space;\large&space;\frac{ServiceGamesWon}{ServiceGamesPlayed}\end{equation}" title="\begin{equation} 1) \large \frac{Aces}{TotServices}\end{equation}\begin{equation} 2) \large \frac{FirstServicesMade}{FirtServices}\end{equation}\begin{equation} 3) \large \frac{ServiceGamesWon}{ServiceGamesPlayed}\end{equation}" />

<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;4)&space;\large&space;\frac{PointsSecondServiceWon}{PointsSecondServicePlayed}\end{equation}\begin{equation}&space;&space;5)&space;\large&space;\frac{PointsFirstServiceWon}{PointsFirstServicePlayed}&space;\end{equation}\begin{equation}&space;&space;6)&space;\large&space;\frac{PointsScoredForService}{PointsPlayedForService}\end{equation}" title="\begin{equation} 4) \large \frac{PointsSecondServiceWon}{PointsSecondServicePlayed}\end{equation}\begin{equation} 5) \large \frac{PointsFirstServiceWon}{PointsFirstServicePlayed} \end{equation}\begin{equation} 6) \large \frac{PointsScoredForService}{PointsPlayedForService}\end{equation}" />


Aiming at dimensionality reduction we summarize those indexes, using Principal Components Analysis (PCA).\\ PCA analysis consists in a linear transformation necessary for rebuilt (interpolate) data without loosing too much information, using the dependencies between the variables to represent data in a more tractable way. We assume data as centred, so with mean = 0. \\
The main goal is to maximize variance; it might sound more plausible to look for the projection with the smallest average (mean-squared) distance between the original vectors and their projections on to the principal components; this turns out to be equivalent to maximizing the variance.
Through this method, we obtain one principal component for the first six indices about Service: this new variable has an Explained Variance of 45\%.  \
Our PCA application resulted in a one-dimensional synthetic index that ranged from [-10,10].Positive values of this service pca indicated good service statistics while negative values indicated poor statistics. This index was then taken and used to compute an exponential moving average for every player. To face missing values the last known EMA value was taken and replaced with NA values. This way there was no bias introduction because the Exponential Moving Average of a player's service PCA index is a representation of his career performance.

## Return Rating 
We also need to define 5 more indices for the Return:\\
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;1)&space;\large&space;\frac{FirstServeReturnPointsMade-FirstServeErrorsOpponents}{FirstServeReturnPointsAttempted}\end{equation}&space;\begin{equation}&space;&space;2)&space;\large&space;\frac{SecondServeReturnPointsMade-DoubleFaultsOpponent}{SecondServeReturnPointsAttempted}\end{equation}" title="\begin{equation} 1) \large \frac{FirstServeReturnPointsMade-FirstServeErrorsOpponents}{FirstServeReturnPointsAttempted}\end{equation} \begin{equation} 2) \large \frac{SecondServeReturnPointsMade-DoubleFaultsOpponent}{SecondServeReturnPointsAttempted}\end{equation}" />
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;3)&space;\large&space;\frac{ReturnPointsWon-DoubleFaultsOpponent}{ReturnPointsAttempted}&space;\end{equation}\begin{equation}&space;&space;4)&space;\large&space;\frac{BreackPointsMade}{BreakPointsAttempted}\end{equation}" title="\begin{equation} 3) \large \frac{ReturnPointsWon-DoubleFaultsOpponent}{ReturnPointsAttempted} \end{equation}\begin{equation} 4) \large \frac{BreackPointsMade}{BreakPointsAttempted}\end{equation}" />
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;5)&space;\large&space;\frac{GamesWon-ServiceGamesWon}{ReturnGamesPlayed}=\frac{ReturnGamesWon}{ReturnGamesPlayed}\end{equation}" title="\begin{equation} 5) \large \frac{GamesWon-ServiceGamesWon}{ReturnGamesPlayed}=\frac{ReturnGamesWon}{ReturnGamesPlayed}\end{equation}" />

We get the second principal component for the other group of indices regarding Return: this one has an Explained Variance of 50\%. The return index had a range of [-6,10]. We then computed the EMA values for every player in the same way that we did for service rating PCA's. The missing values were also treated in the same way. 

## Head to Head Balance
Head to Head Balance is a variable that provides the history of head-to-head matches between two players i and j. 
We defined this variable as: 
<img src="https://latex.codecogs.com/svg.image?\begin{equation}\large&space;HTH_i_j=&space;GamesWon_i-GamesWon_j&space;\end{equation}" title="\begin{equation}\large HTH_i_j= GamesWon_i-GamesWon_j \end{equation}" />


## Performance
Another important feature is the Performance Lij that considers the following variables:
s for Sets, g for Games, x for points,  Delta Elo_i_j=Elo_i-Elo_j, \
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\large&space;L_i_j=(s_i-s_j)&plus;&space;\frac{1}{6}(g_i-g_j)&plus;\frac{1}{24}(x_i-x_j)&space;\end{equation}" title="\begin{equation} \large L_i_j=(s_i-s_j)+ \frac{1}{6}(g_i-g_j)+\frac{1}{24}(x_i-x_j) \end{equation}" /> 
Lij e Delta Elo_i_j are standardized on a scale [-1,1].
Performance is then calculated with a 'checkboard' classification system [fig.1]. The idea behind this is that we want to reward players who achieve a good result (represented by L_{ij}) against players that have a much greater ELO rank. On the other hand we want to penalize players who lose games against opponents that have a much lower ELO rank. The performance values are then taken and Averaged for every player with the Exponential Moving Average method. The result is a dynamic index that tries to model player performance on past games based on match results.\
![4](https://user-images.githubusercontent.com/87206202/154717887-25d1e7c2-5bcf-41f2-976c-d71367b1575b.PNG)

## Mistakes on service
This feature is built on the 2 following statistics:
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;1)&space;\large&space;\frac{DoubleFaults}{SecondServiceAttempts}\end{equation}\begin{equation}&space;&space;2)&space;\large&space;\frac{FirstServeErrors}{FirstServeAttemps}\end{equation}" title="\begin{equation} 1) \large \frac{DoubleFaults}{SecondServiceAttempts}\end{equation}\begin{equation} 2) \large \frac{FirstServeErrors}{FirstServeAttemps}\end{equation}" />
PCA (one-dimensional) is applied to these two statistics and finally as for service and return rating an EMA is computed on top of the values of the Principal Component. 
## Age Difference
This feature represents the difference between the date of birth of any given player and the date of the matches played.
# Features Distributions and visualizations
In the end of the feature extraction process we finally get 8 trainable features in the logistic model.\
![5](https://user-images.githubusercontent.com/87206202/154717881-24d31359-6673-45ec-bf63-998dd0fefca4.PNG)
\
For sake of model performance the samples are scaled using Standardization technique. For this purpose a Scaler is used: the scaler is fitted on the training portion of the set and then used for the remainder or test portion of the dataset. 
Fig.2 shows the density plots for our features, we can see that all of the features except Head to Head Balance have a normal density.
![6](https://user-images.githubusercontent.com/87206202/154717870-7a84af00-ec1a-4114-9293-8306611c3250.PNG)

#  Model and Training
Our final dataset had a total of 125938 samples, these samples were split into training and test using a 70/30 ratio. 
Training was done using Sklearn, specifically SGDClassifier function. Hyper-parameter optimization was computed inside training. A very important hyper-parameter to optimize was alpha
which is the constant that multiplies the regularization term. L2 regularization was used and alpha was selected from [0.1,0.01,0.001,0.0001,0.00001,0.000001] using random grid search method with K-fold cross validation with 30 folds.
The model was then fitted and the results were that the optimal alpha was 0.01.
Learning rate was set to 'optimal' which uses
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;\large&space;\eta&space;=&space;\frac{1}{\alpha*(t&plus;t_0)}\end{equation}" title="\begin{equation} \large \eta = \frac{1}{\alpha*(t+t_0)}\end{equation}" />
where t_0 is obtained using the Leon Bottou method.

# Model
We use the logistic regression that is a classification algorithm because it is attractive in the context of tennis prediction for its speed of training, resistance to overfitting, and for directly returning a match-winning probability. However, without additional modification, it cannot model complex relationships between the input features.
The logistic function sigma(t) is defined as:
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\large&space;&space;\sigma(t)=&space;\frac{1}{1&plus;e^-^t}\end{equation}&space;" title="\begin{equation} \large \sigma(t)= \frac{1}{1+e^-^t}\end{equation} " />
and it maps real-valued inputs between −∞ and +∞ to
values between 0 and 1, allowing for its output to be interpreted as a probability.
To make a prediction using the model, we first project a point in our n-dimensional feature space to a real number:
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\Large&space;z=\beta_0&plus;\beta_1x_1&plus;...&plus;\beta_nx_n\end{equation}&space;" title="\begin{equation} \Large z=\beta_0+\beta_1x_1+...+\beta_nx_n\end{equation} " />
where x = [x_1,...,x_n] is the vector of n match features and beta = [\beta_0,...,\beta_n] is the vector of n+1 real-valued model parameters. 
Using the logistic function we can map z to a value in the acceptable range of probability (0 to 1):
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\large&space;p=\sigma(z)=\frac{1}{1&plus;e^-^z}&space;\end{equation}" title="\begin{equation} \large p=\sigma(z)=\frac{1}{1+e^-^z} \end{equation}" />
The training of the model consists of optimising the parameters $\beta$ that is done by minimising the logistic loss, which gives a measure of the error of the model in predicting outcomes of matches used for training.
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\large&space;L(p)=-\frac{1}{N}\sum_{i=1}^{N}p_i&space;log(y_i)&plus;(1-p_i)log(1-y_i)&space;\end{equation}" title="\begin{equation} \large L(p)=-\frac{1}{N}\sum_{i=1}^{N}p_i log(y_i)+(1-p_i)log(1-y_i) \end{equation}" />
Where: 
    - N = number of training matches
    - pi = predicted probability of a win for match i
    - yi = actual outcome of match i (0 for loss, 1 for win) 
# Stochastic Gradient Descent 
The purpose of Stochastic Gradient descent is to iteratively find a minimum for the log loss function. in our case we add L2 regularization to prevent overfitting. The SGD algorithm combined with a log loss function provides a logistical prediction model as an output. For every iteration the gradient of the loss function is computed on a sample of the data and then the weights are updated accordingly. The method can be summarized by the following equations :

<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;\large&space;J^{LOG}(w)&space;=&space;-\frac{1}{n}\sum_{i}^{}\log&space;p(y^{(i)}|x^{(i)},w)\end{equation}" title="\begin{equation} \large J^{LOG}(w) = -\frac{1}{n}\sum_{i}^{}\log p(y^{(i)}|x^{(i)},w)\end{equation}" />
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;\large&space;\nabla&space;J^{LOG}(w)&space;=&space;\frac{1}{n}\sum_{i}^{}(y^{(i)}-\sigma(w*x^{(i)}))x^{(i)}\end{equation}" title="\begin{equation} \large \nabla J^{LOG}(w) = \frac{1}{n}\sum_{i}^{}(y^{(i)}-\sigma(w*x^{(i)}))x^{(i)}\end{equation}" />
<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;&space;&space;&space;\large&space;w_{t&plus;1}&space;=&space;w_t&space;&plus;&space;\eta&space;*\frac{1}{n}\sum_{i}^{}(y^{(i)}-\sigma(w*x^{(i)}))x^{(i)}\end{equation}" title="\begin{equation} \large w_{t+1} = w_t + \eta *\frac{1}{n}\sum_{i}^{}(y^{(i)}-\sigma(w*x^{(i)}))x^{(i)}\end{equation}" />
The first two equations describe the log loss and the gradient of the log loss. 
the last one explains how the weights are updated for every iteration, the $\eta$ used is defined at the start of the section. 
convergence is achieved when a tolerance constant is satisfied: J(w*) <= epsilon
# Results
The obtained results were quite satisfactory, our model achieved a mean level of accuracy of 66 percent. This is below the models we took inspiration from [cit]. Table.3 Provides information on the classification report, we can see that Metrics are all in the range o 0.65 - 0.67. We plotted a Precision Recall curve which can be seen in figure.3, it shows the relationship between precision and recall: precision is the ability of the classifier not to label as positive a sample that is negative while recall is the ability of a classifier to find all the positive samples. The graph shows the trade off between precision and recall, a high area under the curve represents high precision and high recall. We can see that the average precision is around 0.71. The Curve is quite balanced as it cuts the plane in a diagonal manner, this means that there is a balanced relationship between precision and recall in our model.

Learning Curves can be seen in fig.5 and fig.6, the first one was plotted in regards to accuracy while the second one with regards to log loss. We can see that accuracy increases with train size and that log loss decreases with train size. The range of accuracy and log loss is relatively short, this is because our model had a very fast convergence with Stochastic Gradient Ascent.

Receiver operating characteristic curve was plotted and is shown in fig.7. This curve explains the relationship between True positive rate and False positive rate, this can be interpreted as the performance of a binary classifier. The curve is plotted by evaluating TPR and FPR as the discrimination treshold of the binary classifier is varied. We can see that the Area Under Curve is 0.72, this number summarizes the informations provided in the plot. An AUC value of 0.72 is a good result considering that a perfect classifier achieves a value of 1. The AUC value is the probability that the model ranks a random positive higher than a random negative. 

![7](https://user-images.githubusercontent.com/87206202/154717865-29bf9175-cd3f-488e-ae62-197032e676d1.PNG)
![8](https://user-images.githubusercontent.com/87206202/154717857-db8b433a-0c24-4320-9d52-19c8c0e2dd01.PNG)
![9](https://user-images.githubusercontent.com/87206202/154717854-82d9f699-9502-4c62-b2e2-1b024dd18f05.PNG)
![10](https://user-images.githubusercontent.com/87206202/154717850-8ed17e7e-970c-4547-8480-44a626648cf0.PNG)
![11](https://user-images.githubusercontent.com/87206202/154717845-6fd181d3-b329-4efc-9e19-203e65b38c21.PNG)
![12](https://user-images.githubusercontent.com/87206202/154717841-e877f066-7ef6-4285-9c2c-f6cfa729aefa.PNG)

# Interpretability
For Interpretability we used two methods: the first is very straightforward and consists on looking at the odds ratios, while the second is based on SHAP values. Odds ratios are a way of interpreting the weights of the features in a logistic model. They are defined as the exp(b_j) where b_j are the features weights. In our case the features are numerical, this means that the OR's represent the change of odds that one would observe by increasing the value of a feature by one unit. This change is of a factor of exp(b_j).

In figure 8. we can see the odds ratios plotted, the following considerations are made :
    -  Service Mistakes and Performance have an OR that is less than 1, so this means that increasing the Mistakes or Performance by one unit provides a negative change in the odds.
    - The rest of the Features have OR's greater than 1, so increasing them by one unit provides a positive change in estimated odds. In particular we can see that ELO and ELO by surface have the highest Odds Ratios, this means that an increase in ELO ranking impacts the output greatly. 


Confidence intervals and p-values were not computed for the Odds Ratios.

SHAP values summary plot in fig.9 shows the impact on the model by the features. This summary combines features importance with feature effects. Each point on the summary plot is a Shapley value for a feature and an instance. The colors indicate a positive or negative feature value. We can see that the highest impact is provided by ELO rankings. Secondly the Service and Performance component, while the Age and Return Component don't seem to have a meaningful impact on the model. The Shap Summary plot is consistent with the Odds Ratios.

![13](https://user-images.githubusercontent.com/87206202/154717838-94f10822-a56e-4d16-9e02-3a5d4b4eefc8.PNG)
![14](https://user-images.githubusercontent.com/87206202/154717828-407d72c3-beb4-4076-9acb-0e7cbb7fdec3.PNG)

# Final Considerations and Further Work
This Project was a good opportunity to explore the world of machine learning applied to sports betting. Only the logistical model was used because it is regarded as the best approach for tennis predictions in the papers that we selected. Our approach was oriented towards creating a simple log regression model so that it could be understandable. Further improvement could consist in better data, and the adding of other features in order to provide the model with more information. This Project also shows how Machine Learning applied to tennis is becoming more approachable over time and that it presents itself as a great challenge. The model was built using Python with the following libraries: Pandas,Sklearn,Numpy,Pyplot,Seaborn,Shap.
## Further Work
Improvements can be made on this model, one of them could be to implement a non-linear approach to dimensionality reduction by using an autoencoder.
By using an AutoEncoder in place of PCA dimensionality reduction for the service and return statistics we would  get reduced features that could model the instrinsic
non linearity of tennis statistics.

# References
[1]  M.Sipko - Machine Learning for the Prediction of
Professional Tennis Matches \newline
http://www.doc.ic.ac.uk/teaching/distinguished-projects/2015/m.sipko.pdf  

[2] Producing Win Probabilities
for Professional Tennis
Matches from any Score

http://nrs.harvard.edu/urn-3:HUL.InstRepos:41024787

[3]
https://www.kaggle.com/ehallmar/a-large-tennis-dataset-for-atp-and-itf-betting?select=all_matches.csv
