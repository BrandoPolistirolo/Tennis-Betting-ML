# Tennis-Betting-ML
Machine Learning model(specifically log-regression with stochastic gradient descent) for tennis matches prediction. 
# Abstract
In this project we built a machine learning algorithm for tennis match predictions. We used a logistic model with Stochastic Gradient Descent. 
# Summary
This assignment was divided in four parts. 1) Data Processing, 2) Features engineering and extraction 3) Training and validation 4) Interpretability  

# Introduction
We have chosen to predict outcome of tennis matches because it is a sport that implies one winner for every game so there isn't tie result, there are just two possible outcomes and just two players for each match: we take advantages by the structure of scoring.\\This system in tennis has a hierarchical structure, with a match being composed of sets, which in turn are composed of games, which are composed of individual points. By assuming that points are independently and identically distributed (iid), the expressions only need the probabilities of the two players winning a point on their serve.\\Another reason to choose tennis is because the statistics are available and complete. 

# Tennis Betting 
There are two main categories of tennis betting: pre-game and in-game. We will focus only on the pre-game bets. Bets on tennis matches can be placed either with bookmakers or on betting exchanges. Traditional bookmakers set odds for the different outcomes of a match and a bettor competes against the bookmakers. In the case of betting exchanges costumers can bet against odds set by other costumers.The exchange matches the customers’ bets to earn a risk-free profit by charging a commission on each bet matched. Betting odds represents the return a bettor receives from correctly predicting the outcome of an event. Those provide an implied probability of the outcome of a match, the bookmakers' estimate of the true probability. For example for odds X/Y for a player winning a match, the implied probability p of winning is <img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;p=&space;\frac{Y}{Y&plus;X}&space;\end{equation}\\" title="\begin{equation} p= \frac{Y}{Y+X} \end{equation}\\" /> 
Given the betting odds and a predicted probability of a match outcome, different strategies will result in a different return on investment (profit or loss resulting from a bet).

# The Dataset

The Dataset was taken from Kaggle, the creator is Evan Hallmark. It is a very large dataset that contains a lot of useful data, for example betting odds of various kind. The data was scraped by the creator of the dataset from the official ATP World Tour website, since we cannot see the scraping program we do not have a clear idea of how reliable the data is. Web Scraping is often a faulty process, and is very rarely void of errors. We chose this dataset mainly because by looking around online we found that the best source of tennis statistics is without a doubt the official ATP website. We found that this dataset was the most complete among the ones that scraped the ATP website. The following table is a full account of the columns contained in the dataset.

