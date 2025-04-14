#🧩 1. Data Collection
import yfinance as yf  #Pulls the S&P 500 index from Yahoo Finance (1927-2025)
import matplotlib.pyplot as plt 
#you have to pipinstall these



sp500 = yf.Ticker("^GSPC")

#will query all data from the beginning when the index was created
sp500 = sp500.history(period="max") 
# print(sp500)


#this would print out every single date (1927 - 2025) that the stock market was open.
# print(sp500.index)

#📈 2. Data Visualization
#Plotting the Closing price over time
sp500.plot.line(y="Close", use_index=True)
plt.show()

# 🧹 3. Data Cleaning
#delete extra columns
del sp500['Dividends']
del sp500['Stock Splits']

#🎯 4. Feature Engineering
#Setting up the target, creating a column named Tomorrow, taking the Closed prices and shifting it by ONE day
sp500['Tomorrow'] = sp500['Close'].shift(-1)

#Will return 1 if the price went up, 0 if it went down
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


#Remove all data before 1990, this will only select data from 1990 and after
sp500 = sp500.loc["1990-01-01":].copy()
print(sp500)

#Training the model

from sklearn.ensemble import RandomForestClassifier #trains individual decision trees with randomized parameters

#creates a Random Forest Classification model with scikit-learn
#This will build 100 individual ecision trees and require 100 samples to split a node
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

#Puts everything except the last 100 rows into the training set
train = sp500.iloc[:-100]

#Puts the last 100 rows into the test set
test = sp500.iloc[-100:]

#list all of all the columns we will use to predict
predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])



#Imports precison score which will evaluate the model's performance
from sklearn.metrics import precision_score

#Making predictions on the test set by selecting the features we're using to predict
#model.predict will return an array of 1s and 0s
preds = model.predict(test[predictors])

#imports pandas to convert the array into a series
import pandas as pd
preds = pd.Series(preds, index=test.index)
# print(preds)

#Calculating the precision score, our model is only going to be right 51% of the time
# print(precision_score(test['Target'], preds))

#Plot the predictions
combined = pd.concat([test["Target"], preds], axis=1)
combined.columns = ['Actual', 'Predicted']  # Give meaningful column names
combined.plot()

#the orange line (zero) is our predictions, and the blue lines are what happened.
plt.show()

#Building a backtesting system
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

#  we take the first 10 years of data to predict the 11th year of data and so on.
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)

# This will predict how many days the market goes up vs down
# Will guess that the market will go down > 3000 days
# Will guess that the market will go up > 2000 days
counts = predictions["Predictions"].value_counts()
print(f"\nPredictions breakdown:")
print(f"Down days (0): {counts[0]}")
print(f"Up days (1): {counts[1]}")

print(f"Precision score: {precision_score(predictions['Target'], predictions['Predictions'])}")
