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
print(sp500)