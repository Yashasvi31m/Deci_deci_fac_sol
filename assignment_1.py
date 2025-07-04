

import pandas as pd

data=pd.read_csv("data.csv")

data

import yfinance as yf

aapl=yf.download("AAPL",'2018-01-01','2022-01-01')

aapl

mean_vol=sum(aapl['Volume'])/len(aapl)

mean_vol

signals=[]
for i in range(len(aapl)):
    if aapl['Volume'][i] > mean_vol:
        if aapl['Open'][i] > aapl['Close'][i]:
            # Sell
            signals.append(-1)
        else:
            # Buy
            signals.append(1)
    else:
        signals.append(0)

aapl['signals']=signals

aapl

aapl[aapl['signals']!=0]

capital=[5000]*len(aapl)
aapl["Capital"]=capital
noofshare=0

for i in range (0,len(aapl)):
  if (aapl['signals'][i]==0):
    aapl["Capital"][i]=aapl["Capital"][i-1]
  elif (aapl['signals'][i]== 1):
    # Buying Share
    aapl["Capital"][i]=aapl['Capital'][i-1]-aapl["Close"][i]
    noofshare+=1
  elif (aapl['signals'][i]== -1):
    # Selling share
    aapl["Capital"][i]=aapl['Capital'][i-1]+aapl["Open"][i]
    noofshare-=1

# Squaring off
if noofshare<0:
  aapl["Capital"][-1]=aapl["Capital"][-2]+(noofshare)*aapl["Close"][-1]
else:
  aapl["Capital"][-1]=aapl["Capital"][-2]+(noofshare)*aapl["Open"][-1]

aapl

f=aapl["Capital"][-1]
f

returns=(100*(f-5000))/5000
returns

st=aapl['Capital'].std()
st

rfr=7
sharpe=(((returns)-rfr)/st)*(252**(1/2))
sharpe

dr=[0]*(len(aapl))
aapl["Daily_Return"]=dr
for i in range (1,len(aapl)):
  aapl["Daily_Return"][i]=aapl["Capital"][i]-aapl["Capital"][i-1]
aapl["Daily_Return"][0]=0
aapl

aapl[aapl["Daily_Return"]<0]

so_sd=aapl[aapl["Daily_Return"]<0]["Daily_Return"].std()
so_sd

sortino=((returns-rfr)/so_sd)*(252**(1/2))
sortino
