
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import math
!pip install arch

#!pip install networkx==2.5.1
#import networkx as nx

from arch import arch_model

#import statsmodels
from statsmodels.tsa.api import VAR

import warnings
warnings.filterwarnings('ignore')

ticker_symbol=[47]
data2023=[]
ticker_symbol=['ADANIENT.NS','ADANIPORTS.NS','SHRIRAMFIN.NS','COALINDIA.NS','TATASTEEL.NS','BAJFINANCE.NS','BAJAJ-AUTO.NS','APOLLOHOSP.NS','LT.NS','INDUSINDBK.NS','M&M.NS','SBIN.NS','POWERGRID.NS','ITC.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','WIPRO.NS','BPCL.NS','BAJAJFINSV.NS','RELIANCE.NS','SUNPHARMA.NS','KOTAKBANK.NS','ULTRACEMCO.NS','TATAMOTORS.NS','NTPC.NS','EICHERMOT.NS','ASIANPAINT.NS','BRITANNIA.NS','HEROMOTOCO.NS','JSWSTEEL.NS','AXISBANK.NS','ONGC.NS','BHARTIARTL.NS','HCLTECH.NS','TITAN.NS','TECHM.NS','GRASIM.NS','HINDUNILVR.NS','TCS.NS','TATACONSUM.NS','CIPLA.NS','HINDALCO.NS','MARUTI.NS','DRREDDY.NS','NESTLEIND.NS','DIVISLAB.NS']
start_date='2020-08-01'
close_date='2023-08-01'

for i in range(47):
  data2023.append(yf.download(ticker_symbol[i],start=start_date,end=close_date))

for i in range(len(ticker_symbol)):
  for j in range(len(data2023[i])):
    data2023[i]=data2023[i].assign(returns=((np.log(data2023[i]['Close']))-(np.log(data2023[i]['Close'].shift(1)))))

for i in range(len(ticker_symbol)):
  data2023[i]['returns'][0]=0
  model=arch_model(data2023[i]['returns'],p=1,q=1)
  model_fit=model.fit(disp='off')
  data2023[i]=data2023[i].assign(conditional_volatality=(np.power(model_fit.conditional_volatility,3/2)))

input=[np.array(data2023[i]['returns']) for i in range(len(ticker_symbol))]
print(len(input))
corr_matrix = np.corrcoef(input)
print(corr_matrix)

eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
#print(eigenvalues)
#Find the principal eigenvector
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]
print('largest index:',largest_index,'\nprincipal eigenvector:',principal_eigenvector)

def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices2023 = get_sorted_indices(eve.copy())

for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices2023[i]])

df = pd.concat([data2023[i]['returns'] for i in range(len(ticker_symbol))],axis=1)
returns=pd.DataFrame(df)
#print(returns)
model = VAR(returns)
var_result = model.fit(1)  # Fit model with lag 1

# Print summary of the results
print(var_result.summary())

# Get the covariance matrix of residuals
covariance_matrix = var_result.sigma_u

# Perform Cholesky decomposition
chol_matrix = np.linalg.cholesky(covariance_matrix)

# Print Cholesky matrix
print("Cholesky Matrix:\n", chol_matrix)

# Calculate orthogonalized impulse responses
# For demonstration, compute impulse responses for 10 periods

irf = var_result.irf(10)
orth_irf = irf.orth_irfs
print("Orthogonalized Impulse Responses:\n", orth_irf)

!pip install networkx matplotlib pygraphviz
!apt-get -qq install -y graphviz libgraphviz-dev pkg-config
import networkx as nx
import matplotlib matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

def create_mst_from_correlation(correlation_matrix):
  """
  Creates a minimum spanning tree (MST) from a correlation matrix.

  Args:
    correlation_matrix: A 2D numpy array representing the correlation matrix.

  Returns:
    A networkx minimum spanning tree object.
  """

  # Handle potential errors in correlation matrix
  if correlation_matrix is None or not isinstance(correlation_matrix, np.ndarray):
    raise ValueError("Invalid correlation matrix provided.")

  # Invert the correlation matrix to prioritize stronger negative correlations for edge weights
  edge_weights = 1 - np.abs(correlation_matrix)

  # Create a complete graph from the edge weights
  graph = nx.from_numpy_matrix(edge_weights, create_using=nx.complete_graph(correlation_matrix.shape[0]))

  # Calculate the minimum spanning tree of the graph
  mst = nx.minimum_spanning_tree(graph)

  return mst

# Example usage (replace with your actual correlation matrix)
# corr_matrix = np.array([[1, 0.5, 0.2], [0.5, 1, 0.7], [0.2, 0.7, 1]])
mst = create_mst_from_correlation(corr_matrix)

# Print information about the MST
print("Number of nodes in MST:", mst.number_of_nodes())
print("Number of edges in MST:", mst.number_of_edges())

# Print edges in the MST
for edge in mst.edges():
    print(edge)

# Visualize the MST
# If you are in a non-interactive environment, uncomment the following line:
plt.switch_backend('agg')
nx.draw(mst, with_labels=True)
plt.show()

!pip install slicematrixIO
from slicematrixIO import SliceMatrix as sm

# Assuming 'returns' is your pandas DataFrame
mst = sm.MinimumSpanningTree(dataset =returns)
results = mst.fit()
print(results.summary())



!pip show networkx
!pip.show matplotlib
!jupyter --version

input=[np.array(data2023[i]['conditional_volatality']) for i in range(len(ticker_symbol))]
print(len(input))
corr_matrix = np.corrcoef(input)
print(corr_matrix)

eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
#print(eigenvalues)
#Find the principal eigenvector
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]
#print('largest index:',largest_index,'\nprincipal eigenvector:',principal_eigenvector)

def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices_v2023 = get_sorted_indices(eve.copy())

for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices_v2023[i]])

ticker_symbol=[47]
data=[]
ticker_symbol=['ADANIENT.NS','ADANIPORTS.NS','SHRIRAMFIN.NS','COALINDIA.NS','TATASTEEL.NS','BAJFINANCE.NS','BAJAJ-AUTO.NS','APOLLOHOSP.NS','LT.NS','INDUSINDBK.NS','M&M.NS','SBIN.NS','POWERGRID.NS','ITC.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','WIPRO.NS','BPCL.NS','BAJAJFINSV.NS','RELIANCE.NS','SUNPHARMA.NS','KOTAKBANK.NS','ULTRACEMCO.NS','TATAMOTORS.NS','NTPC.NS','EICHERMOT.NS','ASIANPAINT.NS','BRITANNIA.NS','HEROMOTOCO.NS','JSWSTEEL.NS','AXISBANK.NS','ONGC.NS','BHARTIARTL.NS','HCLTECH.NS','TITAN.NS','TECHM.NS','GRASIM.NS','HINDUNILVR.NS','TCS.NS','TATACONSUM.NS','CIPLA.NS','HINDALCO.NS','MARUTI.NS','DRREDDY.NS','NESTLEIND.NS','DIVISLAB.NS']
start_date='2011-08-01'
close_date='2014-07-31'

for i in range(47):
  data.append(yf.download(ticker_symbol[i],start=start_date,end=close_date))
  #print(len(data[i]))

for i in range(len(ticker_symbol)):
  for j in range(len(data[i])):
    data[i]=data[i].assign(returns=((np.log(data[i]['Close']))-(np.log(data[i]['Close'].shift(1)))))

for i in range(len(ticker_symbol)):
  data[i]['returns'][0]=0
  model=arch_model(data[i]['returns'],p=1,q=1)
  model_fit=model.fit(disp='off')
  data[i]=data[i].assign(conditional_volatality=(np.power(model_fit.conditional_volatility,3/2)))

input=[np.array(data[i]['returns']) for i in range(len(ticker_symbol))]
corr_matrix = np.corrcoef(input)


eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]


def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices1114 = get_sorted_indices(eve.copy())
print('Based on returns the order by eigenvectors is')
for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices1114[i]])




input=[np.array(data[i]['conditional_volatality']) for i in range(len(ticker_symbol))]
corr_matrix = np.corrcoef(input)

eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]


def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices_v1114 = get_sorted_indices(eve.copy())
print('Based on volatility the order by eigenvectors is')
for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices_v1114[i]])

ticker_symbol=[47]
data=[]
ticker_symbol=['ADANIENT.NS','ADANIPORTS.NS','SHRIRAMFIN.NS','COALINDIA.NS','TATASTEEL.NS','BAJFINANCE.NS','BAJAJ-AUTO.NS','APOLLOHOSP.NS','LT.NS','INDUSINDBK.NS','M&M.NS','SBIN.NS','POWERGRID.NS','ITC.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','WIPRO.NS','BPCL.NS','BAJAJFINSV.NS','RELIANCE.NS','SUNPHARMA.NS','KOTAKBANK.NS','ULTRACEMCO.NS','TATAMOTORS.NS','NTPC.NS','EICHERMOT.NS','ASIANPAINT.NS','BRITANNIA.NS','HEROMOTOCO.NS','JSWSTEEL.NS','AXISBANK.NS','ONGC.NS','BHARTIARTL.NS','HCLTECH.NS','TITAN.NS','TECHM.NS','GRASIM.NS','HINDUNILVR.NS','TCS.NS','TATACONSUM.NS','CIPLA.NS','HINDALCO.NS','MARUTI.NS','DRREDDY.NS','NESTLEIND.NS','DIVISLAB.NS']
start_date='2014-08-01'
close_date='2017-07-31'

for i in range(47):
  data.append(yf.download(ticker_symbol[i],start=start_date,end=close_date))
  #print(len(data[i]))

for i in range(len(ticker_symbol)):
  for j in range(len(data[i])):
    data[i]=data[i].assign(returns=((np.log(data[i]['Close']))-(np.log(data[i]['Close'].shift(1)))))

for i in range(len(ticker_symbol)):
  data[i]['returns'][0]=0
  model=arch_model(data[i]['returns'],p=1,q=1)
  model_fit=model.fit(disp='off')
  data[i]=data[i].assign(conditional_volatality=(np.power(model_fit.conditional_volatility,3/2)))

input=[np.array(data[i]['returns']) for i in range(len(ticker_symbol))]
corr_matrix = np.corrcoef(input)


eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]


def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices1417 = get_sorted_indices(eve.copy())
print('based on return the order by eigenvector is')
for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices1417[i]])



input=[np.array(data[i]['conditional_volatality']) for i in range(len(ticker_symbol))]
corr_matrix = np.corrcoef(input)

eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]


def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices_v1417 = get_sorted_indices(eve.copy())
print('Based on volatility the order by eigenvectors is')
for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices_v1417[i]])

ticker_symbol=[47]
data=[]
ticker_symbol=['ADANIENT.NS','ADANIPORTS.NS','SHRIRAMFIN.NS','COALINDIA.NS','TATASTEEL.NS','BAJFINANCE.NS','BAJAJ-AUTO.NS','APOLLOHOSP.NS','LT.NS','INDUSINDBK.NS','M&M.NS','SBIN.NS','POWERGRID.NS','ITC.NS','HDFCBANK.NS','ICICIBANK.NS','INFY.NS','WIPRO.NS','BPCL.NS','BAJAJFINSV.NS','RELIANCE.NS','SUNPHARMA.NS','KOTAKBANK.NS','ULTRACEMCO.NS','TATAMOTORS.NS','NTPC.NS','EICHERMOT.NS','ASIANPAINT.NS','BRITANNIA.NS','HEROMOTOCO.NS','JSWSTEEL.NS','AXISBANK.NS','ONGC.NS','BHARTIARTL.NS','HCLTECH.NS','TITAN.NS','TECHM.NS','GRASIM.NS','HINDUNILVR.NS','TCS.NS','TATACONSUM.NS','CIPLA.NS','HINDALCO.NS','MARUTI.NS','DRREDDY.NS','NESTLEIND.NS','DIVISLAB.NS']
start_date='2017-08-01'
close_date='2020-07-31'

for i in range(47):
  data.append(yf.download(ticker_symbol[i],start=start_date,end=close_date))
  #print(len(data[i]))

for i in range(len(ticker_symbol)):
  for j in range(len(data[i])):
    data[i]=data[i].assign(returns=((np.log(data[i]['Close']))-(np.log(data[i]['Close'].shift(1)))))

for i in range(len(ticker_symbol)):
  data[i]['returns'][0]=0
  model=arch_model(data[i]['returns'],p=1,q=1)
  model_fit=model.fit(disp='off')
  data[i]=data[i].assign(conditional_volatality=(np.power(model_fit.conditional_volatility,3/2)))

input=[np.array(data[i]['returns']) for i in range(len(ticker_symbol))]
corr_matrix = np.corrcoef(input)


eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]


def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices1720 = get_sorted_indices(eve.copy())

for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices1720[i]])



input=[np.array(data[i]['conditional_volatality']) for i in range(len(ticker_symbol))]
corr_matrix = np.corrcoef(input)

eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
largest_index = np.argmax(eigenvalues)
principal_eigenvector = eigenvectors[:, largest_index]


def get_sorted_indices(eve):
  sorted_data = sorted(eve, reverse=True)
  indices = [np.where(eve == val)[0][0]for val in sorted_data]
  return sorted_data, indices

eve = eigenvalues
sorted_list, initial_indices_v1720 = get_sorted_indices(eve.copy())
print('Based on volatility the order by eigenvectors is')
for i in range(len(sorted_list)):
  print(ticker_symbol[initial_indices_v1720[i]])

array=[initial_indices_v1720,initial_indices_v1417,initial_indices_v1114,initial_indices_v2023]
corr=np.corrcoef(array)
print(corr)

array=[initial_indices1720,initial_indices1417,initial_indices1114,initial_indices2023]
corr=np.corrcoef(array)
print(corr)
