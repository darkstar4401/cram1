
#
# ==================================================
# Computational Risk and Asset Management 1 WS 17/18
# Problem Set 1, Week 1: TEMPLATE
# Portfolio Allocation
# ==================================================
#
# Prepared by Elmar Jakobs
#



#
# !!! HINT: You can find most of the necessary Python commands in the provided real-world example !!!
#



# PORTFOLIO ALLOCATION
# =====================


# Setup
# -----
#
# Import packages for econometric analysis
#

import numpy as np

import pprint

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')



# Import and configure the testing framework
#
from test_suite import Test
checker_results = pd.read_csv('results_ps1.csv').set_index('Task')

# For submission, please change the status to 'SOLN'
#
status = 'SOLN'




# Question 1a: Data Management
# ----------------------------


# Note, historical stock and bond index data are from YAHOO Finance:
#      https://finance.yahoo.com/quote/VBTIX?ltr=1
#      https://finance.yahoo.com/quote/^GSPC/?p=^GSPC
#


#  Hint: Use the 'parse_dates' attribute
#
sp500 = pd.read_csv('sp500_5y_monthly.csv', parse_dates = True)
sp500.head()


#
del sp500['Open']
del sp500['High']
del sp500['Low']
del sp500['Close']
del sp500['Volume']


#
sp500 = sp500.set_index('Date')
sp500.tail()

#
vbtix =  pd.read_csv('vbtix_5y_monthly.csv', parse_dates = True)      # read-in file
vbtix =  vbtix.set_index('Date')      # set index
vbtix.head()


#  Combine both pandas data frames using the '.join' function of the data frames
#  Hint #1: To obtain further information about the '.join' function use '?pd.DataFrame.join'
#  Hint #2: You can directly join the sp500 data frame with 'Adj Close' column of the vbtix data frame (.join(vbtix['Adj Close']) )
#  Hint #3: Since both indices have the same column name use the attribute
#       'rsuffix' = '_vbtix' to rename the VBTIX series
#
indices = sp500.join(vbtix['Adj Close'], rsuffix='_vbtix')
indices = indices.rename( columns = {'Adj Close': 'Adj Close_sp500'} )
indices.head()


#  Sort the index values in ascending order
#  Hint: To find the right function you can use
#       google with 'pandas data frame sort index'
#
indices = indices.sort_index()
indices[0:3]


#  Plot the two indices
#  Hint: Use the DataFrame.plot() functionality, set 'subplots' to True
#
indices.plot(subplots=True, style='b', figsize=(15,12))


# Check of intermediate result (1):
#
Test.assertEquals(np.round(indices.iloc[0][0],2), 1131.42, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(str(np.round(indices.iloc[0][1],2))[0:4], str(checker_results.loc[1][0]), 'incorrect result')




# Question 1b: Asset Returns
# --------------------------


# TD: Calculate the log returns for the two index price series
#  Hint: See the Python real-world financial data example for the calculation
#

indices['sp500_returns'] = np.log(sp500['Adj Close'] / sp500['Adj Close'].shift(-1)) * 100
indices['vbtix_returns'] = np.log(vbtix['Adj Close'] / vbtix['Adj Close'].shift(-1)) * 100


# TD: Plot the return time series
#
indices[['sp500_returns', 'vbtix_returns']].plot(subplots=True, style='b', figsize=(15,12))


# TD: Calculate the annualized mean and standard deviation of the return series
#  Hint: Use the .mean() function
#  Hint #2: To annualize the mean (volatility) multiply the monthly value with 12 ( sqrt(12) )
#


# First, S&P 500
#

# Annualized mean return of S&P500
sp500_mean_ann = indices['sp500_returns'].mean() * 12
print(np.round(sp500_mean_ann, 2))


# Annualized volatility of the S&P500 returns
sp500_vol_ann = indices['sp500_returns'].std() * np.sqrt(12)
print(np.round(sp500_vol_ann, 2))



# Next, VBTIX
#

# Annualized mean return of S&VBTIX
vbtix_mean_ann = indices['vbtix_returns'].mean() * 12
print(np.round(vbtix_mean_ann, 2))


# Annualized volatility of the S&VBTIX returns
vbtix_vol_ann = indices['vbtix_returns'].std() * np.sqrt(12)
print(np.round(vbtix_vol_ann, 2))



# TD: Calculate the correlation between the two return series
#  Hint: Use the '.corr()' function for data frames
#
sp500_vbtix_corr = indices['vbtix_returns'].corr(indices['sp500_returns'])
print(np.round(sp500_vbtix_corr, 2))



# Check of intermediate result (2):
#
Test.assertEquals(np.round(sp500_mean_ann,2), 12.8, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(vbtix_vol_ann,2), checker_results.loc[2][0], 'incorrect result')




# Question 1c: Investment Opportunity Set
# ---------------------------------------


# DONE: Define the risk-free rate
#
rf = 0.005 * 100

#
# DONE: Define the expected returns
#  Hint: With 'np.matrix' you can create a matrix data structure.
#  Hint #2: Using '?np.matrix' you obtain further information on how to create it
#  Hint #3: Inside the 'np.matrix( ... )' command use the double square brackets '[[sp500_mean_ann], [vbtix_mean_ann]]'
#
mu = np.matrix([[sp500_mean_ann], [vbtix_mean_ann]])
mu


# With '.shape' you can check the dimensions of your matrix
#  Hint: A shape of (2, 1) facilitates the further analysis
#
mu.shape


# DONE: Define the covariance matrix
#  Hint: With 'np.matrix' you can create a matrix data structure.
#  Hint #2: You can calculate the covariance of X and Y as:
#           cov_xy = sigma_x * sigma_y * corr_xy
#

# annualize corr?
Cov = np.matrix([[sp500_vol_ann**2, sp500_vol_ann * vbtix_vol_ann * sp500_vbtix_corr],
                 [sp500_vol_ann * vbtix_vol_ann * sp500_vbtix_corr, vbtix_vol_ann**2]
                 ])


# Check of intermediate result (3):
#
Test.assertEquals(np.round(Cov[0,0],2), 121.36, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(Cov[0,1],2), checker_results.loc[3][0], 'incorrect result')






# Question 1d: Minimum Variance Portfolio
# ---------------------------------------


# Identity matrix
# Hint: A shape of (2, 1) facilitates the further analysis
#
dim = np.size(mu)
I   = np.matrix( np.ones(dim) ).getT()
I.shape


# DONE: Calculate the weights of the global minimum variance portfolio
#  Hint: To get the inverse (transpose) of a matrix use the '.getI()' ('.getT()') functions
#  Hint #2: See the probelm set slides or the lecture notes for the closed form solution of the weights
#
w_mvp = (Cov.getI() * I) / (I.getT() * Cov.getI() * I)


# DONE: Calculate the expected mean of the minimum variance portfolio
#  Hint: The portfolio return is defined as mu^T * w_p
#
mu_mvp = mu.getT() * w_mvp


# DONE: Calculate the expected volatility of the minimum variance portfolio
#  Hint: The variance of a portfolio is defined as w_p^T * Sigma * w_p
#
# Sigma = Cov?
#sigma_mvp = w_mvp.getT() * Cov * w_mvp
sigma_mvp = np.sqrt(1/(I.getT() * Cov.getI() * I))




# Check of intermediate result (4):
#
Test.assertEquals(np.round(w_mvp[0],2), 0.07, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(sigma_mvp,2), checker_results.loc[4][0], 'incorrect result')




# Question 1e: Efficient Frontier
# -------------------------------


# DONE: Calculate the unconstrained efficient frontier (analytical solution)
#  Hint: The formulae are given in the lecture notes and in the problem set
#  Hint #2: When calculating g you should convert D, X, Y and Z with .item() to a scalar.
#
#
X = mu.getT() * Cov.getI() * mu
Y = mu.getT() * Cov.getI() * I
Z = I.getT() * Cov.getI() * I
D = X * Z - Y**2
g = ( 1/D.item() ) * (X.item() * Cov.getI() * I - Y.item() * Cov.getI() * mu)
h = ( 1/D.item() ) *  (Z.item() * Cov.getI() *  mu - Y.item() * Cov.getI() * I)



# Set the number of efficient portfolios
#
n = 500


# Define the minimum and maximum return of your efficient frontier
#
mu_min = 0
mu_max = 2 * max(mu)
incr = (mu_max - mu_min) / (n-1)


# Calculate the efficient portfolios along the frontier by incrementing the return
#
w_ef = np.empty( (n, dim) )
sigma_ef = np.empty( (n, 1) )
mu_ef = np.empty( (n, 1) )


for i in range (n):
    mu_i        = i * incr
    w_i         = g + h * mu_i
    w_ef[i,:]   = w_i.getT()
    mu_ef[i]    = mu.getT() * w_i
    sigma_ef[i] = np.sqrt(w_i.getT() * Cov * w_i)




# Check of intermediate result (5):
#
Test.assertEquals(np.round(w_ef[0][0],2), -0.30, 'incorrect result')
Test.assertEquals(np.round(mu_ef[0],2), -0.0, 'incorrect result')
Test.assertEquals(np.round(mu_ef[1],2), 0.05, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(sigma_ef[0],2), checker_results.loc[5][0], 'incorrect result')
    Test.assertEquals(np.round(sigma_ef[5],2), checker_results.loc[6][0], 'incorrect result')
    Test.assertEquals(np.round(sigma_ef[7],2), checker_results.loc[7][0], 'incorrect result')





# Question 1f: Tangency PF
# ------------------------


# DONE: Find the tangency portfolio among the ones in the efficient frontier
#  Hint: use the '.argmax()' function to find the maximum value in an array
#
sharpe = (mu_ef - rf) / sigma_ef           # Calculate the sharpe ratio
ind = sharpe.argmax()            # Find the position of the max sharpe ratio

w_tp     = w_ef[ind,]
mu_tp    = mu_ef[ind]
sigma_tp = sigma_ef[ind]


# Check of intermediate result (6):
#
Test.assertEquals(np.round(sharpe[0],2), -0.1, 'incorrect result')
Test.assertEquals(np.round(sharpe[1],2), -0.09, 'incorrect result')
Test.assertEquals(np.round(ind,2), 103, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(w_tp[0],2), checker_results.loc[8][0], 'incorrect result')
    Test.assertEquals(np.round(mu_tp[0],2), checker_results.loc[9][0], 'incorrect result')
    Test.assertEquals(np.round(sigma_tp[0],2), checker_results.loc[10][0], 'incorrect result')




# Visualization
# -------------


# Print interesting statistics like weights, exp. returns and variance of the minimum variance and tangency portfolio
#
print()
print('Unconstrained Markowitz Optimization')
print('====================================')
print()
print('Minimum Variance Portfolio')
print('--------------------------')
print('Weights:         ', np.round(w_mvp.getT(), 2))
print('Exp. Return:     ', np.round(mu_mvp, 2))
print('Exp. Volatility: ', np.round(sigma_mvp, 2))
print()
print('Tangency Portfolio')
print('------------------')
print('Weights          : ', np.round(w_tp, 2))
print('Exp. Return      : ', np.round(mu_tp, 2))
print('Exp. Volatility  : ', np.round(sigma_tp, 2))
print('Exp. Sharpe Ratio: ', np.round( (mu_tp - rf)/sigma_tp, 2 ) ) # Vol or variance?
print()




# Plot the portfolios defining the efficient frontier
#
plt.figure(3, figsize = (12, 7))

# Add the risk free rate
plt.plot(0, rf, 'o', markersize = 7.5, color = 'gray')
plt.annotate('Risk-free rate', xy = (0, rf), xytext = (0-2.5, rf + 2), color = 'gray', arrowprops = dict(facecolor='black', shrink=0.05, width = 2.5))

# Add the tangency portfolio
plt.plot(sigma_tp, mu_tp, 'o', markersize = 7.5, color = 'gray')
plt.annotate('Tangency portfolio', xy = (sigma_tp, mu_tp), xytext = (sigma_tp - 3.5, mu_tp + 3), color = 'gray', arrowprops = dict(facecolor='black', shrink=0.05, width = 2.5))

# Add the mvp portfolio
plt.plot(sigma_mvp, mu_mvp, 'o', markersize = 7.5, color = 'gray')
plt.annotate('Minimum variance \n portfolio', xy = (sigma_mvp, mu_mvp), xytext = (sigma_mvp + 2.5, mu_mvp ), color = 'gray', arrowprops = dict(facecolor='black', shrink=0.05, width = 2.5))

# Add the S&P500
plt.plot(sp500_vol_ann, sp500_mean_ann, 'o', markersize = 7.5, color = 'gray')
plt.annotate('S&P 500', xy = (sp500_vol_ann, sp500_mean_ann), xytext = (sp500_vol_ann, sp500_mean_ann - 4.5 ), color = 'gray', arrowprops = dict(facecolor='black', shrink=0.05, width = 2.5))

# Add the bond fund
plt.plot(vbtix_vol_ann, vbtix_mean_ann, 'o', markersize = 7.5, color = 'gray')
plt.annotate('Bond fund', xy = (vbtix_vol_ann, vbtix_mean_ann), xytext = (vbtix_vol_ann - 2, vbtix_mean_ann - 4 ), color = 'gray', arrowprops = dict(facecolor='black', shrink=0.05, width = 2.5))

# Add the efficient frontier
plt.scatter(sigma_ef, mu_ef, c = ( (mu_ef - rf) / sigma_ef), marker = 'o')
plt.annotate('Efficient frontier', xy = (15, 16), xytext = (17.5, 12.5), color = 'gray', arrowprops = dict(facecolor='black', shrink=0.05, width = 2.5))


# Add a grid
plt.grid(True)

# Describe the axis
plt.xlabel('Exp. Volatility of PF')
plt.ylabel('Exp. Return of PF')

# Add a title
plt.title('Markowitz Portfolio Allocation (unconstrained)', fontweight = 'bold')

# Add a colorbar to highlight the sharpe ratios
plt.colorbar(label = 'Exp. Sharpe ratio')

# Add the capital market line
sharpe_tp = ( (mu_tp - rf) / sigma_tp )
cx = np.linspace(0.0, np.max(sigma_ef ))
plt.plot(cx, rf + sharpe_tp * cx, lw = 4.0, color = 'black')





print(Test.passed, "tests passed of", Test.numTests,  "(", 100*Test.passed/Test.numTests, "%)")








