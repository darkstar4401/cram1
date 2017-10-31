#
# ================================================
# Computational Risk and Asset Management WS 17/18
# Problem Set 2, Week 2 TEMPLATE
# Modeling of Equity Returns
# ================================================
#
# Prepared by Elmar Jakobs
#




# Modeling of Equity Returns
# ===========================



# Setup
# -----


# Import packages for econometric analysis
#
import numpy as np
import pandas as pd

from scipy import stats



# Plotting library
#
import matplotlib
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use('TkAgg')
matplotlib.style.use('ggplot')


# Import and configure the testing framework
#
from test_suite import Test
# For submission, please change the status to 'SOLN'
status = ''
if status is 'SOLN':
  checker_results = pd.read_csv('results_ps2.csv').set_index('Task')




# Question 1a: Equity Index Returns
# ---------------------------------

# Euro Stoxx 50

# DONE: Read-in the data
#   Hint: To read-in a txt file use the pandas 'read_table' function
#   Hint #2: As a 'delimiter' use ','
#
es50 = pd.read_table('estoxx50.txt', delimiter=',')
es50['Date'] = pd.to_datetime(es50['Date'], format = '%Y-%m-%d')
es50 = es50.set_index('Date')
es50 = es50.sort_index()
es50.head()


# DONE: Calculate log returns
#
es50['LN_Returns'] = np.log(es50['Adj Close'] / es50['Adj Close'].shift(1)) * 100
print(len(es50))
es50.head()



# DONE: Plot daily prices and return time series
#
es50[['Adj Close', 'LN_Returns']].plot(subplots=True, style='b', figsize=(15,12))


# DONE: Plot histogram
#
es50[['LN_Returns']].hist(bins=20)



# Note, for the further analysis it is helpful to create a new numpy array
# for the returns
#
es50_ret = np.array(es50['LN_Returns'][1:len(es50['LN_Returns'])])
type(es50_ret)



# Check of intermediate result (1):
#
Test.assertEquals(np.round(es50_ret[0],3), 0.167, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(str(np.round(es50_ret[1],3)), str(checker_results.loc[1][0])[0:6], 'incorrect result')






# Question 1b: Jarque-Bera Test
# -----------------------------

# Function to calculate the sample skewness
#
def skewness(x):

    n = len(x)

    num = 1/n * sum( (x - np.mean(x))**3 )

    den = ( 1/n * sum( (x - np.mean(x))**2 ) )**(3/2)

    S = num / den

    return S


# Function to calculate the sample kurtosis
#
def kurtosis(x):

    n = len(x)

    num = 1/n * sum( (x - np.mean(x))**4 )

    den = ( 1/n * sum( (x - np.mean(x))**2 ) )**2

    K = num / den

    return K



# DONE: Calculate higher moments
#
es50_skew = skewness(es50_ret)
print(es50_skew)

# Double check your implementation
print(stats.skew(es50_ret))


es50_kurt = kurtosis(es50_ret)
print(es50_kurt - 3)

# Double check your implementation
print(stats.kurtosis(es50_ret))



# DONE: Implement Jarque-Bera test
#
def calc_jb_test( return_series ):


    n = len(return_series) # length of the return series

    S = skewness(return_series) # sample skewness
    K = kurtosis(return_series) # sample kurtosis

    jb = n * ( ((S**2)/6) + (((K-3)**2)/24) ) # formula is given in problem set

    jb_pv = stats.chi2.sf(jb, 2)

    return jb, jb_pv

#
# # DONE: Apply JB test to ES50 return series
# #
es50_jb_tstat = calc_jb_test(es50_ret)[0]
es50_jb_pval = calc_jb_test(es50_ret)[1]
print(es50_jb_tstat, es50_jb_pval)



# Check of intermediate result (2):
#
Test.assertEquals(np.round(es50_skew,2), -0.16, 'incorrect result')
Test.assertEquals(np.round(es50_jb_tstat, 1), 191.3, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(es50_kurt, 2), checker_results.ix[2,0], 'incorrect result')
    Test.assertEquals(np.round(es50_jb_pval, 3), checker_results.ix[3,0], 'incorrect result')







# Question 1c: Simulation of Returns
# ----------------------------------

# Define simulation parameters
#
n = len(es50_ret) # Number of simulation steps
print(n)


# Instantiate the random number generator (RNG)
#
rng = np.random.RandomState(10)


# DONE: Simulate Gaussian White Noise (GWN)
#   Hint: Use the 'standard normal' function of the 'rng' object

# Initialize vector
#
gwn = np.zeros(n)

for t in range(n):
     gwn[t] = rng.standard_normal() # Simulate GWN


# DONE: Simulate AR(1) time series
#  Hint: Use the above generated GWN as the unexpected shocks

# Initialize vector
#
ar1 = np.zeros(n)

phi = [0, 0.8]
sigma = 1

for t in range(1, n):
    ar1[t] = phi[0] + (phi[1] * ar1[t-1]) + (sigma * gwn[t])


# Time series plots
#
plt.figure(2); plt.figure(figsize = (16,15)); plt.grid(True); plt.axis('tight')
plt.subplot(411); plt.plot(es50['LN_Returns'], color = '#1E90FF'); plt.xlabel('ESTOXX 50 Daily Returns')
plt.subplot(412); plt.plot(gwn, color = '0.00'); plt.xlabel('Gaussian White Noise'); plt.xlim(0, n)
plt.subplot(413); plt.plot(ar1, color = '0.20'); plt.xlabel('AR(1)'); plt.xlim(0, n)


# Histograms
#
plt.figure(3); plt.figure(figsize = (20,10))
plt.subplot(221); plt.hist(es50_ret, color = '#1E90FF'); plt.xlabel('ESTOXX 50 Daily Returns')
plt.subplot(222); plt.hist(gwn, color='0.00'); plt.xlabel('Gaussian White Noise')
plt.subplot(223); plt.hist(ar1, color='0.20'); plt.xlabel('AR(1)')

#plt.show(block=True)


# DONE: Calculate higher Moments for GWN and AR(1) series
#
gwn_skew = skewness(gwn)
ar1_skew = skewness(ar1)

gwn_kurt = kurtosis(gwn)
ar1_kurt = kurtosis(ar1)



print("Skewness", "                                  Excess Kurtosis")
print("--------", "                                  ---------------")
print("ESTOXX 50: " + str(np.round(es50_skew,4)) + "            " + str(np.round(es50_kurt-3,4)))
print("GWN:       " + str(gwn_skew)  +             "            " + str(gwn_kurt))
print("AR(1):     " + str(ar1_skew)  +             "            " + str(ar1_kurt))
print("")


# DONE: Apply JB test to GWN and AR(1) series
#

print("JB test")
print(calc_jb_test(gwn))
print(calc_jb_test(ar1))





# Check of intermediate result (3):
#
Test.assertEquals(np.round(ar1[1], 2), 0.72, 'incorrect result')
Test.assertEquals(np.round(ar1_skew, 2), -0.18, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(ar1[2], 2), checker_results.loc[4][0], 'incorrect result')
    Test.assertEquals(np.round(ar1_kurt, 2), checker_results.loc[5][0], 'incorrect result')








# Question 1d: Ordinary Least Squares
# -----------------------------------


# Object-oriented
#
class OLS(object):


    def __init__(self, Y, X, const = True):

        self.Y = np.matrix(Y).getT()

        self.X = np.matrix(X).getT()

        if (const == True):

             self.X = np.insert(self.X, obj = 0, values = 1, axis = 1)



    def run_OLS(self, summary = True):

        # Formulae are given in the problem set

        nobs = int(self.X.shape[0]) #T
        nvar = int(self.X.shape[1]) #p

        beta = (self.X.getT() * self.X).getI() * self.X.getT() * self.Y

        epsilon = self.Y - (self.X * beta)

        sigma2_eps = (1/(nobs - nvar)) * (epsilon.getT() *  epsilon).item() # Hint: Use the .item()
        # function to
        # convert the
        #  (1,
        # 1) matrix into a
        # scalar

        var = sigma2_eps * (self.X.getT()*self.X).getI()

        se = np.sqrt(var.diagonal()).getT()

        tstat = beta / se

        sigma2_y = 1 / (nobs - 1) * sum(np.array((self.Y - np.mean(self.Y))) ** 2)

        adj_rsqr = 1 - (sigma2_eps / sigma2_y)


        # Set object variables
        #
        self.beta       = beta
        self.tstat      = tstat
        self.adj_rsqr   = adj_rsqr
        self.sigma2_eps = sigma2_eps

        # Print results
        #
        if (summary == True):
            print('Betas: ', beta)
            print('t-stats: ', tstat)
            print('R^2: ', adj_rsqr)


        return self.beta, self.tstat, self.adj_rsqr, sigma2_eps





# Question 1e: Parameter Estimation
# ----------------------------------

# DONE: Estimate AR(1) parameters for (simulated) stock returns
#   Note, include the constant in the estimation
#
y_dat = ar1[1:len(ar1)]
x_dat = ar1[0:len(ar1)-1]

reg_ar1 = OLS(y_dat, x_dat)      # Create OLS object
ar1_ols = reg_ar1.run_OLS()      # Run OLS estimation

ar1_ols_betas = ar1_ols[0]
ar1_ols_tstats = ar1_ols[1]
ar1_ols_adj_rsqr = ar1_ols[2]
ar1_ols_sigma2_eps = ar1_ols[3]


# Double check with statsmodels package
#
Y = ar1[1:len(ar1)]
X = ar1[0:len(ar1)-1]

# Add constant
X = sm.add_constant(X)

# Fit regression model
model = sm.OLS(Y, X)
results = model.fit()

# Print summary
print(results.summary())


reg_gwn = OLS(gwn[1:len(gwn)], gwn[0:len(gwn)-1])
gwn_ols = reg_gwn.run_OLS()

reg_es50 = OLS(es50_ret[1:len(es50_ret)], es50_ret[0:len(es50_ret)-1])
es50_ols = reg_es50.run_OLS()

# Check of intermediate result (4/5):
#
Test.assertEquals(np.round(ar1_ols_betas[0], 4),   -0.0023, 'incorrect result')
Test.assertEquals(np.round(ar1_ols_tstats[0], 4),  -0.0885, 'incorrect result')
Test.assertEquals(np.round(ar1_ols_sigma2_eps, 4),  0.9174, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(ar1_ols_betas[1], 4),  checker_results.loc[6][0],'incorrect result')
    Test.assertEquals(np.round(ar1_ols_tstats[1], 2), checker_results.loc[7][0], 'incorrect result')
    Test.assertEquals(np.round(ar1_ols_adj_rsqr, 2),  checker_results.loc[8][0], 'incorrect result')










# Question 1f: Forecasting
# ------------------------


# AR(1) coefficents from 1e
#
c   = ar1_ols_betas[0].item() # constant
phi = ar1_ols_betas[1].item()



# Function
#
def ar1_forecast(phi, c, j, r_t):

    # E_t[r_t+j] = c x Sum phi^i + phi^j x r_t
    #
    phi_sum = 0

    # i goes from 0 to j-1
    #
    for i in range(0, j):
        phi_sum += phi**i


    forecast = c * phi_sum + phi**j * r_t
    return forecast




r_t = ar1[len(ar1)-1] # Take last value of simulated series



# Forecasts
#
print('1 step ahead forecast: ', np.round( ar1_forecast(phi, c, 1, r_t), 4) )
print('2 step ahead forecast: ', np.round( ar1_forecast(phi, c, 2, r_t), 4) )

forecast_j1 = ar1_forecast(phi, c, 1, r_t)
forecast_j2 = ar1_forecast(phi, c, 2, r_t)

# Function
#
def ar1_forecast_var(phi, sigma2, j):

    # Var_t[r_t+j] = sigma^2 x Sum phi^2k
    #
    sum_phi = 0

    # i goes from 0 to j-1
    #
    for k in range(0, j):

        sum_phi += phi**(2*k)

    forecast_variance = sigma2 * sum_phi

    return forecast_variance



# Unbiased OLS for sigma^2 is e'* e / (T-K)
#
sigma2_eps = ar1_ols[3]


# Uncertainty of forecast
#
print('1 step ahead uncertainty of forecast: ', np.round( ar1_forecast_var(phi, sigma2_eps, 1), 4) )
print('2 step ahead uncertainty of forecast: ', np.round( ar1_forecast_var(phi, sigma2_eps, 2), 4) )

forecast_j1_var = ar1_forecast_var(phi, sigma2_eps, 1)
forecast_j2_var = ar1_forecast_var(phi, sigma2_eps, 2)


# Check of intermediate result (6):
#
Test.assertEquals(np.round(forecast_j1, 2), -0.37, 'incorrect result')
Test.assertEquals(np.round(forecast_j1_var, 2), 0.92, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(forecast_j2, 2),     checker_results.loc[9][0], 'incorrect result')
    Test.assertEquals(np.round(forecast_j2_var, 2), checker_results.loc[10][0], 'incorrect result')






print(Test.passed, "tests passed of", Test.numTests,  "(", 100*Test.passed/Test.numTests, "%)")