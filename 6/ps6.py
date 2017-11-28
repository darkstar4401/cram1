#
# =========================================================
# Computational Risk and Asset Management WS 17/18
# Problem Set 6, Week 6 TEMPLATE
# The Volatility of Stock Returns and ARCH Modeling
# =========================================================
#
# Prepared by Elmar Jakobs
#





# Setup
# -----

# Import packages for econometric analysis
#
import numpy as np
import pandas as pd

import statsmodels.api as sm

from arch import arch_model
from arch.univariate import ZeroMean, ARCH


# Plotting library
#
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.style.use('ggplot')





# Import and configure the testing framework
#
from test_suite import Test
# For submission, please change the status to 'SOLN'
status = 'SOLN'

if status == 'SOLN':
    checker_results = pd.read_csv('results_ps6.csv').set_index('Task')





# Question 1a: Euro Stoxx 50 - Intra-Day Index Values
#  --------------------------------------------------


# TODO: Read-in intra-day Euro Stoxx 50 index values
#
es50_id = pd.read_csv("es50_id.csv", delimiter=";")
es50_id.head()

# TODO: Format date and set as index
#
es50_id['loctimestamp'] = pd.to_datetime(es50_id['loctimestamp'], format="%Y-%m-%d %H:%M:%S") # Format the date
es50_id = es50_id.rename(columns={'loctimestamp': 'Date'})
es50_id = es50_id.set_index('Date') # Set the index

del es50_id['instrumentid']

es50_id.head()

# TODO: Plot time series
#

es50_id.plot()
plt.ylabel('Euro Stoxx 50')



# TODO: Select one day
#
date = '06/10/2010'  # Specify date

es50_id_sub = es50_id[date]  # Select index values of specific day
es50_id_sub.head()



# TODO: Plot intra-day time series
#
es50_id_sub.plot()
#plt.show()




# TODO: Calculate returns
#
es50_id_sub['returns'] = np.log(es50_id_sub['price'] / es50_id_sub['price'].shift(1)) * 100
es50_id_sub.head()


# TODO: Plot intra-day returns
#
es50_id_sub['returns'].plot(title='intra-day returns')
#plt.show()



# TODO: Calculate cumulative returns
#
es50_id_sub['cum_returns'] = np.log(es50_id_sub['price'] / es50_id_sub['price'][0])*100
es50_id_sub.set_value( es50_id_sub.index[0], 'cum_returns', 0 )
es50_id_sub.head()

#
# TODO: Plot intra-day cumulative return series
#
es50_id_sub['cum_returns'].plot(title='intra-day cumulative returns')
#plt.show()



# Plot intra-day ECB returns
#

start    = date + ' 09:00:00'
end      = date + ' 17:30:00'

pr       = date + ' 13:45:00' # Press Release: 13:45:00 (Announcement)
pr_txt   = date + ' 13:55:00' # Press Release: 13:45:00 (Announcement)

pc_start = date + ' 14:30:00' # Press Conference: 14:30:00 (Start time)
pc_end   = date + ' 15:30:00' # Press Conference: 15:30:00 (End time)
pc_txt   = date + ' 14:55:00'


es50_id_sub['cum_returns'].loc[start:end].plot( color = 'blue' )
plt.title( 'ECB Monetary Policy Decision: ' + date)
plt.vlines( pr, -0.5, 3.0, linestyles = 'dashed', color = 'orange', label = 'Press Release' )
plt.vlines( pc_start, -0.5, 3.0, linestyles = 'dashed', color = 'orange', label = 'Press Conference' )
plt.vlines( pc_end, -0.5, 3.0, linestyles = 'dashed', color = 'orange', label = 'Press Conference' )
plt.text( pr_txt, 3.0,  'Press Release',   rotation = 90, bbox = dict(facecolor = 'green', alpha = 0.5) )
plt.text( pc_txt, 3.0, 'Press Conference', rotation = 90, bbox = dict(facecolor = 'green', alpha = 0.5) )
#plt.show()



# Read-in (average) ECB drift
#   Thx @F.T. for nice plots :)
#
img     = mpimg.imread( 'ecb_drift.png' )
imgplot = plt.imshow( img )
#plt.show()




# Check of intermediate result (1):
#
Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:00:15']['price'],  2), 2542.01,   'incorrect result')
Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:00:30']['returns'],4),   -0.1354, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:00:45']['cum_returns'], 4), checker_results.loc[1][0], 'incorrect result')
    Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:01:00']['cum_returns'], 4), checker_results.loc[2][0], 'incorrect result')








# Question 1b:  Stock Return Volatility
# -------------------------------------


#
# Problem: Volatility not observed!
#  -> Approximation: Sum of 5 min squared (tick-by-tick) returns (see Lucca and Moench (2015))
#

# TODO: Calculate squared returns
#
es50_id_sub['returns_squ'] = es50_id_sub['returns']**2


# TODO: Calculate 5 min sum of 15 sec returns
#
es50_id_sub['volatility'] = pd.rolling_sum(es50_id_sub['returns_squ'], 20)
es50_id_sub[15:25]


# TODO: Plot intra-day stock returns and volatility
#
es50_id_sub.plot(y=['returns', 'volatility'])
#plt.show()



# Plot intra-day ECB returns
#

start    = date + ' 09:00:00'
end      = date + ' 17:30:00'

pr       = date + ' 13:45:00' # Press Release: 13:45:00 (Announcement)
pr_txt   = date + ' 13:55:00' # Press Release: 13:45:00 (Announcement)

pc_start = date + ' 14:30:00' # Press Conference: 14:30:00 (Start time)
pc_end   = date + ' 15:30:00' # Press Conference: 15:30:00 (End time)
pc_txt   = date + ' 14:55:00'


es50_id_sub['volatility'].loc[start:end].plot( color = 'navy' )
plt.title( 'ECB Monetary Policy Decision: ' + date)
plt.vlines( pr, 0, 0.13, linestyles = 'dashed', color = 'orange', label = 'Press Release' )
plt.vlines( pc_start, 0, 0.13, linestyles = 'dashed', color = 'orange', label = 'Press Conference' )
plt.vlines( pc_end, 0, 0.13, linestyles = 'dashed', color = 'orange', label = 'Press Conference' )
plt.text( pr_txt, 0.12,  'Press Release',   rotation = 90, bbox = dict(facecolor = 'green', alpha = 0.5) )
plt.text( pc_txt, 0.12, 'Press Conference', rotation = 90, bbox = dict(facecolor = 'green', alpha = 0.5) )



# # Read-in (average) ECB drift volatility
# #   Thx @F.T. for nice plots :)
# #
# img     = mpimg.imread( 'ecb_drift_vol.png' )
# imgplot = plt.imshow( img )
# plt.show()
# #
#


# Check of intermediate result (2):
#
Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:00:30']['returns_squ'], 4), 0.0183, 'incorrect result')
Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:00:45']['returns_squ'], 4), 0.0007, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:05:15']['volatility'], 4), checker_results.loc[3][0], 'incorrect result')
    Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:05:30']['volatility'], 4), checker_results.loc[4][0], 'incorrect result')





# Question 1c: ECB Monetary Policy Decisions
# ------------------------------------------
#
#
# Please provide qualitative answers in your pdf
#
#
#
#
#
#
#
#
#
# Question 1d: 2 Pass Estimation
# ------------------------------
#


# Class
#
class ARMA_ARCH(object):


    def __init__( self, returns ):

        self.returns = returns



    def two_pass_est( self, print_results = True, plot_results = False ):


        #
        # See Chapter 8.4.1 of the lecture notes
        #


        # TODO: First, specify 'mean equation' and estimate it via OLS
        #

        # Choosing the 'right' (best) model
        #    -> Use information criteria, e.g. AIC/ BIC
        #
        #  We use a parsimonious model here
        #    -> AR(1)
        #

        n = len(self.returns)

        Y = list(self.returns[1:n])

        X = self.returns[0:n-1]
        X = sm.add_constant(X) # Add constant

        model_mean = sm.OLS(Y, X)
        results_mean = model_mean.fit() # Fit the model

        if print_results == True:
            print( results_mean.summary() )



        # TODO: Second, take residuals eps_t and square them (eps^2_t)
        #

        resid = results_mean.resid
        resid_squ = resid**2

        if plot_results == True:
            plt.plot( resid_squ )


        #
        # Test for ARCH effects (not implemented here)



        # TODO: Third, run MLE optimization with eps^2_t as observed time series to estimate ARCH model
        #
        # ARCH(1) model:
        #   sigma^2_t-1 := alpha_0 + alpha_1 * eps^2_t-1,
        #      with alpha_0 > 0 and alpha_1 >= 0
        #

        # Note, 'arch' package requires 'not squared' residuals as input
        # #

        model_vol = ZeroMean(resid) # Instantiate object (mean equation) (mean is included in resid)
        model_vol.volatility = ARCH(p=1)
        results_vol = model_vol.fit() # Fit the model


        volatility = results_vol.conditional_volatility  # Extract the volatility


        if print_results == True:
            print( results_vol.summary() )

        if plot_results == True:
            results_vol.plot()


        return volatility, results_vol, resid_squ


# Question 1e: Model-Implied Volatility
# -------------------------------------


# Length of return series
#
n = len(es50_id_sub['returns']) # 2042

returns = np.array( es50_id_sub.iloc[1:n]['returns'] )
returns[0:5]




# TODO: Two Pass Estimation
#
model_arch = ARMA_ARCH(returns)  # Instantiate class
arch_vol, est_results_vol, resid_squ = model_arch.two_pass_est()  # Run two pass estimation

arch_vol[0:5]
print(est_results_vol)

alpha_0_hat = est_results_vol.params[0]  # Extract parameter
alpha_0_hat

alpha_1_hat = est_results_vol.params[1]  # Extract parameter
alpha_1_hat


resid_squ[0:5]



# Robustness Check: 'arch' package
#   Estimation via joint (!) likelihood
#

#?arch_model

model_arch_package = arch_model(returns, mean='ARX', lags=1, vol='ARCH', p=1)
results_arch_package = model_arch_package.fit()
arch_package_vol = results_arch_package.conditional_volatility




# TODO: Comparison with squared returns
#
es50_id_sub.loc[2:n, 'volatility_arch_2pass'] = arch_vol
es50_id_sub.loc[1:n, 'volatility_arch_package'] = arch_package_vol

es50_id_sub.head()

es50_id_sub[['volatility_arch_2pass', 'volatility_arch_package']].plot( subplots = True )

es50_id_sub[['volatility_arch_2pass', 'volatility_arch_package', 'volatility']].plot()
#plt.show()


# TODO: Obtain correlation matrix between volatility proxies
#
es50_id_sub[['volatility_arch_2pass', 'volatility_arch_package', 'volatility']].cov()





# Check of intermediate result (3):
#
Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:00:45']['volatility_arch_2pass'], 4), 0.0309, 'incorrect result')
Test.assertEquals(np.round(es50_id_sub.loc['2010-06-10 09:01:00']['volatility_arch_2pass'], 4), 0.0269, 'incorrect result')
Test.assertEquals(np.round(resid_squ[0], 4), 0.0001, 'incorrect result')
Test.assertEquals(np.round(resid_squ[1], 4), 0.0003, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(alpha_0_hat, 4),  checker_results.loc[5][0], 'incorrect result')
    Test.assertEquals(np.round(alpha_1_hat, 4),  checker_results.loc[6][0], 'incorrect result')
    Test.assertEquals(np.round(resid_squ[2], 4), checker_results.loc[7][0], 'incorrect result')
    Test.assertEquals(np.round(resid_squ[3], 4), checker_results.loc[8][0], 'incorrect result')







# Question 1f: Volatility Forecasting
# -----------------------------------


# Recursive variance forecasting with ARCH(1) model:
#   E_t[sigma^2_t+h] = alpha_0 + alpha_1 * E[a^2_t+h-1]
#


# Forecast 1 period ahead
#

# TODO: Take last known value
#
last_value = es50_id_sub['returns_squ'][-1]


# TODO: Get parameter estimates for alpha_0 and alpha_1
#
alpha_0_hat = results_arch_package.params[2]

alpha_1_hat = results_arch_package.params[3]


# TODO: Forecast 1, 2 and 3 period(s) ahead
exp_var_h1 = alpha_0_hat + alpha_1_hat * last_value
exp_var_h2 = alpha_0_hat + (alpha_0_hat + alpha_1_hat) * exp_var_h1
exp_var_h3 = alpha_0_hat + (alpha_0_hat + alpha_1_hat) * exp_var_h2


print( 'Forecast 1 period  ahead: ', np.round(exp_var_h1, 6) )
print( 'Forecast 2 periods ahead: ', np.round(exp_var_h2, 6) )
print( 'Forecast 3 periods ahead: ', np.round(exp_var_h3, 6) )



# Double check with 'arch' package
#
exp_var_package = results_arch_package.forecast( horizon = 3, start = 2040 )
print( exp_var_package.variance.iloc[-3:] )


#
# Distribution of expected stock returns 1 step ahead?
#





# Check of intermediate result (4):
#
Test.assertEquals(np.round(exp_var_h1, 4), 0.0013, 'incorrect result')
Test.assertEquals(np.round(exp_var_h2, 4), 0.0009, 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(np.round(exp_var_h3, 4),     checker_results.loc[9][0], 'incorrect result')
    Test.assertEquals(np.round(resid_squ[-1], 4),  checker_results.loc[10][0], 'incorrect result')





print(Test.passed, "tests passed of", Test.numTests,  "(", 100*Test.passed/Test.numTests, "%)")


