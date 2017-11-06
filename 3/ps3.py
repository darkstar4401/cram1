#
# ================================================
# Computational Risk and Asset Management WS 17/18
# Problem Set 3, Week 3 TEMPLATE
# Linear Factor Models and Equity Risk Premium
# ================================================
#
# Prepared by Elmar Jakobs
#




# Linear Factor Models and Equity Risk Premium
# ============================================


# Setup
# -----

# Import packages for econometric analysis
#

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Import and configure the testing framework
#

from test_suite import Test
#from test_suite import Test

from algorithms.ols import OLS
from algorithms.gwn import GWN

# For submission, please change the status to 'SOLN'
status = ''

if status == 'SOLN':
    checker_results = pd.read_csv('results_ps3.csv').set_index('Task')







# Question 1a: Test Assets and Factor Returns
# -------------------------------------------


#
# Data
# xxxx
#
# 16 (4x4) Portfolios formed on beta & book-to-market
#
# Data from http://www.cfr-cologne.de/english/version06/html/research.php?topic=paper&auswahl=data&v=dl
#
# Description
# ...........
#
# Beta is estimated relative to DAFOX (CDAX from 2005 onwards) at the end of June of each year T using rolling five year time series regressions with monthly returns.
# To calculate BE/ME in year T we divide book equity for the fiscal year ending in calendar year T-1 by the market value of equity at the end of December in calendar year T-1.
# row (BE/ME), column(beta); 1=low, 4=high
#




# DONE: Read-in test portfolios (16x)
#  Hint: Use the pandas read_table function to read-in the text file
#  Hint #2: To skip the description in the text file set the 'skiprows = 4' argument
#
pf_16 = pd.read_table('16pf_bm_beta.TXT', skiprows = 4)
pf_16.head()


# DONE: Convert the date and set the index
#  Hint: The format of the date is '%Y%m'
#
pf_16['date'] = pd.to_datetime(pf_16['date'], format='%Y%m')
pf_16 = pf_16.set_index('date')
pf_16.tail()

pf_16 = pf_16.resample('M').ffill()
pf_16.head()



# DONE: Read-in the factors for German market (RF is risk free rate):
#   1. RM (Market factor)
#   2. SMB (Fama-French factor)
#   3. HML (Fama-French factor)
#   4. WML (Carhart factor)
#

factors = pd.read_table('monthly_factors.TXT')

# DONE: Convert the date and set the index
#
factors['date'] = pd.to_datetime(factors['date'], format='%Y%m')
factors = factors.set_index('date')
factors = factors.resample('M').ffill()

factors.head()






# Check of intermediate result (1):
#
Test.assertEquals(str(np.round(pf_16.loc['2013-12-31']['11'],2))[0:5],   '2.7', 'incorrect result')
Test.assertEquals(str(np.round(factors.loc['2013-12-31']['rm'],2))[0:5], '1.37', 'incorrect result')
if status == 'SOLN':
    Test.assertEquals(str(np.round(pf_16.loc['2013-12-31']['44'], 2))[0:5],    str(checker_results.loc[1][0])[0:5], 'incorrect result')
    Test.assertEquals(str(np.round(factors.loc['2013-12-31']['HML'], 2))[0:5], str(checker_results.loc[2][0])[0:5], 'incorrect result')

# Question 1b: Capital Asset Pricing Model
# ----------------------------------------


# Estimate the Capital Asset Pricing Model for the '11' and '44' portfolios
#

# DONE: First, calculate excess returns
#
X = factors['rm']-factors['rf']
X.columns = ['rm']
#X = sm.add_constant(X)
X.head()

Y = pf_16['11']-factors['rf']
Y.head()



# DONE: Next, run ols regression to estimate the parameters
#
model_11 = OLS(Y, X) # we already added a constant
results_11 = model_11.run_OLS() # Fit the model


# DONE: Redo exercise for portfolio '44'
#

Y = pf_16['44']-factors['rf']
Y.head()

model_44 = OLS(Y, X)
results_44 = model_44.run_OLS() # Fit the model




# DONE: Forecasting using the CAPM
#   Expected market risk premium: 6%
#   CAPM prices stock returns perfectly
#     -> alpha =       ?
#     -> Exp Ret =     ?
#
alpha_11 = results_11.beta[0]
exp_ret_11 = alpha_11 + (results_11.beta[1]*6)

alpha_44 = results_44.beta[0]
exp_ret_44 = alpha_44 + (results_44.beta[1]*6)

print( 'Expected return for portfolio 11: ', np.round(exp_ret_11, 2) )
print( 'Expected return for portfolio 11: ', np.round(exp_ret_44, 2) )


# Check of intermediate result (2):
#
#Test.assertEquals(str(np.round(results_11.params[1],2))[0:5],   '0.33', 'incorrect result')
#Test.assertEquals(str(np.round(exp_ret_11,2))[0:5],             '1.96', 'incorrect result')
Test.assertEquals(np.round(results_11.params[1],2),   0.33, 'incorrect result')
Test.assertEquals(np.round(exp_ret_11,2),   1.96, 'incorrect result')

if status == 'SOLN':
    Test.assertEquals(str(np.round(results_44.params[1],2))[0:5],    str(checker_results.loc[3][0])[0:5], 'incorrect result')
    Test.assertEquals(str(np.round(exp_ret_44, 2))[0:5],             str(checker_results.loc[4][0])[0:5], 'incorrect result')




#
#
# # Question 1c: Fama MacBeth Methodology
# # -------------------------------------
#
#
# # RQ: What is the price of (equity) risk? (vs. e.g. risk-free bond)
# #     --> 5 %? 10 %?
# #
#
#
#
#
#
#
# # Question 1d: Class 'ERP'
# # ------------------------
#
# # Object-oriented
# #
# class ERP(object):
#
#
#     def __init__( self, equ_rets, factor_rets):
#
#         self.equity_rets = equ_rets
#
#         self.factor_rets = factor_rets
#
#
#
#     def estimate_ERP(self, summary = True):
#
#         #
#         # Two stage (Fama MacBeth) Regression
#         #
#
#         # 1. Stage: Estimate betas
#         #
#
#         # TODO: Set number of test assets
#         #  Hint: use the '.shape()' function
#         #
#         nr_assets =    # E.g. 16 (portfolios)
#
#
#         # TODO: Create an empty vector which saves the betas for each test asset
#         #
#         betas =
#
#
#         for i in range(0, nr_assets):
#
#             # TODO: First, calculate excess returns
#             #
#             pf_ex_rets     =
#
#             market_ex_rets =
#
#
#             # TODO: Next, estimate beta coefficient
#             #
#             Y       =
#
#             X       =
#             X       =     # Add constant
#
#             model   =
#             results =     # Fit model
#
#
#             # TODO: Save estimated betas (beta_hat)
#             #
#             betas[i] =
#
#             betas[i] = format(betas[i], '.2f')
#
#
#         if summary == True:
#
#             betas_tup = tuple( ( betas[x], self.equity_rets.columns[x] ) for x in range(0, nr_assets) )
#
#             betas_sorted = sorted(betas_tup, reverse = True)
#
#             print('Portfolio \t \t Beta')
#             print('--------- \t \t ----')
#
#             for i in range( 0, nr_assets ):
#
#                 print(betas_sorted[i][1], '\t \t \t \t', format(betas_sorted[i][0], '.2f'))
#
#             print('')
#
#
#
#
#         # 2. Stage: Estimate factor premiums
#         #
#
#         # TODO: Set number of time points T
#         #  Hint: use the '.shape()' function
#         #
#         T          =     # Number of asset returns (time series)
#
#
#         # TODO: Create an empty vector which saves the lambdas and alphas for each time point
#         #
#         mrp_all    =
#         alphas_all =
#
#
#         # For every time step, run cross-sectional regression with different returns but same beta estimates
#         #
#         for i in range(0, T):
#
#
#             # TODO: First, calculate excess returns
#             #
#             pf_ex_rets    =
#
#
#             # TODO: Next, estimate alphas and lambdas
#             #
#             Y              =
#             Y              = Y.reset_index()
#             del Y['index']
#             Y.columns      = ['pf_rets_t']
#
#             X              = pd.DataFrame( data = betas, columns = ['lambda'] )
#             X              = sm.add_constant(X)
#
#             model         =
#             results       =
#
#             mrp_all[i]    =
#             alphas_all[i] =
#
#
#         # TODO: Calculate average alpha and lambda
#         #
#         alpha  =
#         mrp    =
#
#
#         # Calculate t-stat
#         #
#
#         t_stat_mrp =
#
#
#         print('')
#         print('Equity Risk Premium')
#         print('-------------------')
#         print( 'Alpha: ', np.round(alpha, 2) )
#         print ('MRP:   ', np.round(mrp, 2), '(', np.round(t_stat_mrp, 2), ')' )
#         print('')
#
#         return alpha, mrp, t_stat_mrp
#
#
#
#
#
# # Question 1e: Estimate Equity Risk Premium
# # -----------------------------------------
#
# # TODO: Estimate ERP
# #
# erp_ger       =     # Instantiate the object with the previously created data frames for the test assets and factors
# erp_ger_est   =     # Run the .estimate_ERP function
#
# ger_alpha     = erp_ger_est[0]
# ger_erp       = erp_ger_est[1]
# ger_erp_tstat = erp_ger_est[2]
#
#
#
#
#
# # Check of intermediate result (3):
# #
# Test.assertEquals(np.round(ger_alpha,2), 0.48, 'incorrect result')
# Test.assertEquals(np.round(ger_alpha,2), 0.48, 'incorrect result')
# if status == 'SOLN':
#     Test.assertEquals(np.round(ger_erp, 2),       checker_results.loc[5][0], 'incorrect result')
#     Test.assertEquals(np.round(ger_erp_tstat, 2), checker_results.loc[6][0], 'incorrect result')
#
#
#
#
#
#
# # Question 1f: Fama-French 3 Factor Model
# # ---------------------------------------
#
#
# # Estimate the Fama-French 3 Factor Model for portfolios '11' and '44'
# #
#
# # TODO: First, calculate market excess return
# #
# X            =
# X.columns    = ['rm']
#
#
# # TODO: Second, add the two other factors
# #
# X['SMB']     =
# X['HML']     =
# X            = sm.add_constant(X)
# X.head()
#
#
# # TODO: Next, calculate portfolio excess return
# #
# Y =
# Y.head()
#
#
# # TODO: Next, estimate the FF3F model
# #
# model_11   =
# results_11 =  # Fit model
#
# # Note, portfolios are sorted based on market beta and HML
# #
# results_11.summary()
#
#
# # TODO: Redo exercise for '44' portfolio
#
# Y =
# Y.head()
#
# model_44   =
# results_44 =  # Fit model
# results_44.summary()
#
#
#
#
#
#
# # Check of intermediate result (4):
# #
# Test.assertEquals(np.round(results_11.params[1],2),   0.48, 'incorrect result')
# Test.assertEquals(np.round(results_11.params[3],2),  -0.09, 'incorrect result')
# Test.assertEquals(np.round(results_11.tvalues[1],2), 23.63, 'incorrect result')
# Test.assertEquals(np.round(results_11.tvalues[3],2), -2.95, 'incorrect result')
# if status == 'SOLN':
#     Test.assertEquals(np.round(results_44.params[1], 2),  checker_results.loc[7][0], 'incorrect result')
#     Test.assertEquals(np.round(results_44.params[3], 2),  checker_results.loc[8][0], 'incorrect result')
#     Test.assertEquals(np.round(results_44.tvalues[1], 2), checker_results.loc[9][0], 'incorrect result')
#     Test.assertEquals(np.round(results_44.tvalues[3], 2), checker_results.loc[10][0], 'incorrect result')
#
#
#
#
#
#
#
#
# print(Test.passed, "tests passed of", Test.numTests,  "(", 100*Test.passed/Test.numTests, "%)")
#
#
