import numpy as np


class OLS(object):


  def __init__(self, Y, X, const = True):

      self.Y = np.matrix(Y).getT()

      self.X = np.matrix(X).getT()

      self.beta = 0
      self.tstat = 0
      self.adj_rsqr = 0

      if (const == True):

           self.X = np.insert(self.X, obj = 0, values = 1, axis = 1)


  def print_summary(self):
      print('Betas: ', self.beta)
      print('t-stats: ', self.tstat)
      print('R^2: ', self.adj_rsqr)


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
      self.params = self.beta[0], self.beta[1], self.tstat, self.adj_rsqr

      # Print results
      #
      if (summary == True):
          self.print_summary()
          #print('Betas: ', beta)
          #print('t-stats: ', tstat)
          #print('R^2: ', adj_rsqr)

      return self

      #return self.beta, self.tstat, self.adj_rsqr, sigma2_eps


