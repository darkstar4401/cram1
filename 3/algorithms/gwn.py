import numpy as np



class GWN:
  def __init__(self, n):
    # Instantiate the random number generator (RNG)
    #
    self.rng = np.random.RandomState(10)

    # Initialize vector
    #
    self.gwn = np.zeros(n)

    for t in range(n):
      self.gwn[t] = self.rng.standard_normal()  # Simulate GWN


  def get_gwn(self):
    return self.gwn