import numpy as np
from .strategy import Strategy

class RandomSampling(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(RandomSampling, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n , replace=False)
