import numpy as np
from .strategy import Strategy
import os
import pickle


class LeastConfidence(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(LeastConfidence, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		U = probs.max(1)[0]

		# Save uncertainity vector
		fname = os.path.join(self.dataset + '_results', self.method)
		if not os.path.exists(fname):
			os.makedirs(fname)
		with open(fname + "/uncertainty_cycle_" + str(self.cycle) + ".pkl", "wb") as f:
			pickle.dump(U, f)

		return idxs_unlabeled[U.sort()[1][:n]]
