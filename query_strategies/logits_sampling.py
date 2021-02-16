import numpy as np
import torch
from .strategy import Strategy
import os
import pickle
import pdb
import matplotlib.pyplot as plt
from random import *
import random

import copy


class LogitSampling(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(LogitSampling, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs, P, logits = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

		L1_norm = torch.sum(torch.abs(logits),1)
		L2_norm = torch.sum(logits ** 2, 1).sqrt()
		sorted, _ = torch.sort(logits)

		#return idxs_unlabeled[L1_norm.sort()[1][:n]]
		#return idxs_unlabeled[L2_norm.sort()[1][:n]]
		return idxs_unlabeled[(sorted[:,-1]-sorted[:,-2]).sort()[1][:n]]

