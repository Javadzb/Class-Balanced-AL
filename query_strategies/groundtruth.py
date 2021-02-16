import numpy as np
import torch
from .strategy import Strategy
import os
import pickle
import pdb

class Groundtruth(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(Groundtruth, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)

		if self.dataset == 'cifar10':
			num_classes=10
		elif self.dataset == 'cifar100':
			num_classes=100

		global_indices_in_unlabeled_set = []
		for i in range(0, num_classes):
			#samples_of_class_i = [j for j, value in enumerate(P) if P[j] == i and P[j]!= self.Y[idxs_unlabeled[j]]]
			samples_of_class_i = [j for j, value in enumerate(P) if self.Y[idxs_unlabeled[j]] == i and P[j]!= self.Y[idxs_unlabeled[j]]]
			indices= np.random.choice(samples_of_class_i, int(n / num_classes), replace=False)
			global_indices_in_unlabeled_set.extend(idxs_unlabeled[indices])
		return global_indices_in_unlabeled_set

		#global_indices_in_unlabeled_set = []
		#for i in range(0, num_classes):
		#    global_loc = [j for j, value in enumerate(eval_labels) if eval_labels[j] == i]
		#    _, indices = torch.topk(dispersion[global_loc], len(global_loc), largest=True)
		#    # HEY JAVAD ! Remember to replace the following line (which is only valid for Caltech 256)
		#    temp = [unlabeled_set[global_loc[j]] for j in indices[0:budget / num_classes]]
		#    # temp = [unlabeled_set[global_loc[j]] for j in indices[0:np.random.choice([5, 6], 1, p=[0.7, 0.3])[0]]]
		#    global_indices_in_unlabeled_set.extend(temp)
		#active_set.extend(global_indices_in_unlabeled_set)
		#unlabeled_set = np.setdiff1d(np.asarray(unlabeled_set), global_indices_in_unlabeled_set).tolist()

		#return idxs_unlabeled[U.sort()[1][:n]]
