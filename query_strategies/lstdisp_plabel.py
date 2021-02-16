import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import pdb
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist
import copy


class LstDisp_PLabel(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(LstDisp_PLabel, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query_pseudoLabel(self, n):

		pseudo_Y = copy.deepcopy(self.Y)
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		U = probs.max(1)[0]

		indices_of_pseudos = U.sort()[1][-n:]
		global_indices_in_unlabeled_set_for_pseudo_labeled = idxs_unlabeled[indices_of_pseudos]

		# computing the accuracy of pseudo labels
		pseudo_accuracy = np.zeros(len(global_indices_in_unlabeled_set_for_pseudo_labeled))
		global_indices_in_unlabeled_set_for_pseudo_labeled = idxs_unlabeled[indices_of_pseudos]
		for sample in range(len(indices_of_pseudos)):
			pseudo_accuracy[sample] = (
						P[indices_of_pseudos[sample]] == self.Y[global_indices_in_unlabeled_set_for_pseudo_labeled[sample]])

		print('\n')
		print('correct pseudo labels: ', sum(pseudo_accuracy))
		print('total pseudos: ', len(pseudo_accuracy))
		print('acurracy of pseudo label: ', float(sum(pseudo_accuracy)) / float(len(pseudo_accuracy)))

		# Change gt labels with last label
		for sample in indices_of_pseudos:
			pseudo_Y[idxs_unlabeled[sample]] = P[sample]

		return idxs_unlabeled[U.sort()[1][-n:]], pseudo_Y


	def query(self, n):
		lb_flag = self.idxs_lb.copy()

		# Dispersion
		# load inferred classes for sampling
		with open(self.dataset + '_results/' + self.method + '/AUX_inferred_classes_cycle_' + str(self.cycle) + ".pkl",
					  "rb") as f:
				inferred_classes = pickle.load(f)

		# computing dispersion
		dispersion = torch.zeros(inferred_classes.size(0))
		correct_pred = 0
		for sample in range(0, inferred_classes.size(0)):
			class_mode, _ = torch.mode(inferred_classes[sample, :])
			dispersion[sample] = 1 - (
					sum(inferred_classes[sample, :] == class_mode.item()).to(dtype=torch.float) / float(self.args['n_epoch'] + 1))

		values, indices = torch.topk(dispersion, n, largest=True)
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		return np.asarray(idxs_unlabeled)[indices]
