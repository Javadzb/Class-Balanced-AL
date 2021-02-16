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


class Disp_Pseudo_Label(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(Disp_Pseudo_Label, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query_pseudoLabel(self, n):

		lb_flag = copy.deepcopy(self.idxs_lb)
		pseudo_Y = copy.deepcopy(self.Y)

		# Dispersion
		# load inferred classes for sampling
		with open(self.dataset + '_results/' + self.method + '/inferred_classes_cycle_' + str(self.cycle) + ".pkl",
					  "rb") as f:
				inferred_classes = pickle.load(f)

		# computing dispersion
		dispersion = torch.zeros(inferred_classes.size(0))
		correct_pred = 0
		for sample in range(0, inferred_classes.size(0)):
			class_mode, _ = torch.mode(inferred_classes[sample, :])
			dispersion[sample] = 1 - (sum(inferred_classes[sample, :] == class_mode.item()).to(dtype=torch.float) / float(self.args['n_epoch'] + 1))

		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		if 'balanced' in self.method:
			temp = []
			for i in range(0, num_classes):
				global_loc = [j for j, value in enumerate(inferred_classes) if torch.mode(inferred_classes[j, :])[0] == i]
				_, indices = torch.topk(dispersion[global_loc], len(global_loc), largest=False)
				temp.extend([idxs_unlabeled[global_loc[j]] for j in indices[0:int(budget / num_classes)]])
			indices_of_pseudos = [idxs_unlabeled.index(k) for k in temp]

		else:
			_, indices_of_pseudos = torch.topk(dispersion, n, largest=False)

		# pdb.set_trace()
		global_indices_in_unlabeled_set_for_pseudo_labeled = np.asarray(idxs_unlabeled)[indices_of_pseudos]

		# computing the accuracy of pseudo labels
		pseudo_accuracy = np.zeros(len(global_indices_in_unlabeled_set_for_pseudo_labeled))
		for sample in range(len(indices_of_pseudos)):
			pseudo_accuracy[sample] = (torch.mode(inferred_classes[indices_of_pseudos[sample], :])[0] == self.Y[global_indices_in_unlabeled_set_for_pseudo_labeled[sample]])
		print('\n')
		print('correct pseudo labels: ', sum(pseudo_accuracy))
		print('total pseudos: ', len(pseudo_accuracy))
		print('acurracy of pseudo label: ', float(sum(pseudo_accuracy)) / float(len(pseudo_accuracy)))

		# Change gt labels with mode
		for sample in indices_of_pseudos:
			class_mode, _ = torch.mode(inferred_classes[sample, :])
			pseudo_Y[idxs_unlabeled[sample]] = class_mode

		return global_indices_in_unlabeled_set_for_pseudo_labeled, pseudo_Y


	def query(self, n):
		lb_flag = self.idxs_lb.copy()

		# Dispersion
		# load inferred classes for sampling
		with open(self.dataset + '_results/' + self.method + '/AUX_inferred_classes_cycle_' + str(self.cycle) + ".pkl",
		# for classic AL cycle:
		#with open(self.dataset + '_results/' + self.method + '/inferred_classes_cycle_' + str(self.cycle) + ".pkl",
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
