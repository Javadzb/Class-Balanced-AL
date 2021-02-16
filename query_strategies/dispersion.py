import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import pdb
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist
import os
import random
import copy

class Dispersion(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(Dispersion, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):

		# Dispersion
		# load inferred classes for sampling
		inferred_classes=torch.load(self.dataset + '_results/' + self.method + '/inferred_classes_cycle_' + str(self.cycle) + ".pt")
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]


		# computing dispersion
		dispersion = torch.zeros(inferred_classes.size(0))
		for sample in range(0, inferred_classes.size(0)):
			class_mode, _ = torch.mode(inferred_classes[sample, 75:]) #for CIFAR 100 take the mode of 75:
			dispersion[sample] = 1 - (
					sum(inferred_classes[sample,:] == class_mode.item()).to(dtype=torch.float) / float(self.args['n_epoch'] + 1))

		if 'normal' in self.method:
			values, indices = torch.topk(dispersion, n, largest=True)
			return idxs_unlabeled[indices]

		if self.dataset == 'cifar10':
			num_classes=10
		elif self.dataset == 'cifar100':
			num_classes=100


		if 'balanced' in self.method:
			#temp = []
			indices_of_pseudos=[]
			for i in range(0, num_classes):
				global_loc = [j for j, value in enumerate(inferred_classes) if torch.mode(inferred_classes[j, :])[0] == i]
				_, indices = torch.topk(dispersion[global_loc], len(global_loc), largest=True)
				indices_of_pseudos.extend([global_loc[j] for j in indices[0:int(n / num_classes)]])
			#indices_of_pseudos = [idxs_unlabeled[k] for k in temp]
		#else:
		# 	_, indices_of_pseudos = torch.topk(dispersion, n, largest=True)

		elif 'hard_mining' in self.method:
			indices_of_pseudos=[]
			k=int(0.2*len(idxs_unlabeled))
			_, INDICES = torch.topk(dispersion, k, largest=False) # the most certain k samples
			classes = np.zeros(k)
			for ind in range(len(INDICES)):
				classes[ind], _ = torch.mode(inferred_classes[INDICES[ind].item(), 75:])
			class_counts=np.zeros(num_classes)
			for c in range(num_classes):
				class_counts[c]=np.count_nonzero(classes == c)

			class_share = class_counts / k

			underpresented = np.where(class_counts == 0)
			class_share[underpresented]=1
			inverse_counts=1/class_share
			inverse_counts[underpresented]=n
			#samples_share=n * x / sum(x)

			# Sample share equalizer
			# setting the threshold
			samples_share = np.zeros(100)
			sort_idx = np.argsort(class_counts)

			th = 50
			samples_share[sort_idx[0:th]] = int(n / th)

			# Visualization
			sort_idx = np.argsort(class_counts)
			plt.plot(class_counts[sort_idx], color='coral', linewidth=4)
			gt_class_counts = np.zeros(num_classes)
			for c in range(num_classes):
				gt_class_counts[c] = np.count_nonzero(self.Y[idxs_unlabeled[INDICES]] == c)
			plt.plot(gt_class_counts[sort_idx], color='orange', linewidth=3, linestyle='dashed')
			plt.plot(samples_share[sort_idx], color='green', linewidth=4)

			for i in range(0, num_classes):
				global_loc = [j for j, value in enumerate(inferred_classes) if torch.mode(inferred_classes[j, :])[0] == i]

				_, indices = torch.topk(dispersion[global_loc], len(global_loc), largest=True)
				indices_of_pseudos.extend([global_loc[j] for j in indices[0:int(samples_share[i])]])

			global_indices_in_unlabeled_set_for_pseudo_labeled = np.asarray(idxs_unlabeled)[indices_of_pseudos]

			AS_class_counts = np.zeros(num_classes)
			for c in range(num_classes):
				AS_class_counts[c] = np.count_nonzero(self.Y[global_indices_in_unlabeled_set_for_pseudo_labeled] == c)
			plt.plot(AS_class_counts[sort_idx], color='limegreen', linewidth=3, linestyle='dashed')

			# saving class distribution figure
			if not os.path.exists(self.dataset + '_results/' + self.method):
				os.mkdir(self.dataset + '_results/' + self.method)
			plt.savefig(self.dataset + '_results/' + self.method + '/active_set_dist_cycle_' + str(self.cycle))
			plt.close()

		elif 'L1' in self.method:

			indices_of_pseudos=[]
			labeled_classes=self.Y[self.idxs_lb]
			_, counts = np.unique(labeled_classes, return_counts=True)
			class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
			class_share=class_threshold-counts
			samples_share= [0 if c<0 else c for c in class_share]

			for i in range(0, num_classes):
				global_loc = [j for j, value in enumerate(inferred_classes) if torch.mode(inferred_classes[j, 75:])[0] == i]

				_, indices = torch.topk(dispersion[global_loc], len(global_loc), largest=True)
				indices_of_pseudos.extend([global_loc[j] for j in indices[0:int(samples_share[i])]])

			global_indices_in_unlabeled_set_for_pseudo_labeled = np.asarray(idxs_unlabeled)[indices_of_pseudos]

			so_far_selected = len(global_indices_in_unlabeled_set_for_pseudo_labeled)

			if n < so_far_selected:
				random.shuffle(global_indices_in_unlabeled_set_for_pseudo_labeled)
				global_indices_in_unlabeled_set_for_pseudo_labeled = global_indices_in_unlabeled_set_for_pseudo_labeled[so_far_selected - n:]

			if so_far_selected < n:  # to compensate lack of available samples in some classes
				print('->->Compensating lack of samples ...')
				forbidden = copy.deepcopy(global_indices_in_unlabeled_set_for_pseudo_labeled)
				allowed = [a for a in idxs_unlabeled if a not in forbidden]
				global_indices_in_unlabeled_set_for_pseudo_labeled.extend(np.random.choice(allowed, n - so_far_selected, replace=False))

		return global_indices_in_unlabeled_set_for_pseudo_labeled
