import numpy as np
import pdb
from .strategy import Strategy
import torch
import random
import copy

class RandomSampling(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(RandomSampling, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):

		return np.random.choice(np.where(self.idxs_lb==0)[0], n , replace=False)

		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		if self.dataset == 'cifar10':
			num_classes=10
		elif self.dataset == 'cifar100':
			num_classes=100
		# #-------------------------GT L1 random---------------------------------------------
		# selected_samples=[]
		# for i in range(0, num_classes):
		# 	samples_of_class_i = [j for j in range(len(idxs_unlabeled)) if self.Y[idxs_unlabeled[j]] == i] # balancing using GT
		# 	#print('class '+str(i)+ '= '+str(len(samples_of_class_i)))
		# 	selected_samples.extend(idxs_unlabeled[np.random.choice(samples_of_class_i, int(n / num_classes), replace=False)])
		# return selected_samples

		# ##------------------------------L1 balancing---------------------------------------------
		# _, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		# labeled_classes = self.Y[self.idxs_lb]
		# _, counts = np.unique(labeled_classes, return_counts=True)
		# class_threshold = int((2 * n + (self.cycle + 1) * n) / num_classes)
		# class_share = class_threshold - counts
		# samples_share = [0 if c < 0 else c for c in class_share]
		# global_indices_in_unlabeled_set = []
		# extra = []
		# for i in range(0, num_classes):
		# 	if samples_share[i] == 0:
		# 		continue
		# 	samples_of_class_i = [j for j, value in enumerate(P) if P[j] == i]  # balancing using Model prediction
		#
		# 	if len(samples_of_class_i) < int(samples_share[i]):
		# 		samples_share[i] = len(samples_of_class_i)
		# 	else:
		# 		extra.extend(samples_of_class_i)
		#
		# 	indices=np.random.choice(len(samples_of_class_i), int(samples_share[i]), replace=False)
		#
		# 	if len(indices) == 1:
		# 		global_indices_in_unlabeled_set.append(idxs_unlabeled[torch.IntTensor(samples_of_class_i)[indices]])
		# 	else:
		# 		global_indices_in_unlabeled_set.extend(idxs_unlabeled[torch.IntTensor(samples_of_class_i)[indices]])
		#
		# so_far_selected = len(global_indices_in_unlabeled_set)
		#
		# def intersection(lst1, lst2):
		# 	lst3 = [value for value in lst1 if value in lst2]
		# 	return lst3
		#
		# if n < so_far_selected:
		# 	print('so_far_selected= ', so_far_selected)
		# 	print('Remove few extra selected samples')
		# 	extra_global = idxs_unlabeled[extra]
		# 	to_be_removed = intersection(global_indices_in_unlabeled_set, extra_global)
		# 	random.shuffle(to_be_removed)
		# 	for i in range(so_far_selected - n):
		# 		global_indices_in_unlabeled_set.remove(to_be_removed[i])
		#
		# if so_far_selected < n:  # to compensate lack of available samples in some classes
		# 	print('so_far_selected= ', so_far_selected)
		# 	print('->->Compensating lack of samples ...')
		# 	forbidden = copy.deepcopy(global_indices_in_unlabeled_set)
		# 	allowed = [a for a in idxs_unlabeled if a not in forbidden]
		# 	global_indices_in_unlabeled_set.extend(np.random.choice(allowed, n - so_far_selected, replace=False))
		# 	print('selected after compensation = ', len(global_indices_in_unlabeled_set))
		#
		# return global_indices_in_unlabeled_set