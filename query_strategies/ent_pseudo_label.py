import numpy as np
import torch
from .strategy import Strategy
import copy
import pdb

class Ent_Pseudo_Label(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(Ent_Pseudo_Label, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query_pseudoLabel(self, n):

		pseudo_Y = copy.deepcopy(self.Y)
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		probs,P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)

		indices_of_pseudos=U.sort()[1][-n:]
		global_indices_in_unlabeled_set_for_pseudo_labeled=idxs_unlabeled[indices_of_pseudos]

		# computing the accuracy of pseudo labels
		pseudo_accuracy = np.zeros(len(global_indices_in_unlabeled_set_for_pseudo_labeled))
		for sample in range(len(indices_of_pseudos)):
			pseudo_accuracy[sample] = (P[indices_of_pseudos[sample]] == self.Y[global_indices_in_unlabeled_set_for_pseudo_labeled[sample]])

		print('\n')
		print('correct pseudo labels: ', sum(pseudo_accuracy))
		print('total pseudos: ', len(pseudo_accuracy))
		print('acurracy of pseudo label: ', float(sum(pseudo_accuracy))/float(len(pseudo_accuracy)))

		# Change gt labels with last label
		for sample in indices_of_pseudos:
			pseudo_Y[idxs_unlabeled[sample]] = P[sample]

		return idxs_unlabeled[U.sort()[1][-n:]], pseudo_Y

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs,P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]]