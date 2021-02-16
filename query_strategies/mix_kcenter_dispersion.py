import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import pdb
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import cdist


class MixKcenterDispersion(Strategy):
	def __init__(self, X, Y, X_te, Y_te,dataset, idxs_lb, net, handler, args, cycle):
		super(MixKcenterDispersion, self).__init__(X, Y, X_te, Y_te, dataset, idxs_lb, net, handler, args, cycle)

	def query(self, n):
		lb_flag = self.idxs_lb.copy()
		embedding = self.get_embedding(self.X, self.Y)
		embedding = embedding.numpy()

		from datetime import datetime

		print('calculate distance matrix')
		# t_start = datetime.now()
		# dist_mat = np.matmul(embedding, embedding.transpose())
		# sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
		# dist_mat *= -2
		# dist_mat += sq
		# dist_mat += sq.transpose()
		# dist_mat = np.sqrt(dist_mat)
		# print(datetime.now() - t_start)

		# python library to compute distance matrix
		dist_mat = cdist(embedding, embedding, metric="cosine")

		mat = dist_mat[~lb_flag, :][:, lb_flag]

		# finding 2*budget centers with the order of importance
		length=4*n
		Q = np.zeros(length)
		for i in range(length):
			if i%10 == 0:
				print('greedy solution {}/{}'.format(i, n))
			mat_min = mat.min(axis=1)
			q_idx_ = mat_min.argmax()
			q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
			lb_flag[q_idx] = True
			mat = np.delete(mat, q_idx_, 0)
			mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
			Q[i] = q_idx


		# Dispersion
		# load inferred classes for sampling
		with open(self.dataset + '_results/' + type(self).__name__ + '/inferred_classes_cycle_' + str(self.cycle) + ".pkl",
					  "rb") as f:
				inferred_classes = pickle.load(f)

		# computing dispersion
		dispersion = torch.zeros(inferred_classes.size(0))
		correct_pred = 0
		for sample in range(0, inferred_classes.size(0)):
			class_mode, _ = torch.mode(inferred_classes[sample, :])
			dispersion[sample] = 1 - (
					sum(inferred_classes[sample, :] == class_mode.item()).to(dtype=torch.float) / float(self.args['n_epoch'] + 1))


		# if the dispersion of sample is among the highest entropies ...
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		for thresh in [0.4,0.5,0.6,0.7,0.8]:
			freq=np.histogram(dispersion)
			bin=np.true_divide(np.cumsum(freq[0]), sum(freq[0])) > thresh
			counter=0
			candidates = self.idxs_lb.copy()
			for cand in Q.astype(int):
				if dispersion[np.where(idxs_unlabeled == cand)] <= freq[1][np.where(bin == True)[0][0]]:
					#select the sample
					candidates[cand] = True
					counter=counter+1
				if counter == n:
					break
			if counter == n:
				break
		print('threshold= ', thresh)
		#pdb.set_trace()

		return np.arange(self.n_pool)[(self.idxs_lb ^ candidates)]
