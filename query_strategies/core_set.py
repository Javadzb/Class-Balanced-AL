import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
import os
import pdb

class CoreSet(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle, tor=1e-4):
		super(CoreSet, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)
		self.tor = tor

	def query(self, n):
		lb_flag = self.idxs_lb.copy()
		embedding = self.get_embedding_resnet(self.X, self.Y)
		embedding = embedding.numpy()

		print('calculate distance matrix')
		t_start = datetime.now()
		dist_mat = np.matmul(embedding, embedding.transpose())
		sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
		dist_mat *= -2
		dist_mat += sq
		dist_mat += sq.transpose()
		dist_mat = np.sqrt(dist_mat)
		print(datetime.now() - t_start)

		print('calculate greedy solution')
		t_start = datetime.now()
		mat = dist_mat[~lb_flag, :][:, lb_flag]


		for i in range(n):
			if i%100 == 0:
				print('greedy solution {}/{}'.format(i, n))
			mat_min = mat.min(axis=1)
			q_idx_ = mat_min.argmax()
			q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
			lb_flag[q_idx] = True
			mat = np.delete(mat, q_idx_, 0)
			mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)

		print(datetime.now() - t_start)
		opt = mat.min(axis=1).max()

		bound_u = opt
		bound_l = opt/2.0
		delta = opt

		xx, yy = np.where(dist_mat <= opt)
		dd = dist_mat[xx, yy]

		lb_flag_ = self.idxs_lb.copy()
		subset = np.where(lb_flag_==True)[0].tolist()

		if not os.path.exists(self.dataset + '_results/' + self.method):
			os.mkdir(self.dataset + '_results/' + self.method)
		pickle.dump((xx.tolist(), yy.tolist(), dd.tolist(), subset, float(opt), n, self.n_pool), open(self.dataset+'_results/'+self.method+'/mip.pkl', 'wb'), 2)

		#import ipdb
		#ipdb.set_trace()
		# solving MIP
		# download Gurobi software from http://www.gurobi.com/
		# sh {GUROBI_HOME}/linux64/bin/gurobi.sh < core_set_sovle_solve.py


		#pdb.set_trace()

		os.system('python query_strategies/core_set_solve.py --dataset ' + self.dataset +' --method ' + self.method)

		#pdb.set_trace()
		# SEED IS UNDEFINED !

		sols = pickle.load(open('sols.pkl', 'rb'))

		if sols is None:
			q_idxs = lb_flag
		else:
			lb_flag_[sols] = True
			q_idxs = lb_flag_
		print('sum q_idxs = {}'.format(q_idxs.sum()))

		return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]
