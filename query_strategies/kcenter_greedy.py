import numpy as np
from .strategy import Strategy
import copy
import torch

class KCenterGreedy(Strategy):
	def __init__(self, X, Y, X_te, Y_te,dataset, method, idxs_lb, net, handler, args, cycle):
		super(KCenterGreedy, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):

		lb_flag = self.idxs_lb.copy()
		embedding = self.get_embedding_resnet(self.X, self.Y)
		embedding = embedding.numpy()
		embedding=np.around(embedding, decimals=2)

		from datetime import datetime

		print('calculate distance matrix')
		t_start = datetime.now()
		dist_mat = np.matmul(embedding, embedding.transpose())
		sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
		dist_mat *= -2
		dist_mat += sq
		dist_mat += sq.transpose()
		dist_mat = np.sqrt(dist_mat)
		print(datetime.now() - t_start)
		mat = dist_mat[~lb_flag, :][:, lb_flag]

		#===============UNBALANCED active set=========================
		if 'imbalance' in self.method:
			for i in range(n):
				if i%10 == 0:
					print('greedy solution {}/{}'.format(i, n))
				mat_min = mat.min(axis=1)
				q_idx_ = mat_min.argmax()
				q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
				lb_flag[q_idx] = True
				mat = np.delete(mat, q_idx_, 0)
				mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
			return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]

		# # #===============Class balanced active set=======================
		elif 'greedyOpt' in self.method:
			idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
			N = len(idxs_unlabeled)
			probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
			if self.dataset == 'cifar10':
				num_classes=10
				lam=5
			elif self.dataset == 'cifar100':
				num_classes=100
				lam=50

			# Adaptive counts of samples per cycle
			labeled_classes=self.Y[self.idxs_lb]
			_, counts = np.unique(labeled_classes, return_counts=True)
			class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
			class_share=class_threshold-counts
			samples_share= np.array([0 if c<0 else c for c in class_share]).reshape(num_classes,1)
			probs = np.array(probs)
			L1_LOSS=[]
			L1_DISTANCE=[]
			MAX_DIST=[]

			MAT=copy.deepcopy(mat)
			for lamda in [lam]:
				lb_flag = self.idxs_lb.copy()
				Q = copy.deepcopy(probs)
				z=np.zeros(N, dtype=bool)
				mat=MAT
				max_dist=0
				for i in range(n):
					if i%10 == 0:
						print('greedy solution {}/{}'.format(i, n))
					mat_min = mat.min(axis=1)
					SAMPLE_SHARE = np.tile(samples_share, N-i)
					P_Z = np.tile(np.matmul(np.transpose(probs), z), (N-i,1))
					X = SAMPLE_SHARE - np.transpose(Q) - np.transpose(P_Z)
					q_idx_= np.argmin(-mat_min + (lamda/num_classes) * np.linalg.norm(X,axis=0,ord=1))
					max_dist = max_dist + mat_min[q_idx_]
					z_idx = np.arange(N)[~z][q_idx_]
					z[z_idx] = True
					Q = np.delete(probs, np.where(z==1)[0], 0)
					q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
					lb_flag[q_idx] = True
					mat = np.delete(mat, q_idx_, 0)
					mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
				threshold=(2*n/num_classes)+ (self.cycle+1)*n/num_classes #cycle 1
				round=self.cycle+1 #cycle 1
				freq = torch.histc(torch.FloatTensor(self.Y[lb_flag]), bins=num_classes)
				L1_distance= (sum(abs(freq - threshold)) * num_classes / (2 * (2 * n + round * n) * (num_classes - 1))).item()
				L1_DISTANCE.append(L1_distance)
				L1_LOSS.append(sum(np.linalg.norm(X,axis=0,ord=1))/(N-n))
				MAX_DIST.append(max_dist)
				print('lamda = ',lamda)
				print('Maximum Distance Samples = ', max_dist)
				print ('L1 DISTANCE = ', L1_DISTANCE)
				print('L1 LOSS AVERAGE= ',L1_LOSS )
			print('L1 DISTANCE = ',L1_DISTANCE)
			print('L1 LOSS = ',L1_LOSS)
			print('Max Distanced samples = ', MAX_DIST)
			return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]

