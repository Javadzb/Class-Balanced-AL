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
import cvxpy as cp


class EntropySampling(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(EntropySampling, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)

		if self.dataset == 'cifar10':
			num_classes=10
		elif self.dataset == 'cifar100':
			num_classes=100

		if 'imbalance' in self.method:
			return idxs_unlabeled[U.sort()[1][:n]]

		#=======================================================================
		#            Optimization of maximum entropy with balancing
		#=======================================================================

		elif 'optimal' in self.method:
			b=n
			N=len(idxs_unlabeled)
			L1_DISTANCE=[]
			L1_Loss=[]
			ENT_Loss=[]
			probs = probs.numpy()
			U = U.numpy()
			# Adaptive counts of samples per cycle
			labeled_classes=self.Y[self.idxs_lb]
			_, counts = np.unique(labeled_classes, return_counts=True)
			class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
			class_share=class_threshold-counts
			samples_share= np.array([0 if c<0 else c for c in class_share]).reshape(num_classes,1)
			if self.dataset == 'cifar10':
				lamda=0.6
			elif self.dataset == 'cifar100':
				lamda=2

			for lam in [lamda]:

				z=cp.Variable((N,1),boolean=True)
				constraints = [sum(z) == b]
				cost = z.T @ U + lam * cp.norm1(probs.T @ z - samples_share)
				objective = cp.Minimize(cost)
				problem = cp.Problem(objective, constraints)
				problem.solve(solver=cp.GUROBI, verbose=True, TimeLimit=1000)
				print('Optimal value with gurobi : ', problem.value)
				print(problem.status)
				print("A solution z is")
				print(z.value.T)
				lb_flag = np.array(z.value.reshape(1, N)[0], dtype=bool)
				# -----------------Stats of optimization---------------------------------
				ENT_Loss.append(np.matmul(z.value.T, U))
				print('ENT LOSS= ', ENT_Loss)
				threshold = (2 * n / num_classes) + (self.cycle + 1) * n / num_classes
				round=self.cycle+1
				freq = torch.histc(torch.FloatTensor(self.Y[idxs_unlabeled[lb_flag]]), bins=num_classes)+torch.histc(torch.FloatTensor(self.Y[self.idxs_lb]), bins=num_classes)
				L1_distance = (sum(abs(freq - threshold)) * num_classes / (2 * (2 * n + round * n) * (num_classes - 1))).item()
				print('Lambda = ',lam)
				L1_DISTANCE.append(L1_distance)
				L1_Loss_term=np.linalg.norm(np.matmul(probs.T,z.value) - samples_share, ord=1)
				L1_Loss.append(L1_Loss_term)

			print('L1 Loss = ')
			for i in L1_Loss:
				print('%.3f' %i)
			print('L1_distance = ')
			for j in L1_DISTANCE:
				print('%.3f' % j)
			print('ENT LOSS = ')
			for k in ENT_Loss:
				print('%.3f' % k)
			return idxs_unlabeled[lb_flag]
