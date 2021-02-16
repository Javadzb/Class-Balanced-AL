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

		# b=n
		# x=np.zeros(len(idxs_unlabeled))
		# x[U.sort()[1][:n]]=1
		# unif=np.ones((num_classes,1))*b/num_classes
		# print(np.linalg.norm(unif - np.matmul(np.transpose(probs.numpy()), x).reshape(num_classes,1),1))
		# mat=(np.matmul(np.transpose(probs.numpy()), x).reshape(num_classes,1).reshape(num_classes)).sort()
		# pdb.set_trace()
		#
		# plt.plot(mat)
		# fname = os.path.join(self.dataset + '_results', self.method)
		# plt.savefig(fname + '/dist_cycle_'+str(self.cycle))
		# print(np.transpose(np.matmul(np.transpose(probs.numpy()), x).reshape(num_classes, 1)))
		if 'unbal' in self.method:
			return idxs_unlabeled[U.sort()[1][:n]]
		#return idxs_unlabeled[U.sort()[1][n:2*n]]
		#return idxs_unlabeled[U.sort()[1][2*n:3*n]]
		#return idxs_unlabeled[U.sort()[1][3*n:4*n]]
		#return idxs_unlabeled[U.sort()[1][4*n:5*n]]
		#return idxs_unlabeled[U.sort()[1][5*n:6*n]]
		#return idxs_unlabeled[U.sort()[1][6*n:7*n]]
		#return idxs_unlabeled[U.sort()[1][7*n:8*n]]
		#return idxs_unlabeled[U.sort()[1][8*n:9*n]]
		#return idxs_unlabeled[U.sort()[1][9*n:10*n]]
		#return idxs_unlabeled[U.sort()[1][10*n:11*n]]
		#return idxs_unlabeled[U.sort()[1][11*n:12*n]]
		#return idxs_unlabeled[U.sort()[1][12*n:13*n]]


		#return idxs_unlabeled[U.sort()[1][13*n:14*n]]
		#return idxs_unlabeled[U.sort()[1][14*n:15*n]]
		#return idxs_unlabeled[U.sort()[1][15*n:16*n]]
		#return idxs_unlabeled[U.sort()[1][16*n:17*n]]
		#return idxs_unlabeled[U.sort()[1][17*n:18*n]]#6


		##------------------------------Balancing----------------------------------
		# global_indices_in_unlabeled_set = []
		# for i in range(0, num_classes):
		# 	#samples_of_class_i = [j for j, value in enumerate(P) if self.Y[idxs_unlabeled[j]] == i] # balancing using GT
		# 	samples_of_class_i = [j for j, value in enumerate(P) if P[j] == i] #balancing using Model prediction
		# 	#_, indices = torch.topk(U[samples_of_class_i], int(n / num_classes), largest=False)
		#
		# 	if len(U[samples_of_class_i])< 4*int(n / num_classes):
		# 		_,indices=torch.topk(U[samples_of_class_i], int(n / num_classes), largest=False)
		# 	else:
		# 		_, indices_u = torch.topk(U[samples_of_class_i], 4 * int(n / num_classes), largest=False)
		# 		_, indices_l = torch.topk(U[samples_of_class_i], 3 * int(n / num_classes), largest=False)
		# 		indices = np.setdiff1d(indices_u, indices_l)
		#
		# 	global_indices_in_unlabeled_set.extend(idxs_unlabeled[np.array(samples_of_class_i)[indices]])
		# return global_indices_in_unlabeled_set
		#----------------------------------------------------------------------------------------------
		#
		# # ##------------------------------Weighted balancing---------------------------------------------
		# class_counts = np.zeros(num_classes)
		# k = int(0.2 * len(idxs_unlabeled))
		# _, INDICES = torch.topk(U, k, largest=True)
		# classes=P[INDICES]
		#
		# underpresented=[]
		# for c in range(num_classes):
		# 	class_counts[c] = np.count_nonzero(classes == c)
		# 	if class_counts[c] <= 5: # underrepresented classes
		# 		underpresented.append(c)
		# class_share = class_counts/k
		#
		# class_share[underpresented] = 1
		# inverse_counts = 1 / class_share
		# sum_inv_counts=sum(inverse_counts)
		#
		# for unrep in underpresented:
		# 	inverse_counts[unrep] = sum_inv_counts*sum(P==unrep)/n
		#
		# #samples_share = n * inverse_counts / sum(inverse_counts)
		#
		# # Sample share equalizer
		# # setting the threshold
		# samples_share=np.zeros(100)
		# sort_idx = np.argsort(class_counts)
		# th=100
		# #samples_share[sort_idx[0:40]]=62
		# #samples_share[sort_idx[0:50]]=50
		# samples_share[sort_idx[0:th]]=int(n/th)
		#
		# #Visualization
		# sort_idx=np.argsort(class_counts)
		# plt.plot(class_counts[sort_idx],color='coral', linewidth=4)
		#
		# gt_class_counts = np.zeros(num_classes)
		#
		# for c in range(num_classes):
		# 	gt_class_counts[c] = np.count_nonzero(self.Y[idxs_unlabeled[INDICES]] == c)
		# plt.plot(gt_class_counts[sort_idx],color='orange',linewidth=3, linestyle='dashed')
		#
		# plt.plot(samples_share[sort_idx],color='green', linewidth=4)
		#
		# global_indices_in_unlabeled_set = []
		#
		# for i in range(0, num_classes):
		# 	if samples_share[i]==0:
		# 		continue
		#
		# 	#samples_of_class_i = [j for j, value in enumerate(P) if self.Y[idxs_unlabeled[j]] == i] # balancing using GT
		# 	samples_of_class_i = [j for j, value in enumerate(P) if P[j] == i] #balancing using Model prediction
		# 	#_, indices = torch.topk(U[samples_of_class_i], int(samples_share[i])+ randint(0, 1), largest=False)
		# 	#_, indices = torch.topk(U[samples_of_class_i], int(samples_share[i]), largest=False)
		#
		# 	#if len(U[samples_of_class_i])< 4*int(samples_share[i]):
		# 	_,indices=torch.topk(U[samples_of_class_i], int(samples_share[i]), largest=False)
		# 	#else:
		# 	#	_, indices_u = torch.topk(U[samples_of_class_i], 4 * int(samples_share[i]), largest=False)
		# 	#	_, indices_l = torch.topk(U[samples_of_class_i], 3 * int(samples_share[i]), largest=False)
		# 	#	indices = np.setdiff1d(indices_u, indices_l)
		#
		# 	if len(indices)==0:
		# 		continue
		# 	elif len(indices) == 1:
		# 		global_indices_in_unlabeled_set.append(idxs_unlabeled[torch.IntTensor(samples_of_class_i)[indices]])
		# 	else:
		# 		#print(int(samples_share[i]))
		# 		global_indices_in_unlabeled_set.extend(idxs_unlabeled[torch.IntTensor(samples_of_class_i)[indices]])
		#
		# so_far_selected=len(global_indices_in_unlabeled_set)
		# #if len(global_indices_in_unlabeled_set)<n: # to compensate lack of available samples in some classes
		# #	forbidden=copy.deepcopy(global_indices_in_unlabeled_set)
		# #	allowed=[a for a in idxs_unlabeled if a not in forbidden]
		# #	global_indices_in_unlabeled_set.extend(np.random.choice(allowed,n-so_far_selected))
		#
		# if n<so_far_selected:
		# 	random.shuffle(global_indices_in_unlabeled_set)
		# 	del global_indices_in_unlabeled_set[0:so_far_selected-n]
		#
		# if so_far_selected<n: # to compensate lack of available samples in some classes
		# 	print('->->Compensating lack of samples ...')
		# 	forbidden=copy.deepcopy(global_indices_in_unlabeled_set)
		# 	allowed=[a for a in idxs_unlabeled if a not in forbidden]
		# 	global_indices_in_unlabeled_set.extend(np.random.choice(allowed,n-so_far_selected, replace=False))
		#
		# AS_class_counts=np.zeros(num_classes)
		# for c in range(num_classes):
		# 	AS_class_counts[c] = np.count_nonzero(self.Y[global_indices_in_unlabeled_set] == c)
		# plt.plot(AS_class_counts[sort_idx],color='limegreen', linewidth=3, linestyle='dashed')
		#
		# #saving class distribution figure
		# if not os.path.exists(self.dataset + '_results/' + self.method):
		# 	os.mkdir(self.dataset + '_results/' + self.method)
		# plt.savefig(self.dataset+'_results/' + self.method +'/active_set_dist_cycle_'+str(self.cycle))
		# plt.close()
		#
		# return global_indices_in_unlabeled_set
		#----------------------------------------------------------------------------------------------

		# ##------------------------------L1 balancing---------------------------------------------
		# labeled_classes=self.Y[self.idxs_lb]
		# _, counts = np.unique(labeled_classes, return_counts=True)
		# class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
		# class_share=class_threshold-counts
		# samples_share= [0 if c<0 else c for c in class_share]
		# global_indices_in_unlabeled_set = []
		# extra=[]
		# for i in range(0, num_classes):
		# 	if samples_share[i]==0:
		# 		continue
		# 	#samples_of_class_i = [j for j, value in enumerate(P) if self.Y[idxs_unlabeled[j]] == i] # balancing using GT
		# 	samples_of_class_i = [j for j, value in enumerate(P) if P[j] == i] #balancing using Model prediction
		#
		# 	if len(samples_of_class_i) < int(samples_share[i]):
		# 		samples_share[i] = len(samples_of_class_i)
		# 	else:
		# 		extra.extend(samples_of_class_i)
		#
		#
		# 	_, indices = torch.topk(U[samples_of_class_i], int(samples_share[i]), largest=False)
		#
		# 	if len(indices)==1:
		# 		global_indices_in_unlabeled_set.append(idxs_unlabeled[torch.IntTensor(samples_of_class_i)[indices]])
		# 	else:
		# 		global_indices_in_unlabeled_set.extend(idxs_unlabeled[torch.IntTensor(samples_of_class_i)[indices]])
		#
		# so_far_selected=len(global_indices_in_unlabeled_set)
		#
		# def intersection(lst1, lst2):
		# 	lst3 = [value for value in lst1 if value in lst2]
		# 	return lst3
		#
		# if n<so_far_selected:
		# 	print('so_far_selected= ',so_far_selected)
		# 	print('Remove few extra selected samples')
		# 	extra_global=idxs_unlabeled[extra]
		# 	to_be_removed=intersection(global_indices_in_unlabeled_set,extra_global)
		# 	random.shuffle(to_be_removed)
		# 	for i in range(so_far_selected-n):
		# 		global_indices_in_unlabeled_set.remove(to_be_removed[i])
		#
		# 	#random.shuffle(global_indices_in_unlabeled_set)
		# 	#del global_indices_in_unlabeled_set[0:so_far_selected-n]
		#
		# if so_far_selected<n: # to compensate lack of available samples in some classes
		# 	print('so_far_selected= ', so_far_selected)
		# 	print('->->Compensating lack of samples ...')
		# 	forbidden=copy.deepcopy(global_indices_in_unlabeled_set)
		# 	allowed=[a for a in idxs_unlabeled if a not in forbidden]
		# 	global_indices_in_unlabeled_set.extend(np.random.choice(allowed,n-so_far_selected, replace=False))
		# 	print('selected after compensation = ',len(global_indices_in_unlabeled_set))
		#
		# IDX=np.zeros((len(idxs_unlabeled),1),dtype=bool)
		# for i in global_indices_in_unlabeled_set:
		# 	IDX[np.where(idxs_unlabeled==i)[0][0]]=True
		# ENT=sum(U.numpy()[IDX.reshape(len(IDX))])
		# print('Entropy of L1 method =',ENT)
		#
		# L1_Loss_term = np.linalg.norm(np.matmul(probs.T, IDX) - n/num_classes, ord=1)
		# print('L1_Loss_term =', L1_Loss_term)
		#
		# return global_indices_in_unlabeled_set

		# # # # # #=======================================================================
		# # # # #            Optimization of maximum entropy with balancing
		# # # # # #=======================================================================
		elif 'optimal' in self.method:
			b=n
			N=len(idxs_unlabeled)
			L1_DISTANCE=[]
			L1_Loss=[]
			ENT_Loss=[]
			probs = probs.numpy()
			U = U.numpy()
			unif = np.ones((num_classes, 1)) * b / num_classes
			# Adaptive counts of samples per cycle
			labeled_classes=self.Y[self.idxs_lb]
			_, counts = np.unique(labeled_classes, return_counts=True)
			class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
			class_share=class_threshold-counts
			samples_share= np.array([0 if c<0 else c for c in class_share]).reshape(num_classes,1)

			#------------------Modify the norm part of objective to incorporate prior balancing
			#freq_np, _ = np.histogram(self.Y[self.idxs_lb], bins=num_classes)
			#q=freq_np
			#Q=np.tile(q, (N, 1))
			#R=probs / Q
			#rows_sum=R.sum(axis=1)
			#probs_modified=R/rows_sum[:,np.newaxis]
			#---------------------------------------------------------------
			# -----------------Toy sample share-----------------------------
			#x=10*np.ones((1,50))
			#y=40*np.ones((1,50))
			#samples_share=np.concatenate((x,y),axis=1).reshape(num_classes,1)
			#---------------------------------------------------------------

			#for lambd in np.arange(0, 3, 0.2):

			j = 0
			Pseudo_Label_mat = np.zeros((len(P), num_classes))
			for i in P:
				Pseudo_Label_mat[j, i] = 1
				j = j + 1

			#========================================================================
			#for lambd in [2]:
			#for lambd in np.arange(0, 3.2, 0.2): # CIFAR 100
			#for lambd in np.arange(0, 1.1, 0.1):  # CIFAR 10 # lamda= 0.4 (optimal)
			if self.dataset == 'cifar100':
				lamda=2
			elif self.dataset == 'cifar10':
				lamda=0.2#0.1#0.3#0.4#0.5#0.6

			#for lam in np.arange(0, 3.2, 0.1):#[lamda]:
			for lam in [lamda]:

				z=cp.Variable((N,1),boolean=True)
				## -----------------Initilize z-----------------------
				#IDX = np.argsort(U)[:n]
				#init_z=np.zeros((N,1))
				#init_z[IDX]=True
				#z.value=init_z
				#----------------------------------------------------
				constraints = [sum(z) == b]
				#cost=z.T @ U + lambd * cp.norm1(probs.T @ z - samples_share)
				#cost=z.T @ U + lambd * cp.sum(cp.pos(probs.T @ z - samples_share))
				cost = z.T @ U + lam * cp.norm1(probs.T @ z - samples_share)
				#cost = z.T @ U + lambd * cp.norm1(Pseudo_Label_mat.T @ z - samples_share)
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
				round=self.cycle+1 #cycle 1
				freq = torch.histc(torch.FloatTensor(self.Y[idxs_unlabeled[lb_flag]]), bins=num_classes)+torch.histc(torch.FloatTensor(self.Y[self.idxs_lb]), bins=num_classes)
				#L1_distance = (sum(abs(freq+(2*b/num_classes) - threshold)) * num_classes / (2 * (2 * n + i * n) * (num_classes - 1))).item()
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


		# # ===============GREEDY Optimized active set=======================
		elif 'greedyOpt' in self.method:
			idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
			N = len(idxs_unlabeled)

			probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
			if self.dataset == 'cifar10':
				num_classes = 10
				lamda=0.6

			elif self.dataset == 'cifar100':
				num_classes = 100
				lamda=2

			b=n
			N=len(idxs_unlabeled)
			probs = probs.numpy()
			U = U.numpy()
			unif = np.ones((num_classes, 1)) * b / num_classes
			# Adaptive counts of samples per cycle
			labeled_classes=self.Y[self.idxs_lb]
			_, counts = np.unique(labeled_classes, return_counts=True)
			class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
			class_share=class_threshold-counts
			samples_share= np.array([0 if c<0 else c for c in class_share]).reshape(num_classes,1)

			L1_LOSS = []
			L1_DISTANCE = []
			MAX_ENT = []
			U_COPY=copy.deepcopy(U)

			#for lamda in [0, 1, 2, 5, 10, 20, 50, 100, 200, 300]:
			#for lamda in np.arange(0, 3.2, 0.2):

			lb_flag = self.idxs_lb.copy()
			U = U_COPY
			Q = copy.deepcopy(probs)
			z = np.zeros(N, dtype=bool)
			max_ent=0

			for i in range(n):
				if i % 100 == 0:
					print('greedy solution {}/{}'.format(i, n))
				P_Z = np.tile(np.matmul(np.transpose(probs), z),(N - i, 1))
				SAMPLE_SHARE = np.tile(samples_share, N - i)
				X = SAMPLE_SHARE - np.transpose(Q) - np.transpose(P_Z)
				q_idx_ = np.argmin(U + lamda * np.linalg.norm(X, axis=0, ord=1))
				max_ent = max_ent + U[q_idx_]
				z_idx = np.arange(N)[~z][q_idx_]
				z[z_idx] = True
				Q = np.delete(probs, np.where(z == 1)[0], 0)
				q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
				lb_flag[q_idx] = True
				U = np.delete(U, q_idx_, 0)

			threshold = (2 * n / num_classes) + (self.cycle + 1) * n / num_classes  # cycle 1
			round = self.cycle + 1
			freq = torch.histc(torch.FloatTensor(self.Y[lb_flag]), bins=num_classes)
			L1_distance = (sum(abs(freq - threshold)) * num_classes / (2 * (2 * n + round * n) * (num_classes - 1))).item()
			L1_DISTANCE.append(L1_distance)
			L1_LOSS.append(sum(np.linalg.norm(X,axis=0,ord=1))/(N-n))
			MAX_ENT.append(max_ent)
			# print('lamda = ', lamda)
			# print('L1 DISTANCE = ', L1_DISTANCE)
			# print('L1 LOSS = ', L1_LOSS)
			# print('Max Entropy = ', max_ent)
			print('L1 DISTANCE = ', L1_DISTANCE)
			print('L1 LOSS = ', L1_LOSS)
			print('Max Entropy = ', MAX_ENT)
			return np.arange(self.n_pool)[(self.idxs_lb ^ lb_flag)]
