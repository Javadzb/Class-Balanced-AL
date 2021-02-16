import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans
import pdb
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import copy
import torch
import random
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import plot_confusion_matrix


class KMeansSampling(Strategy):
	def __init__(self, X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle):
		super(KMeansSampling, self).__init__(X, Y, X_te, Y_te, dataset, method, idxs_lb, net, handler, args, cycle)

	# def query(self, n):
	# 	idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
	# 	embedding = self.get_embedding_resnet(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
	# 	embedding = embedding.numpy()
	#
	# 	print('Starting Kmeans clustering on embeddings ...')
	# 	cluster_learner = KMeans(n_clusters=n)
	# 	cluster_learner.fit(embedding)
	#
	# 	cluster_idxs = cluster_learner.predict(embedding)
	#
	# 	centers = cluster_learner.cluster_centers_[cluster_idxs]
	# 	dis = (embedding - centers)**2
	# 	dis = dis.sum(axis=1)
	# 	q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])
	#
	# 	return idxs_unlabeled[q_idxs]



	#==========================================================================
	# seeding with mean of labeled samples per class
	def query(self, n):
		embedding = self.get_embedding_resnet(self.X, self.Y)
		embedding = embedding.numpy()

		if self.dataset == 'cifar10':
			num_classes=10
		elif self.dataset == 'cifar100':
			num_classes=100

		avg_emb_labeled_samples=np.zeros((num_classes,embedding.shape[1]))
		for c in range(num_classes):
			idxs=[i for i in range(embedding.shape[0]) if self.Y[i]==c and self.idxs_lb[i]]
			avg_emb_labeled_samples[c]= np.mean(embedding[idxs,:],axis=0)

		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		embedding=embedding[idxs_unlabeled,:]

		print('Starting Kmeans clustering on embeddings ...')
		cluster_learner = KMeans(init = avg_emb_labeled_samples, n_clusters=num_classes)
		cluster_learner.fit(embedding)
		cluster_idxs = cluster_learner.predict(embedding)

		## ==============Mapping cluster idxs and check the accuracy of clustering and compare with model predictions============
		# saved_cluster_idxs=copy.deepcopy(cluster_idxs)
		# probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		# ACC_model= float(sum(np.array(P) == self.Y[idxs_unlabeled]))/len(idxs_unlabeled)
		# print('-----------------------------------------')
		# print('ARI model vs GT = ',adjusted_rand_score(self.Y[idxs_unlabeled],P))
		# print('Acc. of model = ', ACC_model)
		# CM_gt_model=contingency_matrix(self.Y[idxs_unlabeled], P)
		# #arg_max_CM_gt_model=np.argmax(contingency_matrix(self.Y[idxs_unlabeled], P) axis=1)
		# #plt.subplot(131)
		# #plt.xlabel('Model prediction')
		# #plt.ylabel('Groundtruth')
		# #plt.pcolormesh(CM_gt_model)
		# #plot_confusion_matrix(CM_gt_model)
		# #arg_max_CM_gt_model=np.argmax(contingency_matrix(self.Y[idxs_unlabeled], P), axis=1)
		# print('-----------------------------------------')
		# print('ARI kmeans vs GT = ',adjusted_rand_score(self.Y[idxs_unlabeled],cluster_idxs))
		# print('Acc. of clustering = ', float(sum(cluster_idxs == np.array(self.Y[idxs_unlabeled])))/len(idxs_unlabeled))
		# #arg_max_CM_gt_kmeans=np.argmax(contingency_matrix(self.Y[idxs_unlabeled], cluster_idxs), axis=1)
		# CM_gt_kmeans=contingency_matrix(self.Y[idxs_unlabeled], cluster_idxs)
		# #plt.subplot(132)
		# #plt.xlabel('Cluster idx')
		# #plt.ylabel('Groundtruth')
		# #plt.pcolormesh(CM_gt_kmeans)
		# #plot_confusion_matrix(CM_gt_kmeans)
		# print('-----------------------------------------')
		# cluster_idxs=saved_cluster_idxs
		# print('ARI model vs Kmeans = ',adjusted_rand_score(P,cluster_idxs))
		# CM_model_kmeans = contingency_matrix(P, cluster_idxs)
		# #arg_max_CM_model_kmeans=np.argmax(contingency_matrix(P, cluster_idxs), axis=1)
		# #plt.subplot(133)
		# #plt.xlabel('Cluster idx')
		# #plt.ylabel('Model prediction')
		# #plt.pcolormesh(CM_model_kmeans)
		# #plot_confusion_matrix(CM_model_kmeans)
		#
		# #plt.show()
		#=================================================================================
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs, P = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)

		agree=0
		disagree=0
		both_correct=0
		agree_both_wrong=0
		model_correct=0
		cluster_correct=0
		disagree_both_wrong=0
		ent_1=[]
		ent_2=[]
		ent_3=[]
		ent_4=[]
		ent_5=[]
		for i in np.arange(len(P)):
			if P[i]==cluster_idxs[i]:
				agree+=1
				if P[i]==self.Y[idxs_unlabeled][i]:
					both_correct+=1
					ent_1.append(U[i].item())
				else:
					agree_both_wrong+=1
					ent_2.append(U[i].item())
			else:
				disagree+=1
				if P[i] == self.Y[idxs_unlabeled][i]:
					model_correct+=1
					ent_3.append(U[i].item())
				elif cluster_idxs[i]== self.Y[idxs_unlabeled][i]:
					cluster_correct+=1
					ent_4.append(U[i].item())
				else:
					disagree_both_wrong+=1
					ent_5.append(U[i].item())


		print('agree =',agree)
		print('both_correct= ',both_correct)
		print('both_wrong= ',agree_both_wrong)
		print('disagree= ',disagree)
		print('model_correct= ',model_correct)
		print('cluster_correct= ',cluster_correct)
		print('disagree_both_wrong= ',disagree_both_wrong)

		print(str(np.mean(ent_1)) +'+-'+ str(np.std(ent_1)))
		print(str(np.mean(ent_2)) +'+-'+ str(np.std(ent_2)))
		print(str(np.mean(ent_3)) +'+-'+ str(np.std(ent_3)))
		print(str(np.mean(ent_4)) +'+-'+ str(np.std(ent_4)))
		print(str(np.mean(ent_5)) +'+-'+ str(np.std(ent_5)))



		#pdb.set_trace()

		#for i in np.arange(num_classes):
		#	cluster_idxs[cluster_idxs==i]=CM_model_kmeans

		#print('Matching of clustering and model predicton = ', float(sum(cluster_idxs == np.array(P)))/len(idxs_unlabeled))


		#choosing randomly from samples that model prediction and clustering disagree
		#return np.random.choice(idxs_unlabeled[cluster_idxs != np.array(P)], n, replace=False) # disgaree
		#-----------------------------------------------------------------------------------------------
		#log_probs = torch.log(probs)
		#U = (probs*log_probs).sum(1)

		disagreement=cluster_idxs != np.array(P)
		IDXS = U[disagreement].sort()[1]
		return idxs_unlabeled[np.arange(len(U))[disagreement][IDXS][-n:]]

		# ##------------------------------Kmeans L1 balancing---------------------------------------------

		# labeled_classes=self.Y[self.idxs_lb]
		# _, counts = np.unique(labeled_classes, return_counts=True)
		# class_threshold=int((2*n+(self.cycle+1)*n)/num_classes)
		# class_share=class_threshold-counts
		# samples_share= [0 if c<0 else c for c in class_share]
		# global_indices_in_unlabeled_set = []
		# for i in range(0, num_classes):
		# 	if samples_share[i]==0:
		# 		continue
		# 	#samples_of_class_i = [j for j, value in enumerate(P) if self.Y[idxs_unlabeled[j]] == i] # balancing using GT
		# 	samples_of_class_i = [j for j, value in enumerate(P) if P[j] == i and cluster_idxs[j] != np.array(P)[j] ] #balancing using Model prediction
		#
		# 	if len(samples_of_class_i) < int(samples_share[i]):
		# 		samples_share[i] = len(samples_of_class_i)
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
		# print('selected after compensation = ',len(global_indices_in_unlabeled_set))
		#
		# return global_indices_in_unlabeled_set


		#===============================================================================================
		####	Standard Kmeans?
		# centers = cluster_learner.cluster_centers_[cluster_idxs]
		# dis = (embedding - centers) ** 2
		# #dis = cdist(embedding, centers, metric="cosine")
		# dis = dis.sum(axis=1)
		# class_share = int(n / num_classes)
		#
		# # for max distance:  [-class_share:], for min distance [:class_share]
		# q_idxs = np.array([[np.arange(embedding.shape[0])[cluster_idxs == c][dis[cluster_idxs == c].argsort()[-class_share:]]] for c in range(num_classes)])
		#
		# #if so_far_selected < n:  # to compensate lack of available samples in some classes
		# #	print('->->Compensating lack of samples ...')
		# # 	forbidden=copy.deepcopy(global_indices_in_unlabeled_set)
		# # 	allowed=[a for a in idxs_unlabeled if a not in forbidden]
		# # 	global_indices_in_unlabeled_set.extend(np.random.choice(allowed,n-so_far_selected, replace=False))
		#
		# return idxs_unlabeled[np.reshape(q_idxs,-1)]



