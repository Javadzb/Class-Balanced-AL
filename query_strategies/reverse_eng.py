import numpy as np
from .strategy import Strategy
from sklearn.neighbors import NearestNeighbors
import pickle
from datetime import datetime
import os
import pdb
from scipy.spatial.distance import cdist
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
#from fastcluster import linkage
from numpy import linalg as LA

class ReverseEng(Strategy):
    def __init__(self, X, Y, X_te, Y_te, dataset, idxs_lb, net, handler, args, cycle, tor=1e-4):
        super(ReverseEng, self).__init__(X, Y, X_te, Y_te, dataset, idxs_lb, net, handler, args, cycle)
        self.tor = tor

    def query(self, n):
        lb_flag = self.idxs_lb.copy()

        # Embedding of train samples
        embedding = self.get_embedding(self.X, self.Y)
        embedding = embedding.numpy()

        if self.dataset == 'cifar100':
            num_classes = 100
        elif self.dataset == 'cifar10':
            num_classes = 10
            classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


        # #TSNE representation of the train set
        # tsne = TSNE().fit_transform(embedding)
        # tx, ty = tsne[:, 0], tsne[:, 1]
        # tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        # ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
        # jet = plt.get_cmap('jet')
        # colors = iter(jet(np.linspace(0, 1, num_classes)))
        # for i in range(num_classes):
        #     y_i = self.Y == i
        #     plt.scatter(tx[y_i], ty[y_i], color=next(colors), s=1)
        #     #plt.scatter(tx[y_i], ty[y_i], color='r', s=2)
        #
        # plt.legend(loc=4)
        # plt.gca().invert_yaxis()
        # plt.savefig(os.path.join(self.dataset + '_results', 'train_tsne_cifar.jpg'), bbox_inches='tight')
        # np.save(os.path.join(self.dataset + '_results', 'tsne.npy'), tsne)
        # plt.close()


        # Shuffleing test samples and labels
        indices = np.arange(len(self.X_te))
        np.random.shuffle(indices)
        X_te = self.X_te[indices,:,:,:]
        Y_te = self.Y_te[indices]

        # Embedding of test samples
        embedding_test = self.get_embedding(X_te, Y_te)
        embedding_test = embedding_test.numpy()

        # Normalizing the test embedding
        norm_embedding_test = LA.norm(embedding_test, 2, axis=1)
        embedding_test = embedding_test/norm_embedding_test[:, None]

        # Random Embedding
        #embedding_test = np.random.rand(10000, 100)

        # python library to compute distance matrix
        #TEST_DIST_MATRIX = cdist(embedding_test, embedding_test, metric="euclidean")

        # coreset distance matrix
        print('calculate distance matrix')
        t_start = datetime.now()
        dist_mat = np.matmul(embedding_test, embedding_test.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(self.X_te), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        TEST_DIST_MATRIX = np.sqrt(dist_mat)
        print(datetime.now() - t_start)

        # Kmeans clustering
        #kmeans = KMeans(n_clusters=10, random_state=0).fit(embedding_test)


        # # Normalizing distances for every pair of points
        # SUM = np.sum(TEST_DIST_MATRIX, axis=1)
        # DENOM = np.transpose(np.zeros((len(TEST_DIST_MATRIX), len(TEST_DIST_MATRIX))) + SUM)
        # TEST_DIST_MATRIX = np.divide(TEST_DIST_MATRIX, DENOM)

        #heat_map = np.empty((0, 10000))

        #np.fill_diagonal(TEST_DIST_MATRIX, 0)

        # #------------------------------------------------------------------------------
        # #          CLUSTERING AND VISUALIZATION WITH HEAT MAP
        # #------------------------------------------------------------------------------
        #
        # methods = ["ward"]#, "average", "complete"]
        #
        # def seriation(Z, N, cur_index):
        #     '''
        #         input:
        #             - Z is a hierarchical tree (dendrogram)
        #             - N is the number of points given to the clustering process
        #             - cur_index is the position in the tree for the recursive traversal
        #         output:
        #             - order implied by the hierarchical tree Z
        #
        #         seriation computes the order implied by a hierarchical tree (dendrogram)
        #     '''
        #     if cur_index < N:
        #         return [cur_index]
        #     else:
        #         left = int(Z[cur_index - N, 0])
        #         right = int(Z[cur_index - N, 1])
        #         return (seriation(Z, N, left) + seriation(Z, N, right))
        #
        # for method in methods:
        #     print("Method:\t", method)
        #
        #     N = len(TEST_DIST_MATRIX)
        #     flat_dist_mat = squareform(TEST_DIST_MATRIX)
        #     res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
        #     res_order = seriation(res_linkage, N, N + N - 2)
        #     ordered_dist_mat = np.zeros((N, N))
        #     a, b = np.triu_indices(N, k=1)
        #     ordered_dist_mat[a, b] = TEST_DIST_MATRIX[[res_order[i] for i in a], [res_order[j] for j in b]]
        #     ordered_dist_mat[b, a] = ordered_dist_mat[a, b]
        #
        #     plt.pcolormesh(ordered_dist_mat)
        #     plt.xlim([0, N])
        #     plt.ylim([0, N])
        #     plt.show()
        #

        #------------------------------------------------------------------------------
        #           HISTOGRAM OF INTER/A CLASS DISTANCES
        # ------------------------------------------------------------------------------
        for i in range(len(classes)):

            #print(TEST_DIST_MATRIX[self.Y_te == i, :].shape)
            # heat_map = np.append(heat_map, TEST_DIST_MATRIX[self.Y_te == i, :], axis=0)

            # histogram plot of lower triangular elements of distance matrix
            class_mat = TEST_DIST_MATRIX[self.Y_te == 0, :][:, self.Y_te == i]
            print(class_mat.shape)
            plt.subplot(5, 2, i+1)
            # plt.hist(TEST_DIST_MATRIX[self.Y_te == 0, :][:, self.Y_te == 0][np.tril(np.ones(1000, dtype=bool), k=-1)], bins=1000, histtype='step',color='r')
            plt.hist(class_mat[np.tril(np.ones(1000, dtype=bool), k=0)], bins=1000, histtype='step')

        plt.show()
        pdb.set_trace()

        #dist_mat = cdist(embedding, embedding_test, metric='mahalanobis', VI=None)
        #dist_mat = cdist(embedding, embedding_test, metric="chebyshev")
        dist_mat = cdist(embedding, embedding_test, metric="cosine")
        #dist_mat = cdist(embedding, embedding_test, metric="euclidean")


        # closest_neighbor = np.zeros(dist_mat.shape[1])
        # for i in range(dist_mat.shape[1]):
        #     closest_neighbor[i] = np.argmin(dist_mat[:, i])

        # option_1 (using groundtruth)
        P = self.predict(X_te, Y_te)
        miss_classified_samples = P != Y_te

        # option_2 (using output probability)
        # probs = self.predict_prob(X_te, Y_te)
        # values, indices = torch.topk(probs, budget, largest=False)
        # miss_classified_samples = ...


        mat = dist_mat[~lb_flag, :][:, miss_classified_samples]
        lenght=n
        Q = np.zeros(lenght)
        for i in range(lenght):
            if i%10 == 0:
                print('closest unlabeled sample to wrongly classified test samples {}/{}'.format(i, n))
            # q_idx_ = np.argmin(mat[:, i])
            # q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]

            # sorting the distances increasing order
            min_idxs=np.argsort(mat[:, i])
            for j in min_idxs:
                q_idx = np.arange(self.n_pool)[~lb_flag][j]
                # selecting the closest sample with the same class
                if self.Y[q_idx] == Y_te[np.arange(10000)[miss_classified_samples][i]]:
                    lb_flag[q_idx] = True
                    # mat = np.delete(mat, q_idx_, 0)
                    mat = np.delete(mat, j, 0)
                    Q[i] = q_idx
                    break

        q_idxs = lb_flag
        print('sum q_idxs = {}'.format(q_idxs.sum()))

        print('same class samples in train-test = ', sum(self.Y[Q] == Y_te[np.arange(10000)[miss_classified_samples][0:lenght]]))


        #TSNE representation of test set

        mixed_embedding=np.append(embedding_test, embedding[Q.astype(int), :], axis=0)

        tsne = TSNE().fit_transform(mixed_embedding)

        tx, ty = tsne[:, 0], tsne[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

        #for i in range(10000):
        #    plt.scatter(tx[i], ty[i], color='b', s=1)

        wrongs = np.arange(10000)[miss_classified_samples][:n]
        # plot missclassified test samples
        plt.scatter(tx[wrongs], ty[wrongs], color='r', s=1)
        # plot closest unlabeled samples
        plt.scatter(tx[10000:len(mixed_embedding)], ty[10000:len(mixed_embedding)], color='g', s=1)

        #plt.legend(loc=4)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(self.dataset + '_results', 'test_tsne_cifar.jpg'), bbox_inches='tight')
        np.save(os.path.join(self.dataset + '_results', 'tsne.npy'), tsne)
        plt.close()

        pdb.set_trace()







        return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]

        # print('calculate distance matrix')
        # t_start = datetime.now()
        # dist_mat = np.matmul(embedding, embedding.transpose())
        # sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
        # dist_mat *= -2
        # dist_mat += sq
        # dist_mat += sq.transpose()
        # dist_mat = np.sqrt(dist_mat)
        # print(datetime.now() - t_start)


    #
    # print('calculate greedy solution')
    # t_start = datetime.now()
    # mat = dist_mat[~lb_flag, :][:, lb_flag]
    #
    # for i in range(n):
    # 	if i%10 == 0:
    # 		print('greedy solution {}/{}'.format(i, n))
    # 	mat_min = mat.min(axis=1)
    # 	q_idx_ = mat_min.argmax()
    # 	q_idx = np.arange(self.n_pool)[~lb_flag][q_idx_]
    # 	lb_flag[q_idx] = True
    # 	mat = np.delete(mat, q_idx_, 0)
    # 	mat = np.append(mat, dist_mat[~lb_flag, q_idx][:, None], axis=1)
    #
    # print(datetime.now() - t_start)
    #return np.arange(self.n_pool)[(self.idxs_lb ^ q_idxs)]


