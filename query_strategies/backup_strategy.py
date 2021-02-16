import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from torch.optim.lr_scheduler import MultiStepLR
import pdb
import os
from model import get_net
import pickle


# class Strategy:
class Strategy(object):
    def __init__(self, X, Y, X_te, Y_te, cycle, dataset, method, idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.cycle = cycle
        self.dataset = dataset
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.method = method
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n):
        pass

    def query_pseudoLabel(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, loader_tr, optimizer):
        self.clf.train()
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)

            sys.stdout.write('\r')
            sys.stdout.write('Loss : %3f ' % loss.item())
            # sys.stdout.flush()

            loss.backward()
            optimizer.step()

        # Update optimizer step

    def train(self):
        n_epoch = self.args['n_epoch']

        # =======================================================================
        # uncomment for training from scratch
        # self.clf = get_net(str.upper(self.dataset), self.cycle).to(self.device)

        optimizer = optim.SGD(
            self.clf.parameters(),
            lr=0.02,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[60, 80], gamma=0.5)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])

        if 'Disp' in type(self).__name__:
            inferred_classes = torch.zeros(sum(~self.idxs_lb), n_epoch) - 1
            for epoch in range(1, n_epoch + 1):
                # sys.stdout.write('\r')
                sys.stdout.write('Epoch %3d' % epoch)
                # sys.stdout.flush()
                self._train(loader_tr, optimizer)
                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ', pg['lr'])

                P = self.predict(self.X[~self.idxs_lb], self.Y[~self.idxs_lb])
                inferred_classes[:, epoch - 1] = P
            # Save inferred classes
            fname = os.path.join(self.dataset + '_results', self.method)
            if not os.path.exists(fname):
                os.makedirs(fname)
            with open(fname + "/inferred_classes_cycle_" + str(self.cycle) + ".pkl", "wb") as f:
                pickle.dump(inferred_classes, f)

        else:
            for epoch in range(1, n_epoch + 1):
                # sys.stdout.write('\r')
                sys.stdout.write('Epoch %3d' % epoch)
                # sys.stdout.flush()
                self._train(loader_tr, optimizer)
                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ', pg['lr'])

        save_point = os.path.join(self.dataset + '_results', self.method, 'checkpoints')
        if not os.path.exists(save_point):
            os.makedirs(save_point)
        state = {
            'epoch': epoch,
        }
        torch.save(state, os.path.join(save_point, 'checkpoint_cycle_' + str(self.cycle) + '.t7'))
        print('checkpoint saved ...')

    def predict(self, X, Y):

        if len(X) == len(self.X_te) or len(X) == 0.9 * len(self.X):  # only for cycle 0
            self.clf = self.net.to(self.device)
            print('loading checkpoint of cycle 0 for testing...')
            net = torch.load(
                os.path.join(self.dataset + '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
            self.clf.load_state_dict(net['model_state_dict'])
        elif self.cycle > 0:
            print('evaluate the current model ...')
            print('loading checkpoint of current cycle ...')
            net = torch.load(
                os.path.join(self.dataset + '_results', self.method, 'checkpoints',
                             'checkpoint_cycle_' + str(self.cycle) + '.t7'))
            self.clf.load_state_dict(net['model_state_dict'])
        print('inference on ', len(X), ' samples')
        # --------------------------------------------------
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P

    def train_auxilary(self, idxs_pseudo_lb, pseudo_Y):
        n_epoch = self.args['n_epoch']

        # =======================================================================
        # uncomment for training from scratch
        # self.clf = get_net(str.upper(self.dataset), self.cycle).to(self.device)
        optimizer = optim.SGD(
            self.clf.parameters(),
            lr=0.02,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[60, 80], gamma=0.5)

        # load checkpoint of previous cycle
        self.clf = self.net.to(self.device)

        if self.cycle == 0:
            print('loading checkpoint of cycle 0')
            net = torch.load(
                os.path.join(self.dataset + '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
            self.clf.load_state_dict(net['model_state_dict'])
        else:
            print('loading checkpoint of cycle ' + str(self.cycle))
            net = torch.load(
                os.path.join(self.dataset + '_results', self.method, 'checkpoints',
                             'checkpoint_cycle_' + str(self.cycle) + '.t7'))
            self.clf.load_state_dict(net['model_state_dict'])

        temp = np.zeros(len(self.idxs_lb))
        temp[idxs_pseudo_lb] = True
        lb_pseudo_lb = np.logical_or(temp, self.idxs_lb)
        idxs_train = np.arange(self.n_pool)[lb_pseudo_lb]

        loader_tr = DataLoader(self.handler(self.X[idxs_train], pseudo_Y[idxs_train], transform=self.args['transform']),
                               shuffle=True, **self.args['loader_tr_args'])

        if 'Disp' in type(self).__name__:
            inferred_classes = torch.zeros(sum(~lb_pseudo_lb), n_epoch) - 1
            for epoch in range(1, n_epoch + 1):
                # sys.stdout.write('\r')
                sys.stdout.write('Epoch %3d' % epoch)
                # sys.stdout.flush()
                self._train(loader_tr, optimizer)
                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ', pg['lr'])

                P = self.predict(self.X[~lb_pseudo_lb], self.Y[~lb_pseudo_lb])
                inferred_classes[:, epoch - 1] = P
            # Save inferred classes
            fname = os.path.join(self.dataset + '_results', self.method)
            if not os.path.exists(fname):
                os.makedirs(fname)
            with open(fname + "/inferred_classes_cycle_" + str(self.cycle) + ".pkl", "wb") as f:
                pickle.dump(inferred_classes, f)

        else:
            for epoch in range(1, n_epoch + 1):
                # sys.stdout.write('\r')
                sys.stdout.write('Epoch %3d' % epoch)
                # sys.stdout.flush()
                self._train(loader_tr, optimizer)
                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ', pg['lr'])

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()

        return probs

    def get_embedding(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                               shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()

        return embedding

