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
import torchvision.models as models
import torch.nn as nn
from Cutout.model.resnet import ResNet18
from torchvision import transforms


#class Strategy:
class Strategy(object):
    def __init__(self, X, Y, X_te, Y_te, cycle , dataset, method,idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.cycle=cycle
        self.dataset=dataset
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.method=method
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
            #out, e1 = self.clf(x) #vgg
            out, e1 = self.clf(x) #resnet

            loss = F.cross_entropy(out, y)
            sys.stdout.write('\r')
            sys.stdout.write('Loss : %3f ' % loss.item())

            loss.backward()
            optimizer.step()



    def _train_weighted_loss(self, loader_tr, optimizer):

        self.clf.train()
        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'cifar100':
            num_classes = 100

        n=len(self.Y[self.idxs_lb])
        beta=(n-1.0)/n
        #beta=0.99
        _, img_num_per_class = np.unique(self.Y[self.idxs_lb], return_counts=True)

        def CB_loss(labels, logits, samples_per_cls, no_of_classes, beta):


            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)
            weights = weights / np.sum(weights) * num_classes

            #labels_one_hot = F.one_hot(labels, no_of_classes).float().cpu()
            #weights = torch.tensor(weights).float()
            #weights = weights.unsqueeze(0)
            #weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
            #weights = weights.sum(1)
            #weights = weights.unsqueeze(1)
            #weights = weights.repeat(1, no_of_classes)

            #pred = logits.softmax(dim=1)
            #cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot.cuda(), weight=torch.Tensor(weights).cuda())
            #pdb.set_trace()
            cb_loss = F.cross_entropy(logits, labels, weight=torch.Tensor(weights).cuda())

            return cb_loss

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            logits, e1 = self.clf(x) #resnet
            _, img_num_per_class = np.unique(self.Y[self.idxs_lb], return_counts=True)

            cb_loss = CB_loss(y, logits, img_num_per_class, num_classes, beta)

            sys.stdout.write('\r')
            sys.stdout.write('Loss : %3f ' % cb_loss.item())

            cb_loss.backward()
            optimizer.step()



    # def _train_ent_loss(self, loader_tr, optimizer):
    #     self.clf.train()
    #     for batch_idx, (x, y, idxs) in enumerate(loader_tr):
    #         x, y = x.to(self.device), y.to(self.device)
    #         optimizer.zero_grad()
    #         out, e1 = self.clf(x)
    #
    #         #loss = F.cross_entropy(out, y)
    #
    #         logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
    #         softmax = torch.nn.Softmax(dim=1).cuda()
    #         le = - torch.mean(torch.mul(softmax(out), logsoftmax(out)))
    #         loss = F.cross_entropy(out, y) + 0.8*le
    #
    #         sys.stdout.write('\r')
    #         sys.stdout.write('Loss : %3f ' % loss.item())
    #         #sys.stdout.flush()
    #
    #         loss.backward()
    #         optimizer.step()

    def train(self):

        # To initialize model
        #self.clf = get_net(str.upper(self.dataset), self.cycle).to(self.device)

        ## resnet finetuning
        #self.clf = self.net.to(self.device)

        ## resenet train from scratch
        # if self.dataset == 'cifar10':
        #     num_classes = 10
        # elif self.dataset == 'cifar100':
        #     num_classes = 100
        # net = models.resnet18()
        # net.avgpool = nn.AdaptiveAvgPool2d(1)
        # num_ftrs = net.fc.in_features
        # net.fc = nn.Linear(num_ftrs, num_classes)
        # self.net=net
        # self.clf = self.net.to(self.device)

        ## resenet from Cutout train from scratch
        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'cifar100':
            num_classes = 100
        net = ResNet18(num_classes=num_classes)
        self.net = net
        self.clf = self.net.to(self.device)


        n_epoch = self.args['n_epoch']
        optimizer = optim.SGD(
            self.clf.parameters(),
            lr=0.02,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[60, 80], gamma=0.5)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        print('number of training samples : ', len(idxs_train))

        loader_tr = DataLoader(self.handler(self.X[idxs_train], self.Y[idxs_train], transform=self.args['transform']),
                            shuffle=True, **self.args['loader_tr_args'])

        # Comment the following for training from scratch
        #if self.cycle==1:
        #    print('loading checkpoint of cycle 0')
        #    net = torch.load(
        #        os.path.join(self.dataset + '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
        #    self.clf.load_state_dict(net['model_state_dict'])
        #elif self.cycle > 1:
        #    print('loading checkpoint of cycle ' + str(self.cycle-1))
        #    net = torch.load(
        #        os.path.join(self.dataset + '_results', self.method, 'checkpoints', 'AUX_checkpoint_cycle_'+ str(self.cycle-1)+'.t7'))
        #    self.clf.load_state_dict(net['model_state_dict'])


        if 'Disp' in type(self).__name__:

            # SPEED UP
            #inferred_classes = torch.zeros(sum(~self.idxs_lb), 50) - 1 # forward pass only for last one fourth to speed up
            #inferred_classes = torch.zeros(sum(~self.idxs_lb), 25) - 1 # forward pass only for last one fourth to speed up
            inferred_classes = torch.zeros(sum(~self.idxs_lb), n_epoch) - 1

            for epoch in range(1, n_epoch+1):
                sys.stdout.write('Epoch %3d' % epoch)
                self._train(loader_tr, optimizer)
                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ',pg['lr'])

                # SPEED UP
                #if epoch > 75: # forward pass for epoch greather than 75 to speed up
                #    P=self.predict(self.X[~self.idxs_lb], self.Y[~self.idxs_lb])
                #    inferred_classes[:, epoch-76] = P #to speed up

            ##########################################################

                P=self.predict(self.X[~self.idxs_lb], self.Y[~self.idxs_lb])
                inferred_classes[:, epoch-1] = P
            #
            # # Save inferred classes
            fname = os.path.join(self.dataset + '_results', self.method)
            if not os.path.exists(fname):
                os.makedirs(fname)
            torch.save(inferred_classes,  fname + "/inferred_classes_cycle_" + str(self.cycle) + '.pt')

        else:
            for epoch in range(1, n_epoch+1):
                sys.stdout.write('Epoch %3d' % epoch)
                self._train(loader_tr, optimizer)
                #self._train_weighted_loss(loader_tr,optimizer)
                #self._train_ent_loss(loader_tr, optimizer)
                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ',pg['lr'])

        # if 'Random' in self.method:
        #     save_point = os.path.join(self.dataset + '_results', self.method, 'checkpoints')
        #     if not os.path.exists(save_point):
        #       os.makedirs(save_point)
        #     torch.save({
        #       'epoch': epoch,
        #       'model_state_dict': self.clf.state_dict(),
        #       'optimizer_state_dict': optimizer.state_dict()},
        #       os.path.join(save_point, 'checkpoint_cycle_' + str(self.cycle) + '.t7'))
        #     print('checkpoint saved ...')


    def train_auxilary(self, idxs_pseudo_lb, pseudo_Y):
        n_epoch = self.args['n_epoch']

        # To initialize model with pretrained imagenet
        self.clf = get_net(str.upper(self.dataset), self.cycle).to(self.device)

        optimizer = optim.SGD(
            self.clf.parameters(),
            lr=0.02,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
        scheduler = MultiStepLR(
            optimizer, milestones=[60, 80, 100, 120], gamma=0.5)

        ##Comment the following for training from scratch
        #if self.cycle <=1:
        #    print('loading checkpoint of cycle 0')
        #    net = torch.load(
        #        os.path.join(self.dataset + '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
        #    self.clf.load_state_dict(net['model_state_dict'])
        #else:
        #    print('loading checkpoint of cycle ' + str(self.cycle-1))
        #    net = torch.load(
        #        os.path.join(self.dataset + '_results', self.method, 'checkpoints', 'checkpoint_cycle_'+ str(self.cycle)+'.t7')) # for CEAL method put AUX and cycle-1
        #    self.clf.load_state_dict(net['model_state_dict'])


        # Adding pseudo labeled to labeled samples
        pseudo_idx = np.zeros(len(self.idxs_lb))
        pseudo_idx[idxs_pseudo_lb]=True
        lb_pseudo_lb= np.logical_or(pseudo_idx , self.idxs_lb)
        idxs_train = np.arange(self.n_pool)[lb_pseudo_lb]

        print('number of training samples : ', len(idxs_train))
        loader_tr = DataLoader(self.handler(self.X[idxs_train], pseudo_Y[idxs_train], transform=self.args['transform']),
                            shuffle=True, **self.args['loader_tr_args'])

        if 'Disp' in type(self).__name__:
            inferred_classes = torch.zeros(sum(~lb_pseudo_lb), 25) - 1 # to speed up
            #inferred_classes = torch.zeros(sum(~lb_pseudo_lb), n_epoch) - 1
            for epoch in range(1, n_epoch+1):
                #sys.stdout.write('\r')
                sys.stdout.write('Epoch %3d' % epoch)
                #sys.stdout.flush()
                self._train(loader_tr, optimizer)
                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ',pg['lr'])
                if epoch > 75: # forward pass for epoch greather than 75 to speed up
                    P=self.predict(self.X[~lb_pseudo_lb], self.Y[~lb_pseudo_lb])
                    inferred_classes[:, epoch-76] = P #to speed up
                #inferred_classes[:, epoch-1] = P
            # Save inferred classes
            fname = os.path.join(self.dataset + '_results', self.method)
            if not os.path.exists(fname):
                os.makedirs(fname)
            with open(fname + "/AUX_inferred_classes_cycle_" + str(self.cycle) + ".pkl", "wb") as f:
                pickle.dump(inferred_classes, f)

        else:
            for epoch in range(1, n_epoch+1):

                #sys.stdout.write('\r')
                sys.stdout.write('Epoch %3d' % epoch)
                self._train(loader_tr, optimizer)
                #self._train_ent_loss(loader_tr, optimizer)

                scheduler.step(epoch)
                for pg in optimizer.param_groups:
                    print('lr = ',pg['lr'])

        #save_point = os.path.join(self.dataset + '_results', self.method, 'checkpoints')
        #if not os.path.exists(save_point):
        #    os.makedirs(save_point)
        #torch.save({
        #    'epoch': epoch,
        #    'model_state_dict': self.clf.state_dict(),
        #    'optimizer_state_dict': optimizer.state_dict()},
        #    os.path.join(save_point, 'AUX_checkpoint_cycle_' + str(self.cycle) + '.t7'))

        #print('checkpoint saved ...')



    def predict(self, X, Y):

        #if self.cycle > 0:
        #    print('evaluate the current model ...')
            #print('loading checkpoint of current cycle ...')
            #net = torch.load(
            #    os.path.join(self.dataset + '_results', self.method, 'checkpoints', 'checkpoint_cycle_'+ str(self.cycle)+'.t7'))
            #self.clf.load_state_dict(net['model_state_dict'])
        # elif sum(self.idxs_lb)== 0.1*len(self.X): # only for cycle 0
        #     self.clf = self.net.to(self.device)
        #     print('loading checkpoint of cycle 0 for testing...')
        #     net = torch.load(
        #         os.path.join(self.dataset+ '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
        #     self.clf.load_state_dict(net['model_state_dict'])

        print('inference on ', len(X), ' samples')

        #--------------------------------------------------
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                #out, e1 = self.clf(x) #vgg
                out, e1= self.clf(x) # resnet
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P

    def test(self, X, Y):


        if self.cycle > 0 :
            print('test the current model ...')
            #print('loading checkpoint of current cycle ...')
            #net = torch.load(
            #    os.path.join(self.dataset + '_results', self.method, 'checkpoints', 'checkpoint_cycle_'+ str(self.cycle)+'.t7'))
            #self.clf.load_state_dict(net['model_state_dict'])
        elif self.cycle == 0: # only for cycle 0
            self.clf = self.net.to(self.device)
            print('loading checkpoint of cycle 0 for testing...')
            net = torch.load(
                os.path.join(self.dataset+ '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
                #os.path.join(self.dataset + '_results/RandomSampling', 'checkpoints', 'checkpoint_cycle_3.t7'))

            self.clf.load_state_dict(net['model_state_dict'])

        ############################################################################
        print('inference on ', len(X), ' samples')
        ##--------------------------------------------------
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        #loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args'])
        loader_te = DataLoader(self.handler(X,Y, transform= transforms.Compose([transforms.ToTensor(), normalize])))
        self.clf.eval()
        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                #out, e1 = self.clf(x) #vgg
                out, e1 = self.clf(x) #resnet

                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
        return P

    def predict_prob(self, X, Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        #TT=[1, 5, 10]
        TT=1
        if self.dataset=='cifar100':
            num_classes=100
        elif self.dataset=='cifar10':
            num_classes=10
        probs = torch.zeros([len(Y), num_classes])
        P = torch.zeros(len(Y), dtype=torch.int32)
        #P = torch.zeros(len(Y), dtype=int)
        logits= torch.zeros([len(Y), num_classes])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x) #Resnet
                out=out/TT
                #-----------------------
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()
                #--------------------
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
                logits[idxs]=out.cpu()

        #return probs, P, logits
        return probs, P


    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i+1, n_drop))
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
            print('n_drop {}/{}'.format(i+1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    #out, e1 = self.clf(x) # vgg
                    out, e1 = self.clf(x) # resnet
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

    def get_embedding_resnet(self,X,Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        embedding = torch.zeros([len(Y), 512])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                embedding[idxs] = e1.cpu()
        return embedding


    def test_prior_balanced(self, X_te,Y_te,Y_tr ):


        if self.cycle > 0 :
            print('test the current model ...')
        elif self.cycle == 0: # only for cycle 0
            self.clf = self.net.to(self.device)
            print('loading checkpoint of cycle 0 for testing...')
            net = torch.load(
                os.path.join(self.dataset+ '_checkpoints_active_set', 'checkpoint_cycle_0.t7'))
            self.clf.load_state_dict(net['model_state_dict'])

        print('inference on ', len(Y_te), ' samples')

        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        #loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform']), shuffle=False, **self.args['loader_te_args'])
        loader_te = DataLoader(self.handler(X_te,Y_te, transform= transforms.Compose([transforms.ToTensor(), normalize])))
        self.clf.eval()
        P = torch.zeros(len(Y_te), dtype=Y_te.dtype)

        if self.dataset == 'cifar10':
            num_classes = 10
        elif self.dataset == 'cifar100':
            num_classes = 100

        freq_np,_ = np.histogram(Y_tr[self.idxs_lb], bins=num_classes)
        freq=torch.from_numpy(freq_np).cuda()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                balanced_prob = prob / freq
                _, predicted = torch.max(balanced_prob, 1)
                P[idxs] = predicted.cpu()

        return P
