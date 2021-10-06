import numpy as np
from dataset import get_dataset, get_handler
from torchvision import transforms
import torch
import os
import argparse
from Cutout.model.resnet import ResNet18


from query_strategies import RandomSampling, EntropySampling, \
                                KCenterGreedy, BALDDropout

parser = argparse.ArgumentParser(description='CNN')


parser.add_argument(
    '--method',
    default="none",
    help=
    'name of the acquisition method'
)

parser.add_argument(
    '--dataset',
    default="none",
    help=
    'name of the dataset'
)

parser.add_argument(
    '--imb_factor',
    type=float,
    default=1,
    help='Imbalance factor (0,1)'
)

parser.add_argument(
    '--imb_type',
    type=str,
    default='exp',
    help='step or exp'
)


inputs = parser.parse_args()
method = inputs.method
dataset= inputs.dataset
imb_factor=inputs.imb_factor
imb_type=inputs.imb_type

# Seed parameters
SEED = 1
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = True

NUM_INIT_LB=5000
NUM_QUERY = 2500
NUM_ROUND = 5

DATA_NAME = str.upper(dataset)
normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

args_pool = {'CIFAR10':
                {'n_epoch': 100, 'transform': transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 256, 'num_workers': 1}},
            'CIFAR100':
                {'n_epoch': 100, 'transform': transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
                'loader_tr_args': {'batch_size': 256, 'num_workers': 1},
                'loader_te_args': {'batch_size': 256, 'num_workers': 1}}}
args = args_pool[DATA_NAME]



# load dataset
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
n_pool = len(Y_tr)

# ------------------load the saved active set cycle 0 -------------------------------------
active_set = []
if dataset=='cifar10':
    with open(dataset + '_checkpoints_active_set/active_set_cycle_0.txt', 'r') as f:
        for line in f:
            active_set.append(int(line))
elif dataset=='cifar100':
    with open(dataset + '_checkpoints_active_set/active_set_cycle_0.txt', 'r') as f:
        for line in f:
            active_set.append(int(line))

#Making the dataset imbalanced
def get_img_num_per_cls(imb_type, imb_factor):

    if dataset == 'cifar10':
        num_classes = 10
        new_dataset_size=n_pool-len(active_set)
    elif dataset == 'cifar100':
        num_classes = 100
        new_dataset_size=n_pool-len(active_set)

    img_max = new_dataset_size / num_classes
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor ** (cls_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(num_classes // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(num_classes // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * num_classes)
    return img_num_per_cls

#-------------Create imbalanced dataset from original-----------------------------
def gen_imbalanced_data(img_num_per_cls, X_tr, Y_tr, imb_idxs):

    classes = np.unique(Y_tr)
    imb_flag = np.zeros(n_pool, dtype=bool)
    active_set_bool=np.zeros(n_pool, dtype=bool)
    active_set_bool[active_set]=True
    if imb_idxs==[]:
        for c in classes:
            idx = np.where(np.logical_and(np.array(Y_tr == c), active_set_bool == False))[0]
            np.random.shuffle(idx)
            selec_idx = idx[:img_num_per_cls[c]]
            imb_flag[selec_idx]=True
        # Save new long_tailed dataset info to the disk
        print('long-tailed dataset indices is created and saved to disk !')
        with open(dataset + '_checkpoints_active_set/long_tailed_dataset_IF_'+str(int(1/imb_factor))+'.txt', 'w') as f:
            for item in np.arange(n_pool)[imb_flag].tolist():
                f.write('{}\n'.format(item))
        with open(dataset + '_checkpoints_active_set/long_tailed_dataset_IF_' + str(int(1 / imb_factor)) + '.txt','a') as f:
            for item in active_set:
                f.write('{}\n'.format(item))
        imb_flag[imb_idxs] = True
        # Add active set of cycle 0 to the end of newly created dataset
        X_tr_new = np.concatenate((X_tr[np.arange(n_pool)[imb_flag]], X_tr[np.arange(n_pool)[active_set_bool]]), axis=0)
        Y_tr_new = np.concatenate((Y_tr[np.arange(n_pool)[imb_flag]], Y_tr[np.arange(n_pool)[active_set_bool]]), axis=0)
    else:
        print('loading the long-tailed dataset form disk')
        X_tr_new = np.concatenate((X_tr[imb_idxs[:len(imb_idxs)-NUM_INIT_LB]], X_tr[imb_idxs[-NUM_INIT_LB:]]), axis=0)
        Y_tr_new = np.concatenate((Y_tr[imb_idxs[:len(imb_idxs)-NUM_INIT_LB]], Y_tr[imb_idxs[-NUM_INIT_LB:]]), axis=0)

    return X_tr_new, Y_tr_new

# ------------------load the long_tailed dataset if exists----------------------------------
imb_idxs = []
if os.path.exists(dataset + '_checkpoints_active_set/long_tailed_dataset_IF_'+str(int(1/imb_factor))+'.txt'):
    with open(dataset + '_checkpoints_active_set/long_tailed_dataset_IF_'+str(int(1/imb_factor))+'.txt', 'r') as f:
        for line in f:
            imb_idxs.append(int(line))
#-----------------------------------------------------------------------------------

img_num_per_cls=get_img_num_per_cls(imb_type,imb_factor)
X_tr,Y_tr= gen_imbalanced_data(img_num_per_cls, X_tr, Y_tr, imb_idxs)

n_pool = len(Y_tr)
print('New dataset size = ', n_pool)
n_test = len(Y_te)


print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))


# initialization with labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
new_active_set = np.arange(n_pool)[-len(active_set):].tolist()
idxs_lb[new_active_set]=True
cycle = 0
handler = get_handler(DATA_NAME)

## Configure Resnet 18
if dataset == 'cifar10':
    num_classes = 10
elif dataset == 'cifar100':
    num_classes = 100

## Loading resent with cutout
net=ResNet18(num_classes=num_classes)


if 'RandomSampling' in method:
    strategy = RandomSampling(X_tr, Y_tr, X_te, Y_te, cycle, dataset, method, idxs_lb, net, handler, args)
elif 'EntropySampling' in method:
    strategy = EntropySampling(X_tr, Y_tr, X_te, Y_te, cycle, dataset, method, idxs_lb, net, handler, args)
elif 'KCenterGreedy' in method:
    strategy = KCenterGreedy(X_tr, Y_tr, X_te, Y_te, cycle, dataset, method, idxs_lb, net, handler, args)
elif 'BALDDropout' in method:
    strategy = BALDDropout(X_tr, Y_tr,X_te, Y_te, cycle, dataset, method, idxs_lb, net, handler, args, n_drop=10)
elif 'CoreSet' in method:
    strategy = CoreSet(X_tr, Y_tr, X_te, Y_te, cycle, dataset, method, idxs_lb, net, handler, args)

# print info
print(DATA_NAME)
print('SEED {}'.format(SEED))
print(type(strategy).__name__)

## round 0 accuracy
strategy.train()
#Evaluate trained model
P = strategy.test(X_te,Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print('Cycle 0 testing accuracy {}'.format(acc[0]))
