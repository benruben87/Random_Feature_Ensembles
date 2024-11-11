#Library used to make datasets for regression and classification experiments.
import numpy as np
from collections import OrderedDict

import torch
import torchvision
import copy
from torchvision.transforms import Resize
torch.set_default_dtype(torch.float64)
import torchvision.transforms as transforms

import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta



from scipy.special import erfinv

tz = pytz.timezone('US/Eastern')
torch.set_default_dtype(torch.float64)

def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

def matrix_sqrt(A):
    L,V = torch.linalg.eigh(A)
    DiagL = torch.diag(L)
    return V @ torch.sqrt(DiagL) @ V.T

def split_data(P, x, y):
    indices = np.random.permutation(x.shape[0])
    trainIndices = indices[0:int(P)]
    testIndices = indices[int(P):]
    
    train_loader = (x[trainIndices], y[trainIndices])
    test_loader = (x[testIndices], y[testIndices])
    return (train_loader, test_loader)

def sample_data(P, x, y, random=False):
    assert P <= x.shape[0], "P must be less than or equal to the number of samples in x."

    indices = np.random.permutation(x.shape[0]) if random else np.arange(P)
    indices = indices[:P].tolist()  # Convert numpy array to list

    x_sampled = x[indices]

    if isinstance(y, list):
        y_sampled = [tensor[indices] for tensor in y]
    else:
        y_sampled = y[indices]

    return (x_sampled, y_sampled)


def shuffle_data(x, y):
    # Check if y is a list of tensors or a single tensor and get the first element for shape comparison
    y_first = y[0] if isinstance(y, list) else y
    assert x.shape[0] == y_first.shape[0], "Data and labels must have the same size."

    indices = np.random.permutation(x.shape[0])
    x_shuffled = x[indices]
    
    if isinstance(y, list):
        y_shuffled = [tensor[indices] for tensor in y]
    else:
        y_shuffled = y[indices]
    
    return x_shuffled, y_shuffled

def reduce_dim(M, x_train, x_test):
    indices = torch.randint(low=0, high=x_train.shape[1], size=(M,))
    #Downsample train and test matrices for easier computation.
    return x_train[:, indices], x_test[:, indices]

#P is the number of examples in the dataset
#w_star is the ground truth weights as a vector in the eigenbasis
#sigma_s is the eigenvalues of the covariance matrix (assume diagonal, no difference for random features.)
#sigma_eps is the scale of label noise
#C is number of output classes
def makeGaussianDataset_lin(P, w_star, sigma_s, sigma_eps = 0):
    sigma_s_root = torch.sqrt(sigma_s)
    M = sigma_s.shape[0]
    if len(w_star.shape) == 1:
        w_star = w_star.reshape(-1, 1)
    
    C = w_star.shape[1] #number of classes
    
    x = torch.randn(P, M, device = 'cuda')*torch.unsqueeze(sigma_s_root, 0)
    y = (x @ w_star) + sigma_eps*torch.randn(P, C, device = 'cuda')
    return (x.to('cuda'), y.to('cuda'))



def get_binarized_CIFAR10(data_root, class_groups, flatten=True, normalize=True):
    """
    Load and binarize CIFAR10 dataset into two groups of classes.

    Parameters:
    - data_root: location of the files.
    - class_groups: list of two lists, each containing class indices (0-9) to be grouped together.
        For example, [[0,1,7,8,9], [2,3,4,5,6]]
    - flatten: whether to flatten the images into vectors.
    - normalize: whether to normalize the images to have zero mean and unit variance.

    Returns:
    - X_train: Training data, shape (N_train, D)
    - y_train: Training labels, shape (N_train,)
    - X_test: Test data, shape (N_test, D)
    - y_test: Test labels, shape (N_test,)
    """

    # Load CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                           download=True, transform=transform)

    # Create a mapping from old class indices to new binary labels (0 or 1)
    group_labels = {}
    for group_label, class_list in enumerate(class_groups):
        for class_idx in class_list:
            group_labels[class_idx] = 2*group_label-1

    # Binarize training data
    X_train_list = []
    y_train_list = []
    for img, label in trainset:
        if label in group_labels:
            X_train_list.append(img.numpy())
            y_train_list.append(group_labels[label])

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    # Binarize test data
    X_test_list = []
    y_test_list = []
    for img, label in testset:
        if label in group_labels:
            X_test_list.append(img.numpy())
            y_test_list.append(group_labels[label])

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    if normalize:
        # Compute mean and std from training data
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8  # Avoid division by zero

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test


def get_binarized_MNIST(data_root, class_groups, flatten=True, normalize=True):
    """
    Load and binarize MNIST dataset into two groups of classes.

    Parameters:
    - data_root: location of the files.
    - class_groups: list of two lists, each containing class indices (0-9) to be grouped together.
        For example, [[0,1,2,3,4],[5,6,7,8,9]]
    - flatten: whether to flatten the images into vectors.
    - normalize: whether to normalize the images to have zero mean and unit variance.

    Returns:
    - X_train: Training data, shape (N_train, D)
    - y_train: Training labels, shape (N_train,)
    - X_test: Test data, shape (N_test, D)
    - y_test: Test labels, shape (N_test,)
    """

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root=data_root, train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=data_root, train=False,
                                         download=True, transform=transform)

    # Create a mapping from old class indices to new binary labels (-1 or +1)
    group_labels = {}
    for group_label, class_list in enumerate(class_groups):
        for class_idx in class_list:
            group_labels[class_idx] = 2 * group_label - 1  # So labels will be -1 and +1

    # Binarize training data
    X_train_list = []
    y_train_list = []
    for img, label in trainset:
        if label in group_labels:
            X_train_list.append(img.numpy())
            y_train_list.append(group_labels[label])

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    # Binarize test data
    X_test_list = []
    y_test_list = []
    for img, label in testset:
        if label in group_labels:
            X_test_list.append(img.numpy())
            y_test_list.append(group_labels[label])

    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)

    if normalize:
        # Compute mean and std from training data
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8  # Avoid division by zero

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test