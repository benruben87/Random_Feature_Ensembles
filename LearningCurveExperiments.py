# LearningCurveExperiments.py

import numpy as np
import torch
import EnsembleRFs  # Ensure this module is accessible
import EnsRFTheory  # Ensure this module is accessible
import auxFuncs     # Ensure this module is accessible

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pickle

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

#Auxilary function:

def makeGaussianParams(D, alpha, r, device = 'cuda'):
    # Generate sigma_s and w_star
    sigma_s = torch.tensor([n ** (-alpha) for n in range(1, D+1)], dtype=torch.float64, device=device)
    sigma_s = sigma_s / torch.sum(sigma_s)
    w_star = torch.tensor([n ** (-1/2*(1 - alpha + 2 * alpha * r)) for n in range(1, D+1)], dtype=torch.float64, device=device)
    w_star = w_star / torch.linalg.norm(w_star*sigma_s**(1/2))
    return sigma_s, w_star

#Numerical Experiment Functions:


def train_random_feature_models_fixN(X_train, y_train, X_test, y_test, num_trials, KVals, N, lamVals, P_list, ensErrFuncs, nonlinearity=None, Fvar = 1):
    """
    Train random feature models of varying size K while keeping M=N*K fixed.

    Parameters:
    - X_train: Training data, shape (N_train, D)
    - y_train: Training labels, shape (N_train,)
    - X_test: Test data, shape (N_test, D)
    - y_test: Test labels, shape (N_test,)
    - num_trials: Number of trials
    - KVals: List of K values to test
    - N: Features per ensemble member (fixed)
    - lamVals: List of lambda values
    - P_list: List of training sample sizes P
    - nonlinearity: Nonlinearity function (e.g., torch.relu)
    - Fvar: variance of the random weights.  If variance = None, sets variance to 1/N for N = features per ensemble member.
    
    Returns:
    - test_errors: Array of test errors, shape (num_trials, len(KVals), len(P_list), len(lamVals))
    """

    device = X_train.device
    D = X_train.shape[1]  # Data dimension

    numEnsErrFuncs = len(ensErrFuncs)

    numKVals = len(KVals)
    numLamVals = len(lamVals)
    num_PVals = len(P_list)
    test_errors = np.zeros((num_trials, numKVals, num_PVals, numLamVals, numEnsErrFuncs))

    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}")
        for KInd, K in enumerate(KVals):
            M = int(N*K)
            # Create K random feature matrices, each of size N x D
            F_list = EnsembleRFs.create_projection_matrices(K, D, N, variance=Fvar)

            for PInd, P in enumerate(P_list):
                # Sample P training samples
                indices = torch.randperm(X_train.shape[0])[:P]
                X_train_P = X_train[indices]
                y_train_P = y_train[indices]#.unsqueeze(1)

                train_loader = (X_train_P, y_train_P)
                test_loader = (X_test, y_test)#.unsqueeze(1))
                
                test_err = EnsembleRFs.error_sweep_lam(train_loader, test_loader, F_list, lamVals, ensErrFuncs, nonlinearity=nonlinearity, calc_train_err=False, cum = False)
                
                test_errors[trial, KInd, PInd, :, :] = test_err[:, :].cpu().numpy()
                

        print(f'Completed trial {trial+1}/{num_trials}')

    return test_errors

def train_random_feature_models_fixM(X_train, y_train, X_test, y_test, num_trials, KVals, M, lamVals, P_list, ensErrFuncs, nonlinearity=None, Fvar = 1):
    """
    Train random feature models of varying size K while keeping M=N*K fixed.

    Parameters:
    - X_train: Training data, shape (N_train, D)
    - y_train: Training labels, shape (N_train,)
    - X_test: Test data, shape (N_test, D)
    - y_test: Test labels, shape (N_test,)
    - num_trials: Number of trials
    - KVals: List of K values to test
    - M: Total number of features (fixed)
    - lamVals: List of lambda values
    - P_list: List of training sample sizes P
    - nonlinearity: Nonlinearity function (e.g., torch.relu)
    - Fvar: variance of the random weights.  If variance = None, sets variance to 1/N for N = features per ensemble member.
    
    Returns:
    - test_errors: Array of test errors, shape (num_trials, len(KVals), len(P_list), len(lamVals))
    """

    device = X_train.device
    D = X_train.shape[1]  # Data dimension

    numEnsErrFuncs = len(ensErrFuncs)

    numKVals = len(KVals)
    numLamVals = len(lamVals)
    num_PVals = len(P_list)
    test_errors = np.zeros((num_trials, numKVals, num_PVals, numLamVals, numEnsErrFuncs))

    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}")
        for KInd, K in enumerate(KVals):
            N = int(M / K)
            N = max(N, 1)  # Ensure N is at least 1
            # Create K random feature matrices, each of size N x D
            F_list = EnsembleRFs.create_projection_matrices(K, D, N, variance=Fvar)

            for PInd, P in enumerate(P_list):
                # Sample P training samples
                indices = torch.randperm(X_train.shape[0])[:P]
                X_train_P = X_train[indices]
                y_train_P = y_train[indices]#.unsqueeze(1)

                train_loader = (X_train_P, y_train_P)
                test_loader = (X_test, y_test)#.unsqueeze(1))
                
                test_err = EnsembleRFs.error_sweep_lam(train_loader, test_loader, F_list, lamVals, ensErrFuncs, nonlinearity=nonlinearity, calc_train_err=False, cum = False)
                
                test_errors[trial, KInd, PInd, :, :] = test_err[:, :].cpu().numpy()
                

        print(f'Completed trial {trial+1}/{num_trials}')

    return test_errors

def train_random_feature_models_M_K(X_train, y_train, X_test, y_test, num_trials, MVals, KVals, lamVals, P, ensErrFuncs, nonlinearity=None, Fvar=1):
    """
    Train random feature models over varying M and K values, keeping P fixed.
    For each combination of M and K, enforce N = round(M / K)
    
    Parameters:
    - X_train: Training data, shape (N_train, D)
    - y_train: Training labels, shape (N_train,)
    - X_test: Test data, shape (N_test, D)
    - y_test: Test labels, shape (N_test,)
    - num_trials: Number of trials
    - MVals: List of total number of features M to test
    - KVals: List of K values to test
    - lamVals: List of lambda values
    - P: Fixed training sample size
    - ensErrFuncs: ensembling and error functions as list of tuples.
    - nonlinearity: Nonlinearity function (e.g., torch.relu)
    - Fvar: Variance of the random weights. If None, sets variance to 1/N for N = features per ensemble member.
    
    Returns:
    - test_errors: Array of test errors, shape (num_trials, len(MVals), len(KVals), len(lamVals))
    """
    device = X_train.device
    D = X_train.shape[1]  # Data dimension
    
    numEnsErrFuncs = len(ensErrFuncs)
    
    numMVals = len(MVals)
    numKVals = len(KVals)
    numLamVals = len(lamVals)
    test_errors = np.zeros((num_trials, numMVals, numKVals, numLamVals, numEnsErrFuncs))
    
    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}")
        for MInd, M in enumerate(MVals):
            for KInd, K in enumerate(KVals):
                N = max(int(round(M / K)), 1)  # Ensure N is at least 1
                # Create K random feature matrices, each of size N x D
                F_list = EnsembleRFs.create_projection_matrices(K, D, N, variance=Fvar)
    
                # Sample P training samples
                indices = torch.randperm(X_train.shape[0])[:P]
                X_train_P = X_train[indices]
                y_train_P = y_train[indices]
    
                train_loader = (X_train_P, y_train_P)
                test_loader = (X_test, y_test)
    
                # Use the error_sweep_lam function from EnsembleRFs
                test_err = EnsembleRFs.error_sweep_lam(
                    train_loader, test_loader, F_list, lamVals, ensErrFuncs, nonlinearity=nonlinearity, calc_train_err=False
                )
                # test_err shape: (num_lamVals, num_ensErrFuncs, K)
                # We take the cumulative mean over K ensemble members
                test_errors[trial, MInd, KInd, :, :] = test_err.cpu().numpy()
        print(f'Completed trial {trial+1}/{num_trials}')
    
    return test_errors

def train_random_feature_models_N_K(X_train, y_train, X_test, y_test, num_trials, NVals, KVals, lamVals, P, ensErrFuncs, nonlinearity=None, Fvar=1):
    """
    Train random feature models over varying N and K values, keeping P fixed.
    For each combination of N and K, enforce M = M*K
    
    Parameters:
    - X_train: Training data, shape (N_train, D)
    - y_train: Training labels, shape (N_train,)
    - X_test: Test data, shape (N_test, D)
    - y_test: Test labels, shape (N_test,)
    - num_trials: Number of trials
    - NVals: List of total number of features M to test
    - KVals: List of K values to test
    - lamVals: List of lambda values
    - P: Fixed training sample size
    - ensErrFuncs: ensembling and error functions as list of tuples.
    - nonlinearity: Nonlinearity function (e.g., torch.relu)
    - Fvar: Variance of the random weights. If None, sets variance to 1/N for N = features per ensemble member.
    
    Returns:
    - test_errors: Array of test errors, shape (num_trials, len(MVals), len(KVals), len(lamVals))
    """
    device = X_train.device
    D = X_train.shape[1]  # Data dimension
    
    numEnsErrFuncs = len(ensErrFuncs)
    
    numNVals = len(NVals)
    numKVals = len(KVals)
    numLamVals = len(lamVals)
    test_errors = np.zeros((num_trials, numNVals, numKVals, numLamVals, numEnsErrFuncs))
    
    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}")
        for NInd, N in enumerate(NVals):
            for KInd, K in enumerate(KVals):
                M = int(N*K) 
                # Create K random feature matrices, each of size N x D
                F_list = EnsembleRFs.create_projection_matrices(K, D, N, variance=Fvar)
    
                # Sample P training samples
                indices = torch.randperm(X_train.shape[0])[:P]
                X_train_P = X_train[indices]
                y_train_P = y_train[indices]
    
                train_loader = (X_train_P, y_train_P)
                test_loader = (X_test, y_test)
    
                # Use the error_sweep_lam function from EnsembleRFs
                test_err = EnsembleRFs.error_sweep_lam(
                    train_loader, test_loader, F_list, lamVals, ensErrFuncs, nonlinearity=nonlinearity, calc_train_err=False
                )
                # test_err shape: (num_lamVals, num_ensErrFuncs, K)
                # We take the cumulative mean over K ensemble members
                test_errors[trial, NInd, KInd, :, :] = test_err.cpu().numpy()
        print(f'Completed trial {trial+1}/{num_trials}')
    
    return test_errors


def train_random_feature_models_ell(X_train, y_train, X_test, y_test, num_trials, ell_list, M_list, P, lamVals, nonlinearity=None, Fvar=1):
    """
    Train random feature models for varying M and ell while keeping P fixed.

    Parameters:
    - X_train: Training data, shape (N_train, D)
    - y_train: Training labels, shape (N_train,)
    - X_test: Test data, shape (N_test, D)
    - y_test: Test labels, shape (N_test,)
    - num_trials: Number of trials
    - ell_list: List of ell values between 0 and 1
    - M_list: List of total number of features M
    - P: Training sample size (fixed)
    - lamVals: List of lambda values
    - nonlinearity: Nonlinearity function (e.g., torch.relu)
    - Fvar: Variance of the random weights.

    Returns:
    - test_errors: Array of test errors, shape (num_trials, len(M_list), len(ell_list), len(lamVals))
    """

    device = X_train.device
    D = X_train.shape[1]  # Data dimension

    ensFunc = auxFuncs.mean  # Ensemble function
    errFunc = auxFuncs.SquareError  # Error function
    ensErrFuncs = [(ensFunc, errFunc)]

    num_ellVals = len(ell_list)
    num_MVals = len(M_list)
    numLamVals = len(lamVals)
    test_errors = np.zeros((num_trials, num_MVals, num_ellVals, numLamVals))

    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}")
        for MInd, M in enumerate(M_list):
            print(f"Starting M = {M}")
            for ellInd, ell in enumerate(ell_list):
                K = max(round(M ** (1-ell)), 1)
                N = max(round(M / K), 1)
                # Ensure N and K are at least 1
                M_actual = K * N
                # Create K random feature matrices, each of size N x D
                F_list = EnsembleRFs.create_projection_matrices(K, D, N, variance=Fvar)
                # Sample P training samples
                indices = torch.randperm(X_train.shape[0])[:P]
                X_train_P = X_train[indices]
                y_train_P = y_train[indices]

                train_loader = (X_train_P, y_train_P)
                test_loader = (X_test, y_test)

                # Use the error_sweep_lam function from EnsembleRFs
                test_err = EnsembleRFs.error_sweep_lam(train_loader, test_loader, F_list, lamVals, ensErrFuncs,
                                                       nonlinearity=nonlinearity, calc_train_err=False)
                # test_err shape: (num_lamVals, num_ensErrFuncs, K)
                # We take the cumulative mean over K ensemble members
                test_errors[trial, MInd, ellInd, :] = test_err[:, 0, -1].cpu().numpy()

        print(f'Completed trial {trial+1}/{num_trials}')

    return test_errors

def train_random_feature_models_ell_synthetic(
    num_trials, ell_list, M_list, P, lamVals,
    D, alpha, r, Fvar, sigma_eps=0.0
):
    """
    Train linear random feature models for varying M and ell while keeping P fixed.
    Generates synthetic Gaussian datasets at each trial to optimize memory usage.

    Parameters:
    - num_trials: Number of trials
    - ell_list: List of ell values between 0 and 1
    - M_list: List of total number of features M
    - P: Training sample size (fixed)
    - lamVals: List of lambda values
    - D: Dimensionality of data
    - alpha: Exponent for the covariance eigenvalues decay
    - r: Exponent for the ground truth weights decay
    - Fvar: Variance of the random weights.
    - sigma_eps: Noise variance in the labels

    Returns:
    - test_errors: Array of test errors, shape (num_trials, len(M_list), len(ell_list), len(lamVals))
    """
    import torch
    import numpy as np
    import DatasetMaker
    import EnsembleRFs
    import auxFuncs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate sigma_s and w_star once, since they are constant across trials
    sigma_s, w_star = makeGaussianParams(D, alpha, r)

    ensFunc = auxFuncs.mean  # Ensemble function
    errFunc = auxFuncs.SquareError  # Error function
    ensErrFuncs = [(ensFunc, errFunc)]

    num_ellVals = len(ell_list)
    num_MVals = len(M_list)
    numLamVals = len(lamVals)
    test_errors = np.zeros((num_trials, num_MVals, num_ellVals, numLamVals))

    for trial in range(num_trials):
        print(f"Starting trial {trial+1}/{num_trials}")

        # Generate training and test datasets for this trial
        X_train, y_train = DatasetMaker.makeGaussianDataset_lin(
            P, w_star, sigma_s, sigma_eps=sigma_eps
        )

        X_train = X_train.to(device)
        y_train = y_train.to(device)

        D = X_train.shape[1]  # Data dimension

        for MInd, M in enumerate(M_list):
            print(f"Starting M = {M}")
            for ellInd, ell in enumerate(ell_list):
                K = max(round(M ** (1 - ell)), 1)
                N = max(round(M / K), 1)
                # Ensure N and K are at least 1
                M_actual = K * N
                # Create K random feature matrices, each of size N x D
                F_list = EnsembleRFs.create_projection_matrices(K, D, N, variance=Fvar)

                # Use all P training samples (no need to sample since we generate P samples)
                train_loader = (X_train, y_train)
                #test_loader = (X_test, y_test)

                # Use the error_sweep_lam function from EnsembleRFs
                test_err = EnsembleRFs.error_sweep_lam(
                    train_loader, None, F_list, lamVals, ensErrFuncs,
                    calc_train_err=False,
                    useLinEg=True, sigma_s = sigma_s, w_star = w_star, sigma_eps = sigma_eps 
                )
                
                # test_err shape: (num_lamVals, num_ensErrFuncs, K)
                # We take the cumulative mean over K ensemble members
                test_errors[trial, MInd, ellInd, :] = test_err.cpu().numpy()

                # Delete F_list to free up memory
                del F_list
                torch.cuda.empty_cache()

        # Delete datasets to free up memory before the next trial
        del X_train, y_train
        torch.cuda.empty_cache()

        print(f'Completed trial {trial+1}/{num_trials}')

    return test_errors

## Theory Computations:

def compute_theoretical_learning_curves_fixM(Sigma, wbar, KVals, lamVals, P_list, M, sigma_eps = 0, returnBiasVariance = False):
    """
    Compute theoretical learning curves for varying K, lambda, and P.

    Parameters:
    - Sigma: Array of eigenvalues
    - wbar: Array of wbar coefficients
    - KVals: List of K values
    - lamVals: List of lambda values
    - P_list: List of training sample sizes P
    - M: Total number of features (fixed)
    - returnBiasVariance: When True, returns tensors with Bias and Variance contributions to error as well.

    Returns:
    - theoretical_errors: Array of theoretical errors, shape (len(KVals), len(P_list), len(lamVals))
    """

    numKVals = len(KVals)
    numLamVals = len(lamVals)
    num_PVals = len(P_list)
    theoretical_errors = np.zeros((numKVals, num_PVals, numLamVals))
    
    if returnBiasVariance:
        theoretical_bias = np.zeros((numKVals, num_PVals, numLamVals))
        theoretical_variance = np.zeros((numKVals, num_PVals, numLamVals))

    for KInd, K in enumerate(KVals):
        N = int(M / K)
        N = max(N, 1)
        for PInd, P in enumerate(P_list):
            for lamInd, lam in enumerate(lamVals):
                if returnBiasVariance:
                    cur_error, cur_bias, cur_var = EnsRFTheory.Eg_K(
                        Sigma, 
                        wbar, 
                        lam, 
                        P, 
                        N, 
                        K, 
                        sigma_eps = sigma_eps, 
                        returnBiasVariance = True)
                    theoretical_errors[KInd, PInd, lamInd] = cur_error
                    theoretical_bias[KInd, PInd, lamInd] = cur_bias
                    theoretical_variance[KInd, PInd, lamInd] = cur_var
                else:
                    theoretical_errors[KInd, PInd, lamInd] = EnsRFTheory.Eg_K(
                        Sigma, 
                        wbar, 
                        lam, 
                        P, 
                        N, 
                        K, 
                        sigma_eps = sigma_eps, 
                        returnBiasVariance = False)

    if returnBiasVariance:
        return theoretical_errors, theoretical_bias, theoretical_variance
    
    return theoretical_errors

def compute_theoretical_learning_curves_fixN(Sigma, wbar, KVals, lamVals, P_list, N, sigma_eps = 0, returnBiasVariance = False):
    """
    Compute theoretical learning curves for varying K, lambda, and P.

    Parameters:
    - Sigma: Array of eigenvalues
    - wbar: Array of wbar coefficients
    - KVals: List of K values
    - lamVals: List of lambda values
    - P_list: List of training sample sizes P
    - N: features per ensemble member (fixed)
    - returnBiasVariance: When True, returns tensors with Bias and Variance contributions to error as well.

    Returns:
    - theoretical_errors: Array of theoretical errors, shape (len(KVals), len(P_list), len(lamVals))
    """

    assert returnBiasVariance==False, "Haven't implemented returnBiasVariance for fixed N."
    
    numKVals = len(KVals)
    numLamVals = len(lamVals)
    num_PVals = len(P_list)
    theoretical_errors = np.zeros((numKVals, num_PVals, numLamVals))

    for KInd, K in enumerate(KVals):
        for PInd, P in enumerate(P_list):
            for lamInd, lam in enumerate(lamVals):
                theoretical_errors[KInd, PInd, lamInd] = EnsRFTheory.Eg_K(Sigma, wbar, lam, P, N, K, sigma_eps = sigma_eps)

    return theoretical_errors

def compute_theoretical_learning_curves_M_K(Sigma, wbar, MVals, KVals, lamVals, P, sigma_eps=0, returnBiasVariance = False):
    """
    Compute theoretical learning curves for varying M and K, lambda, with fixed P.
    
    Parameters:
    - Sigma: Array of eigenvalues
    - wbar: Array of wbar coefficients
    - MVals: List of total number of features M
    - KVals: List of K values
    - lamVals: List of lambda values
    - P: Fixed training sample size
    - sigma_eps: Noise level in the task
    - returnBiasVariance: When True, returns tensors with Bias and Variance contributions to error as well.  Variance is divided by K to give the ensemble variance.
    
    Returns:
    - theoretical_errors: Array of theoretical errors, shape (len(MVals), len(KVals), len(lamVals))
    """
    numMVals = len(MVals)
    numKVals = len(KVals)
    numLamVals = len(lamVals)
    theoretical_errors = np.zeros((numMVals, numKVals, numLamVals))
    
    if returnBiasVariance:
        theoretical_bias = np.zeros((numMVals, numKVals, numLamVals))
        theoretical_variance = np.zeros((numMVals, numKVals, numLamVals))
    
    for MInd, M in enumerate(MVals):
        for KInd, K in enumerate(KVals):
            N = max(int(round(M / K)), 1)  # Ensure N is at least 1
            for lamInd, lam in enumerate(lamVals):
                if returnBiasVariance:
                    cur_errors, cur_bias, cur_variance = EnsRFTheory.Eg_K(
                        Sigma, wbar, lam, P, N, K, sigma_eps=sigma_eps)
                    theoretical_errors[MInd, KInd, lamInd] = cur_errors
                    theoretical_bias[MInd, KInd, lamInd] = cur_bias
                    theoretical_variance[MInd, KInd, lamInd] = cur_variance          
                else:
                    theoretical_errors[MInd, KInd, lamInd] = EnsRFTheory.Eg_K(
                        Sigma, wbar, lam, P, N, K, sigma_eps=sigma_eps)
    
    if returnBiasVariance:
        return theoretical_errors, theoretical_bias, theoretical_variance
    
    return theoretical_errors

def compute_theoretical_learning_curves_N_K(Sigma, wbar, NVals, KVals, lamVals, P, sigma_eps=0, returnBiasVariance = False):
    """
    Compute theoretical learning curves for varying N and K, lambda, with fixed P.
    
    Parameters:
    - Sigma: Array of eigenvalues
    - wbar: Array of wbar coefficients
    - NVals: List of features per ensemble member
    - KVals: List of K values
    - lamVals: List of lambda values
    - P: Fixed training sample size
    - sigma_eps: Noise level in the task
    - returnBiasVariance: When True, returns tensors with Bias and Variance contributions to error as well.  Variance is divided by K to give the ensemble variance.
    
    Returns:
    - theoretical_errors: Array of theoretical errors, shape (len(NVals), len(KVals), len(lamVals))
    """
    numNVals = len(NVals)
    numKVals = len(KVals)
    numLamVals = len(lamVals)
    theoretical_errors = np.zeros((numNVals, numKVals, numLamVals))
    
    if returnBiasVariance:
        theoretical_bias = np.zeros((numNVals, numKVals, numLamVals))
        theoretical_variance = np.zeros((numNVals, numKVals, numLamVals))
    
    for NInd, N in enumerate(NVals):
        for KInd, K in enumerate(KVals):
            for lamInd, lam in enumerate(lamVals):
                if returnBiasVariance:
                    cur_errors, cur_bias, cur_variance = EnsRFTheory.Eg_K(
                        Sigma, wbar, lam, P, N, K, sigma_eps=sigma_eps)
                    theoretical_errors[NInd, KInd, lamInd] = cur_errors
                    theoretical_bias[NInd, KInd, lamInd] = cur_bias
                    theoretical_variance[NInd, KInd, lamInd] = cur_variance          
                else:
                    theoretical_errors[NInd, KInd, lamInd] = EnsRFTheory.Eg_K(
                        Sigma, wbar, lam, P, N, K, sigma_eps=sigma_eps)
    
    if returnBiasVariance:
        return theoretical_errors, theoretical_bias, theoretical_variance
    
    return theoretical_errors

def compute_theoretical_learning_curves_ell(Sigma, wbar, ell_list, lamVals, P, N_list, sigma_eps=0):
    """
    Compute theoretical learning curves for varying M and ell while keeping P fixed.

    Parameters:
    - Sigma: Array of eigenvalues
    - wbar: Array of wbar coefficients
    - ell_list: List of ell values between 0 and 1
    - lamVals: List of lambda values
    - P: Training sample size (fixed)
    - M_list: List of total number of features M
    - sigma_eps: Noise variance

    Returns:
    - theoretical_errors: Array of theoretical errors, shape (len(M_list), len(ell_list), len(lamVals))
    """

    num_ellVals = len(ell_list)
    num_MVals = len(M_list)
    numLamVals = len(lamVals)
    theoretical_errors = np.zeros((num_MVals, num_ellVals, numLamVals))

    for MInd, M in enumerate(M_list):
        for ellInd, ell in enumerate(ell_list):
            K = M ** (1-ell)
            N = M ** ell
            for lamInd, lam in enumerate(lamVals):
                theoretical_errors[MInd, ellInd, lamInd] = EnsRFTheory.Eg_K(Sigma, wbar, lam, P, N, K, sigma_eps=sigma_eps)
                
    return theoretical_errors


## Data Analysis Functions:

def compute_optimal_errors(mean_test_errors, std_test_errors, test_errors_theory, KVals, P_list):
    """
    Compute the optimal test errors for both numerical and theoretical results.
    If mean_test_errors is None, only the optimal theoretical test errors are computed.

    Parameters:
    - mean_test_errors: Mean of numerical test errors across trials (or None if not available).
    - std_test_errors: Standard deviation of numerical test errors across trials (or None if mean_test_errors is None).
    - test_errors_theory: Theoretical test errors.
    - KVals: List of K values used in the experiment.
    - P_list: List of P values used in the experiment.

    Returns:
    If mean_test_errors is provided:
    - optimal_errors_numerical: Optimal numerical test errors.
    - std_errors_numerical: Standard deviation at the optimal point.
    - optimal_errors_theory: Optimal theoretical test errors.

    If mean_test_errors is None:
    - optimal_errors_theory: Optimal theoretical test errors.
    """
    
    optimal_errors_theory = np.zeros((len(KVals), len(P_list)))

    # If mean_test_errors is None, return only theoretical optimal errors
    if mean_test_errors is None:
        for KInd in range(len(KVals)):
            for PInd in range(len(P_list)):
                min_index = np.argmin(test_errors_theory[KInd, PInd, :])
                optimal_errors_theory[KInd, PInd] = test_errors_theory[KInd, PInd, min_index]
        return None, None, optimal_errors_theory
    
    #if test_errors_theory is None, return only numerical errors
    if test_errors_theory is None:
        optimal_errors_numerical = np.zeros((len(KVals), len(P_list)))
        std_errors_numerical = np.zeros((len(KVals), len(P_list)))
        for KInd in range(len(KVals)):
            for PInd in range(len(P_list)):
                mean_error = mean_test_errors[KInd, PInd, :]
                std_error = std_test_errors[KInd, PInd, :]
                min_index = np.argmin(mean_error)
                optimal_errors_numerical[KInd, PInd] = mean_error[min_index]
                std_errors_numerical[KInd, PInd] = std_error[min_index]

        return optimal_errors_numerical, std_errors_numerical

    # If mean_test_errors is provided, compute both numerical and theoretical optimal errors
    optimal_errors_numerical = np.zeros((len(KVals), len(P_list)))
    std_errors_numerical = np.zeros((len(KVals), len(P_list)))

    for KInd in range(len(KVals)):
        for PInd in range(len(P_list)):
            mean_error = mean_test_errors[KInd, PInd, :]
            std_error = std_test_errors[KInd, PInd, :]
            min_index = np.argmin(mean_error)
            optimal_errors_numerical[KInd, PInd] = mean_error[min_index]
            std_errors_numerical[KInd, PInd] = std_error[min_index]
            optimal_errors_theory[KInd, PInd] = test_errors_theory[KInd, PInd, min_index]

    return optimal_errors_numerical, std_errors_numerical, optimal_errors_theory

def compute_optimal_errors_bias_variance(mean_test_errors, std_test_errors, test_errors_theory, bias_theory, var_theory, KVals, P_list):
    """
    Compute the optimal test errors for both numerical and theoretical results.
    If mean_test_errors is None, only the optimal theoretical test errors are computed.

    Parameters:
    - mean_test_errors: Mean of numerical test errors across trials (or None if not available).
    - std_test_errors: Standard deviation of numerical test errors across trials (or None if mean_test_errors is None).
    - test_errors_theory: Theoretical test errors.
    - bias_theory
    - var_theory
    - KVals: List of K values used in the experiment.
    - P_list: List of P values used in the experiment.

    Returns:
    If mean_test_errors is provided:
    - optimal_errors_numerical: Optimal numerical test errors.
    - std_errors_numerical: Standard deviation at the optimal point.
    - optimal_errors_theory: Optimal theoretical test errors.
    - optimal_bias_theory
    - optimal_var_theory

    If mean_test_errors is None:
    - optimal_errors_theory: Optimal theoretical test errors.
    """
    
    optimal_errors_theory = np.zeros((len(KVals), len(P_list)))
    optimal_bias_theory = np.zeros((len(KVals), len(P_list)))
    optimal_var_theory = np.zeros((len(KVals), len(P_list)))

    # If mean_test_errors is None, return only theoretical optimal errors
    if mean_test_errors is None:
        for KInd in range(len(KVals)):
            for PInd in range(len(P_list)):
                min_index = np.argmin(test_errors_theory[KInd, PInd, :])
                optimal_errors_theory[KInd, PInd] = test_errors_theory[KInd, PInd, min_index]
        return None, None, optimal_errors_theory

    # If mean_test_errors is provided, compute both numerical and theoretical optimal errors
    optimal_errors_numerical = np.zeros((len(KVals), len(P_list)))
    std_errors_numerical = np.zeros((len(KVals), len(P_list)))

    for KInd in range(len(KVals)):
        for PInd in range(len(P_list)):
            mean_error = mean_test_errors[KInd, PInd, :]
            std_error = std_test_errors[KInd, PInd, :]
            min_index = np.argmin(mean_error)
            optimal_errors_numerical[KInd, PInd] = mean_error[min_index]
            std_errors_numerical[KInd, PInd] = std_error[min_index]
            
            min_index_theory = np.argmin(test_errors_theory[KInd, PInd, :])
            optimal_errors_theory[KInd, PInd] = test_errors_theory[KInd, PInd, min_index_theory]
            optimal_bias_theory[KInd, PInd] = bias_theory[KInd, PInd, min_index_theory]
            optimal_var_theory[KInd, PInd] = var_theory[KInd, PInd, min_index_theory]

    return optimal_errors_numerical, std_errors_numerical, optimal_errors_theory, optimal_bias_theory, optimal_var_theory

def compute_constant_bias_errors(mean_test_errors, std_test_errors, test_errors_theory, bias_theory, var_theory, KVals, P_list):
    """
    Compute the test errors at a ridge value which keeps the bias approximately constant across K values.
    The bias is chosen to be a value within the overlapping bias range across K values.

    Parameters:
    - mean_test_errors: Mean of numerical test errors across trials (or None if not available).
    - std_test_errors: Standard deviation of numerical test errors across trials (or None if mean_test_errors is None).
    - test_errors_theory: Theoretical test errors.
    - bias_theory: Theoretical biases.
    - var_theory: Theoretical variances.
    - KVals: List of K values used in the experiment.
    - P_list: List of P values used in the experiment.

    Returns:
    If mean_test_errors is provided:
    - errors_numerical: Numerical test errors at constant bias.
    - std_errors_numerical: Standard deviation at the selected points.
    - errors_theory: Theoretical test errors at constant bias.
    - bias_theory_selected: Selected theoretical bias (approximately constant across K).
    - var_theory_selected: Theoretical variance at the selected points.

    If mean_test_errors is None:
    - errors_theory: Theoretical test errors at constant bias.
    """
    errors_theory = np.zeros((len(KVals), len(P_list)))
    bias_theory_selected = np.zeros((len(KVals), len(P_list)))
    var_theory_selected = np.zeros((len(KVals), len(P_list)))

    # For each PInd, compute the overlapping bias range across KInd
    for PInd in range(len(P_list)):
        min_biases = []
        max_biases = []
        for KInd in range(len(KVals)):
            biases = bias_theory[KInd, PInd, :]
            min_biases.append(np.min(biases))
            max_biases.append(np.max(biases))
        # Overlapping range is between max(min_biases) and min(max_biases)
        bias_lower = max(min_biases)
        bias_upper = min(max_biases)

        if bias_lower > bias_upper:
            # No overlapping bias range, cannot proceed
            #raise ValueError(f"No overlapping bias range across K values for PInd={PInd}")
            print(f"No overlapping bias range across K values for PInd={PInd}")
        
        # Choose target bias within overlapping range, e.g., midpoint
        target_bias = (bias_lower + bias_upper) / 2.0

        for KInd in range(len(KVals)):
            # Find the index where bias_theory[KInd, PInd, :] is closest to target_bias
            biases = bias_theory[KInd, PInd, :]
            bias_diff = np.abs(biases - target_bias)
            bias_index = np.argmin(bias_diff)

            errors_theory[KInd, PInd] = test_errors_theory[KInd, PInd, bias_index]
            bias_theory_selected[KInd, PInd] = biases[bias_index]
            var_theory_selected[KInd, PInd] = var_theory[KInd, PInd, bias_index]

    if mean_test_errors is None:
        return None, None, errors_theory

    # Now compute the numerical test errors at the same indices
    errors_numerical = np.zeros((len(KVals), len(P_list)))
    std_errors_numerical = np.zeros((len(KVals), len(P_list)))

    for PInd in range(len(P_list)):
        # Use the same target_bias as above
        for KInd in range(len(KVals)):
            biases = bias_theory[KInd, PInd, :]
            bias_diff = np.abs(biases - target_bias)
            bias_index = np.argmin(bias_diff)

            mean_error = mean_test_errors[KInd, PInd, :]
            std_error = std_test_errors[KInd, PInd, :]
            errors_numerical[KInd, PInd] = mean_error[bias_index]
            std_errors_numerical[KInd, PInd] = std_error[bias_index]

    return errors_numerical, std_errors_numerical, errors_theory, bias_theory_selected, var_theory_selected


def compute_optimal_errors_ell(mean_test_errors, std_test_errors, test_errors_theory, M_list, ell_list):
    """
    Compute the optimal test errors for both numerical and theoretical results.
    If mean_test_errors is None, only the optimal theoretical test errors are computed.

    Parameters:
    - mean_test_errors: Mean of numerical test errors across trials (or None if not available).
    - std_test_errors: Standard deviation of numerical test errors across trials (or None if mean_test_errors is None).
    - test_errors_theory: Theoretical test errors.
    - M_list: List of M values used in the experiment.
    - ell_list: List of ell values used in the experiment.

    Returns:
    If mean_test_errors is provided:
    - optimal_errors_numerical: Optimal numerical test errors.
    - std_errors_numerical: Standard deviation at the optimal point.
    - optimal_errors_theory: Optimal theoretical test errors.

    If mean_test_errors is None:
    - optimal_errors_theory: Optimal theoretical test errors.
    """
    
    optimal_errors_theory = np.zeros((len(M_list), len(ell_list)))

    # If mean_test_errors is None, return only theoretical optimal errors
    if mean_test_errors is None:
        for MInd in range(len(M_list)):
            for ellInd in range(len(ell_list)):
                min_index = np.argmin(test_errors_theory[MInd, ellInd, :])
                optimal_errors_theory[MInd, ellInd] = test_errors_theory[MInd, ellInd, min_index]
        return None, None, optimal_errors_theory

    # If mean_test_errors is provided, compute both numerical and theoretical optimal errors
    optimal_errors_numerical = np.zeros((len(M_list), len(ell_list)))
    std_errors_numerical = np.zeros((len(M_list), len(ell_list)))

    for MInd in range(len(M_list)):
        for ellInd in range(len(ell_list)):
            mean_error = mean_test_errors[MInd, ellInd, :]
            std_error = std_test_errors[MInd, ellInd, :]
            min_index = np.argmin(mean_error)
            optimal_errors_numerical[MInd, ellInd] = mean_error[min_index]
            std_errors_numerical[MInd, ellInd] = std_error[min_index]
            optimal_errors_theory[MInd, ellInd] = test_errors_theory[MInd, ellInd, min_index]

    return optimal_errors_numerical, std_errors_numerical, optimal_errors_theory

def fit_power_law_to_errors(M_list, errors, ell_list):
    """
    Fit the optimal errors to a power law E_g ~ M^{-s} for each ell, and extract the scaling exponent s.
    
    Parameters:
    - M_list: List of M values used in the experiment.
    - errors: Optimal errors (shape: len(M_list) x len(ell_list)).
    - ell_list: List of ell values used in the experiment.
    
    Returns:
    - scaling_exponents: Array of scaling exponents s for each ell (shape: len(ell_list)).
    """
    scaling_exponents = np.zeros(len(ell_list))

    # Define the power law function
    def power_law(M, a, s):
        return a * M ** (-s)

    for ellInd, ell in enumerate(ell_list):
        M_vals = M_list
        E_vals = errors[:, ellInd]

        # Exclude zero or negative errors
        valid_indices = E_vals > 0
        M_vals_fit = M_vals[valid_indices]
        E_vals_fit = E_vals[valid_indices]

        # Take logarithms for linear fitting
        log_M = np.log(M_vals_fit)
        log_E = np.log(E_vals_fit)

        # Perform linear regression to find the exponent
        try:
            popt, pcov = curve_fit(lambda M, s, a: s * M + a, log_M, log_E)
            s_fit = -popt[0]
            scaling_exponents[ellInd] = s_fit
        except RuntimeError:
            scaling_exponents[ellInd] = np.nan  # Could not fit

    return scaling_exponents

### Plotting Functions:        

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import os

def plot_test_errors_with_shaded_regions(KVals, P_list, lamVals, mean_test_errors, std_test_errors, test_errors_theory=None, title_prefix="Test Errors", save_path=None):
    """
    Plot numerical test errors with shaded regions representing standard deviation, and theoretical test errors if provided.

    Parameters:
    - KVals: List of K values used in the experiment.
    - P_list: List of P values used in the experiment.
    - lamVals: Array of lambda values.
    - mean_test_errors: Mean of numerical test errors across trials.
    - std_test_errors: Standard deviation of numerical test errors across trials.
    - test_errors_theory: Theoretical test errors (optional).
    - title_prefix: Optional prefix for the plot titles (default is "Test Errors").
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    """
    import matplotlib.pyplot as plt
    import os

    for PInd, P in enumerate(P_list):
        plt.figure(figsize=(2.5, 2))  # Reduced figure size
        plt.rc('font', size=10)     # Set default font size

        legend_handles = []

        for KInd, K in enumerate(KVals):
            color = plt.cm.viridis(KInd / len(KVals))
            # Mean and standard deviation of numerical results
            mean_error = mean_test_errors[KInd, PInd, :]
            std_error = std_test_errors[KInd, PInd, :]
            lower_bound = np.maximum(mean_error - std_error, 1e-10)
            upper_bound = mean_error + std_error

            # Plot shaded region for numerical results
            plt.fill_between(lamVals, lower_bound, upper_bound, color=color, alpha=0.3)

            # Plot theoretical predictions with dashed lines (if provided)
            if test_errors_theory is not None:
                plt.loglog(lamVals, test_errors_theory[KInd, PInd, :], '--', color=color)

            # Create custom legend handles with solid lines
            legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='-', label=f'K={K}'))

        plt.title(f'{title_prefix} for P={P}', fontsize=10, pad=20)
        plt.xlabel(r'$\lambda$', fontsize=10)
        plt.ylabel('Test Error', fontsize=10)

        # Add the legend with custom handles (solid lines)
        plt.legend(handles=legend_handles, fontsize='small', bbox_to_anchor=(1, 1.05), loc='upper left', frameon=False)

        if save_path is not None:
            # Save the figure to the specified path, include P in the filename to distinguish plots
            base, ext = os.path.splitext(save_path)
            save_path_full = f"{base}_P{P}{ext}"
            plt.savefig(save_path_full, format='svg', bbox_inches='tight', transparent=True)
        else:
            plt.show()

            
def plot_test_errors_with_errorbars_vs_P(
    KVals, P_list, P_list_theory, lamVals, mean_test_errors, std_test_errors, test_errors_theory=None,
    title_prefix="Test Errors", nth_lam=1, save_path=None, sharey=False):
    """
    Plot numerical test errors with error bars as a function of P for each K value.
    Each lambda value is represented by a different color, and the color varies continuously with lambda.

    Parameters:
    - KVals: List of K values used in the experiment.
    - P_list: List of P values used in the experiment.
    - P_list_theory: Theoretical P values (for theory curves).
    - lamVals: Array of lambda values.
    - mean_test_errors: Mean of numerical test errors across trials.
    - std_test_errors: Standard deviation of numerical test errors across trials.
    - test_errors_theory: Theoretical test errors.
    - title_prefix: Optional prefix for the plot titles (default is "Test Errors").
    - nth_lam: Plot every nth lambda value to reduce clutter (default is 1, i.e., plot all lambda values).
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    - sharey: Whether to share the y-axis among subplots (default is False).
    """

    # Create a color map for lambda values
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=np.min(lamVals), vmax=np.max(lamVals))  # Logarithmic color scale

    # Create subplots: One for each K value
    fig, axs = plt.subplots(1, len(KVals), figsize=(2.5 * len(KVals), 1.75), sharey=sharey)
    plt.rc('font', size=10)

    # Set a title for the figure above the grid of subplots
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=1.35)

    # Adjust spacing between subplots if sharey is False to prevent label overlap
    if not sharey:
        plt.subplots_adjust(wspace=0.2)  # Increase wspace as needed

    # Handle the case where axs is not a list (i.e., only one subplot)
    if len(KVals) == 1:
        axs = [axs]

    for KInd, K in enumerate(KVals):
        ax = axs[KInd]  # Select the current subplot
        for lamInd in range(0, len(lamVals), nth_lam):  # Plot every nth lambda value
            lam = lamVals[lamInd]
            color = cmap(norm(lam))  # Get color corresponding to lambda
            # Mean and standard deviation of numerical results across P for this lambda value
            mean_error = mean_test_errors[KInd, :, lamInd]
            std_error = std_test_errors[KInd, :, lamInd]

            if test_errors_theory is None:
                ax.errorbar(P_list, mean_error, yerr=std_error, color=color, markersize=.75, elinewidth=0.5, lw=1)

            if test_errors_theory is not None:
                # Plot error bars with dots (no line connecting the dots)
                ax.errorbar(P_list, mean_error, yerr=std_error, fmt='o', color=color, markersize=1.5, elinewidth=0.5)
                # Plot the theoretical curve using ax.loglog
                ax.loglog(P_list_theory, test_errors_theory[KInd, :, lamInd], '-', color=color, lw=0.75)

        # Find and plot the optimal lambda for numerical and theoretical errors
        opt_lam_indices_numerical = np.argmin(mean_test_errors[KInd, :, :], axis=1)  # Optimal lambda for numerical errors
        opt_mean_errors_numerical = mean_test_errors[KInd, np.arange(len(P_list)), opt_lam_indices_numerical]  # Numerical optimal errors
        opt_std_errors_numerical = std_test_errors[KInd, np.arange(len(P_list)), opt_lam_indices_numerical]  # Std dev of numerical optimal errors

        if test_errors_theory is not None:
            # Plot optimal errors for numerics as red markers with error bars
            ax.errorbar(P_list, opt_mean_errors_numerical, yerr=opt_std_errors_numerical, fmt='o', color='red', markersize=1.5, elinewidth=.5, zorder=5, label='$\lambda$ optimal')
            
            opt_lam_indices_theory = np.argmin(test_errors_theory[KInd, :, :], axis=1)  # Optimal lambda for theoretical errors
            opt_test_errors_theory = test_errors_theory[KInd, np.arange(len(P_list_theory)), opt_lam_indices_theory]  # Theoretical optimal errors

            # Plot optimal theoretical errors as red dotted lines
            ax.loglog(P_list_theory, opt_test_errors_theory, 'r-', lw=.75, zorder=6)
        else:
            # Plot optimal errors for numerics
            ax.errorbar(P_list, opt_mean_errors_numerical, yerr=opt_std_errors_numerical, color='red', markersize=1.5, elinewidth=.5, zorder=5, label='$\lambda$ optimal', lw=1)

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Adjust axis limits
        ax.set_xlim([min(P_list) * 0.95, max(P_list) * 1.05])  # X-axis range

        if test_errors_theory is None:
            if sharey:
                y_min = np.min(mean_test_errors) * 0.95
                y_max = np.max(mean_test_errors) * 1.05
            else:
                y_min = np.min(mean_test_errors[KInd, :, :]) * 0.95
                y_max = np.max(mean_test_errors[KInd, :, :]) * 1.05
        else:        
            if sharey:
                y_min = min(np.min(mean_test_errors), np.min(test_errors_theory)) * 0.95
                y_max = max(np.max(mean_test_errors), np.max(test_errors_theory)) * 1.05
            else:
                y_min = min(np.min(mean_test_errors[KInd, :, :]), np.min(test_errors_theory[KInd, :, :])) * 0.95
                y_max = max(np.max(mean_test_errors[KInd, :, :]), np.max(test_errors_theory[KInd, :, :])) * 1.05

        ax.set_ylim([y_min, y_max])

        ax.set_title(f'K = {K}', fontsize=10)
        ax.set_xlabel('P', fontsize=10)
        if KInd == 0:
            ax.set_ylabel('Test Error', fontsize=10)

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        # Format y-axis tick labels to be decimals, not scientific notation
        def log_format(x, pos):
            return '{0:g}'.format(x)
        ax.yaxis.set_major_formatter(FuncFormatter(log_format))
        ax.yaxis.set_minor_formatter(FuncFormatter(log_format))

    # Add a colorbar for lambda values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', label=r'$\lambda$', pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Add the custom legend
    if test_errors_theory is not None:
        handles = [
            plt.errorbar([], [], yerr=0.1, fmt='o', color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),
            plt.Line2D([0], [0], color='black', linestyle='-', label='Theory'),
            plt.Line2D([0], [0], color='red', linestyle='-', label='$\lambda$ optimal')
        ]
    else:
        handles = [
            plt.errorbar([], [], yerr=0.1, color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),
            plt.errorbar([], [], yerr=0.1, color='red', markersize=2.5, elinewidth=0.75, label=r'$\lambda$ Optimal'),
        ]

    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=8, frameon=False)

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.show()

        
def plot_optimal_errors_vs_K_Pcolor(
    KVals, P_list, mean_test_errors, std_test_errors, title="Optimal Test Error vs. K",
    save_path=None, nth_P=1
):
    """
    Plot the optimal test errors versus K for different P values on a single plot.
    The color of the lines depends on P, with a corresponding color bar.

    Parameters:
    - KVals: List or array of K values used in the experiment.
    - P_list: List or array of P values used in the experiment.
    - mean_test_errors: Array of mean numerical test errors, shape (len(KVals), len(P_list), len(lamVals)).
    - std_test_errors: Array of standard deviations of numerical test errors, same shape as mean_test_errors.
    - test_errors_theory: (Optional) Array of theoretical test errors, same shape as mean_test_errors.
    - title: (Optional) Title for the plot (default is "Optimal Test Error vs. K").
    - save_path: (Optional) Path to save the figure in SVG format. If None, the figure is displayed.
    - nth_P: (Optional) Plot every nth P value to reduce clutter (default is 1, i.e., plot all P values).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter

    # Ensure inputs are numpy arrays for easier indexing
    mean_test_errors = np.array(mean_test_errors)
    std_test_errors = np.array(std_test_errors)

    # Determine the number of lambda values
    num_lam = mean_test_errors.shape[2]

    # Initialize arrays to store optimal errors and corresponding std deviations
    optimal_errors_numerical = np.min(mean_test_errors, axis=2)  # Shape: (K, P)
    optimal_lam_indices = np.argmin(mean_test_errors, axis=2)   # Shape: (K, P)
    optimal_std_numerical = std_test_errors[np.arange(len(KVals))[:, None], np.arange(len(P_list)), optimal_lam_indices]  # Shape: (K, P)

    # Create a single plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjusted figure size for better layout
    plt.rc('font', size=10)  # Set default font size

    # Create a color map for P values
    cmap = plt.get_cmap('Accent')
    norm = LogNorm(vmin=np.min(P_list), vmax=np.max(P_list))  # Logarithmic color scale for P

    # Select P indices based on nth_P
    selected_P_indices = range(0, len(P_list), nth_P)
    selected_P_list = [P_list[i] for i in selected_P_indices]

    # Loop over selected P values and plot optimal errors
    for idx, PInd in enumerate(selected_P_indices):
        P = P_list[PInd]
        color = cmap(norm(P))

        # Extract optimal errors and std deviations for this P across all K
        optimal_error = optimal_errors_numerical[:, PInd]
        optimal_std = optimal_std_numerical[:, PInd]

        # Plot numerical optimal errors with error bars
        ax.errorbar(
            KVals, optimal_error, yerr=optimal_std, fmt='o-', color=color,
            markersize=3, elinewidth=0.75, capsize=2, label=f'P={P}'
        )

    # Set logarithmic scales for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis labels and title
    ax.set_xlabel('K', fontsize=10)
    ax.set_ylabel('Optimal Test Error', fontsize=10)
    ax.set_title(title, fontsize=10, pad=15)

    # Format x-axis ticks to display integer K values
    ax.set_xticks(KVals)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.get_xaxis().set_minor_formatter(FuncFormatter(lambda x, pos: ''))

    # Format y-axis tick labels to be in decimal form, not scientific notation
    def log_format(x, pos):
        return '{0:g}'.format(x)
    ax.yaxis.set_major_formatter(FuncFormatter(log_format))
    ax.yaxis.set_minor_formatter(FuncFormatter(log_format))

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib

    # Add colorbar to the right of the plot
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='P', pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Adjust legend to be outside the plot to the right
    # To avoid duplicate labels, only include unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=8, loc='upper left', bbox_to_anchor=(1.4, 1),
        frameon=False
    )

    # Adjust layout to make room for the legend
    #plt.tight_layout()

    # Save or display the plot
    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    #     plt.close(fig)  # Close the figure to free memory
    # else:
    #     plt.show()



def plot_test_errors_with_errorbars_vs_M(MVals, MVals_theory, KVals, lamVals, mean_test_errors, std_test_errors, test_errors_theory=None, title_prefix="Test Errors", nth_lam=1, save_path=None, sharey=False):
    """
    Plot numerical test errors with error bars as a function of M for each K value.
    Each lambda value is represented by a different color, and the color varies continuously with lambda.

    Parameters:
    - MVals: List of M values used in the experiment.
    - MVals_theory: M values used for theoretical curves (can be same as MVals).
    - KVals: List of K values used in the experiment.
    - lamVals: Array of lambda values.
    - mean_test_errors: Mean of numerical test errors across trials.
    - std_test_errors: Standard deviation of numerical test errors across trials.
    - test_errors_theory: Theoretical test errors (optional).
    - title_prefix: Optional prefix for the plot titles (default is "Test Errors").
    - nth_lam: Plot every nth lambda value to reduce clutter (default is 1, i.e., plot all lambda values).
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    - sharey: Whether to share the y-axis among subplots (default is False).
    """
    # Create a color map for lambda values
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=np.min(lamVals), vmax=np.max(lamVals))  # Logarithmic color scale

    # Create subplots: One for each K value
    fig, axs = plt.subplots(1, len(KVals), figsize=(2.5 * len(KVals), 1.75), sharey=sharey)
    plt.rc('font', size=10)
    
    # Set a title for the figure above the grid of subplots
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=1.35)

    # Handle the case where axs is not a list (i.e., only one subplot)
    if len(KVals) == 1:
        axs = [axs]

    # Adjust spacing between subplots if sharey is False to prevent label overlap
    if not sharey:
        plt.subplots_adjust(wspace=0.2)  # Increase wspace as needed

    for KInd, K in enumerate(KVals):
        ax = axs[KInd]  # Select the current subplot
        for lamInd in range(0, len(lamVals), nth_lam):  # Plot every nth lambda value
            lam = lamVals[lamInd]
            color = cmap(norm(lam))  # Get color corresponding to lambda
            # Mean and standard deviation of numerical results across M for this lambda value
            mean_error = mean_test_errors[:, KInd, lamInd]
            std_error = std_errors = std_test_errors[:, KInd, lamInd]

            

            if test_errors_theory is not None:
                # Plot error bars with dots (no line connecting the dots)
                ax.errorbar(MVals, mean_error, yerr=std_error, fmt='o', color=color, markersize=1.5, elinewidth=0.5)
                # Plot the theoretical curve using ax.loglog
                ax.loglog(MVals_theory, test_errors_theory[:, KInd, lamInd], '-', color=color, lw=0.75)
            else:
                # Plot error bars with dots (no line connecting the dots)
                ax.errorbar(MVals, mean_error, yerr=std_error, color=color, markersize=.75, elinewidth=0.5, lw = 1)

        # Find and plot the optimal lambda for numerical and theoretical errors
        opt_lam_indices_numerical = np.argmin(mean_test_errors[:, KInd, :], axis=1)  # Optimal lambda for numerical errors
        opt_mean_errors_numerical = mean_test_errors[np.arange(len(MVals)), KInd, opt_lam_indices_numerical]  # Numerical optimal errors
        opt_std_errors_numerical = std_test_errors[np.arange(len(MVals)), KInd, opt_lam_indices_numerical]  # Std dev of numerical optimal errors

        if test_errors_theory is not None:
            opt_lam_indices_theory = np.argmin(test_errors_theory[:, KInd, :], axis=1)  # Optimal lambda for theoretical errors
            opt_test_errors_theory = test_errors_theory[np.arange(len(MVals_theory)), KInd, opt_lam_indices_theory]  # Theoretical optimal errors
            # Plot optimal theoretical errors as red dotted lines
            ax.loglog(MVals_theory, opt_test_errors_theory, 'r-', zorder=6)
            
            # Plot optimal errors for numerics as red markers with error bars
            ax.errorbar(MVals, opt_mean_errors_numerical, yerr=opt_std_errors_numerical, fmt='o', color='red', markersize=1.5, elinewidth=.5, zorder=5, label='$\lambda$ optimal')
            
        else:
            # Plot optimal errors for numerics as red markers with error bars
            ax.errorbar(MVals, opt_mean_errors_numerical, yerr=opt_std_errors_numerical, color='red', markersize=0.75, elinewidth=.5, lw = 1, zorder=5, label='$\lambda$ optimal')

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Adjust axis limits
        ax.set_xlim([min(MVals) * 0.9, max(MVals) * 1.1])  # X-axis range
        if sharey:
            y_min = np.min(mean_test_errors) * 0.95
            y_max = np.max(mean_test_errors) * 1.05
            if test_errors_theory is not None:
                y_min = min(y_min, np.min(test_errors_theory)) * 0.95
                y_max = max(y_max, np.max(test_errors_theory)) * 1.05
        else:
            y_min = np.min(mean_test_errors[:, KInd, :]) * 0.95
            y_max = np.max(mean_test_errors[:, KInd, :]) * 1.05
            if test_errors_theory is not None:
                y_min = min(y_min, np.min(test_errors_theory[:, KInd, :])) * 0.95
                y_max = max(y_max, np.max(test_errors_theory[:, KInd, :])) * 1.05
        ax.set_ylim([y_min, y_max])

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        # Format y-axis tick labels to be decimals, not scientific notation
        def log_format(x, pos):
            return '{0:g}'.format(x)
        ax.yaxis.set_major_formatter(FuncFormatter(log_format))
        ax.yaxis.set_minor_formatter(FuncFormatter(log_format))

        ax.set_title(f'K = {K}', fontsize=10)
        ax.set_xlabel('M', fontsize=10)
        if KInd == 0:
            ax.set_ylabel('Test Error', fontsize=10)

        ax.tick_params(axis='both', which='major', labelsize=8)

    # Add a colorbar for lambda values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', label=r'$\lambda$', pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Add the custom legend
    if test_errors_theory is not None:
        handles = [
            plt.errorbar([], [], yerr=0.1, fmt='o', color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),  # Black marker for Experiment
            plt.Line2D([0], [0], color='black', linestyle='-', label='Theory'),  # Black solid line for theory
            plt.Line2D([0], [0], color='red', linestyle='-', label='$\lambda$ optimal')  # Red solid line in legend for optimal lambda
        ]
    else:
        handles = [
            plt.errorbar([], [], yerr=0.1, color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),  # Black marker for Experiment
            plt.errorbar([], [], yerr=0.1, color='red', markersize=2.5, elinewidth=0.75, label=r'$\lambda$ Optimal'),  # red marker for optimal lambda
        ]

    # Add the legend to the right of the colorbar
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=8, frameon=False)

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.show()
        
def plot_test_errors_with_errorbars_vs_N(NVals, NVals_theory, KVals, lamVals, mean_test_errors, std_test_errors, test_errors_theory=None, title_prefix="Test Errors", nth_lam=1, save_path=None, sharey=False):
    """
    Plot numerical test errors with error bars as a function of N for each K value.
    Each lambda value is represented by a different color, and the color varies continuously with lambda.

    Parameters:
    - NVals: List of N values used in the experiment.
    - NVals_theory: N values used for theoretical curves (can be same as NVals).
    - KVals: List of K values used in the experiment.
    - lamVals: Array of lambda values.
    - mean_test_errors: Mean of numerical test errors across trials.
    - std_test_errors: Standard deviation of numerical test errors across trials.
    - test_errors_theory: Theoretical test errors (optional).
    - title_prefix: Optional prefix for the plot titles (default is "Test Errors").
    - nth_lam: Plot every nth lambda value to reduce clutter (default is 1, i.e., plot all lambda values).
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    - sharey: Whether to share the y-axis among subplots (default is False).
    """
    # Create a color map for lambda values
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=np.min(lamVals), vmax=np.max(lamVals))  # Logarithmic color scale

    # Create subplots: One for each K value
    fig, axs = plt.subplots(1, len(KVals), figsize=(2.5 * len(KVals), 1.75), sharey=sharey)
    plt.rc('font', size=10)
    
    # Set a title for the figure above the grid of subplots
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=1.35)

    # Handle the case where axs is not a list (i.e., only one subplot)
    if len(KVals) == 1:
        axs = [axs]

    # Adjust spacing between subplots if sharey is False to prevent label overlap
    if not sharey:
        plt.subplots_adjust(wspace=0.2)  # Increase wspace as needed

    for KInd, K in enumerate(KVals):
        ax = axs[KInd]  # Select the current subplot
        for lamInd in range(0, len(lamVals), nth_lam):  # Plot every nth lambda value
            lam = lamVals[lamInd]
            color = cmap(norm(lam))  # Get color corresponding to lambda
            # Mean and standard deviation of numerical results across N for this lambda value
            mean_error = mean_test_errors[:, KInd, lamInd]
            std_error = std_errors = std_test_errors[:, KInd, lamInd]

            

            if test_errors_theory is not None:
                # Plot error bars with dots (no line connecting the dots)
                ax.errorbar(NVals, mean_error, yerr=std_error, fmt='o', color=color, markersize=1.5, elinewidth=0.5)
                # Plot the theoretical curve using ax.loglog
                ax.loglog(NVals_theory, test_errors_theory[:, KInd, lamInd], '-', color=color, lw=0.75)
            else:
                # Plot error bars with dots (no line connecting the dots)
                ax.errorbar(NVals, mean_error, yerr=std_error, color=color, markersize=.75, elinewidth=0.5, lw = 1)

        # Find and plot the optimal lambda for numerical and theoretical errors
        opt_lam_indices_numerical = np.argmin(mean_test_errors[:, KInd, :], axis=1)  # Optimal lambda for numerical errors
        opt_mean_errors_numerical = mean_test_errors[np.arange(len(NVals)), KInd, opt_lam_indices_numerical]  # Numerical optimal errors
        opt_std_errors_numerical = std_test_errors[np.arange(len(NVals)), KInd, opt_lam_indices_numerical]  # Std dev of numerical optimal errors

        if test_errors_theory is not None:
            opt_lam_indices_theory = np.argmin(test_errors_theory[:, KInd, :], axis=1)  # Optimal lambda for theoretical errors
            opt_test_errors_theory = test_errors_theory[np.arange(len(NVals_theory)), KInd, opt_lam_indices_theory]  # Theoretical optimal errors
            # Plot optimal theoretical errors as red dotted lines
            ax.loglog(NVals_theory, opt_test_errors_theory, 'r-', zorder=6)
            
            # Plot optimal errors for numerics as red markers with error bars
            ax.errorbar(NVals, opt_mean_errors_numerical, yerr=opt_std_errors_numerical, fmt='o', color='red', markersize=1.5, elinewidth=.5, zorder=5, label='$\lambda$ optimal')
            
        else:
            # Plot optimal errors for numerics as red markers with error bars
            ax.errorbar(NVals, opt_mean_errors_numerical, yerr=opt_std_errors_numerical, color='red', markersize=0.75, elinewidth=.5, lw = 1, zorder=5, label='$\lambda$ optimal')

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Adjust axis limits
        ax.set_xlim([min(NVals) * 0.9, max(NVals) * 1.1])  # X-axis range
        if sharey:
            y_min = np.min(mean_test_errors) * 0.95
            y_max = np.max(mean_test_errors) * 1.05
            if test_errors_theory is not None:
                y_min = min(y_min, np.min(test_errors_theory)) * 0.95
                y_max = max(y_max, np.max(test_errors_theory)) * 1.05
        else:
            y_min = np.min(mean_test_errors[:, KInd, :]) * 0.95
            y_max = np.max(mean_test_errors[:, KInd, :]) * 1.05
            if test_errors_theory is not None:
                y_min = min(y_min, np.min(test_errors_theory[:, KInd, :])) * 0.95
                y_max = max(y_max, np.max(test_errors_theory[:, KInd, :])) * 1.05
        ax.set_ylim([y_min, y_max])

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        # Format y-axis tick labels to be decimals, not scientific notation
        def log_format(x, pos):
            return '{0:g}'.format(x)
        ax.yaxis.set_major_formatter(FuncFormatter(log_format))
        ax.yaxis.set_minor_formatter(FuncFormatter(log_format))

        ax.set_title(f'K = {K}', fontsize=10)
        ax.set_xlabel('N', fontsize=10)
        if KInd == 0:
            ax.set_ylabel('Test Error', fontsize=10)

        ax.tick_params(axis='both', which='major', labelsize=8)

    # Add a colorbar for lambda values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', label=r'$\lambda$', pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Add the custom legend
    if test_errors_theory is not None:
        handles = [
            plt.errorbar([], [], yerr=0.1, fmt='o', color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),  # Black marker for Experiment
            plt.Line2D([0], [0], color='black', linestyle='-', label='Theory'),  # Black solid line for theory
            plt.Line2D([0], [0], color='red', linestyle='-', label='$\lambda$ optimal')  # Red solid line in legend for optimal lambda
        ]
    else:
        handles = [
            plt.errorbar([], [], yerr=0.1, color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),  # Black marker for Experiment
            plt.errorbar([], [], yerr=0.1, color='red', markersize=2.5, elinewidth=0.75, label=r'$\lambda$ Optimal'),  # red marker for optimal lambda
        ]

    # Add the legend to the right of the colorbar
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=8, frameon=False)

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.show()
        
def plot_optimal_errors_vs_K_Ncolor(
    KVals, NVals, mean_test_errors, std_test_errors, title="Optimal Test Error vs. K",
    save_path=None, nth_N=1
):
    """
    Plot the optimal test errors versus K for different N values on a single plot.
    The color of the lines depends on N, with a corresponding color bar.

    Parameters:
    - KVals: List or array of K values used in the experiment.
    - NVals: List or array of N values used in the experiment.
    - mean_test_errors: Array of mean numerical test errors, shape (len(NVals), len(KVals), len(lamVals)).
    - std_test_errors: Array of standard deviations of numerical test errors, same shape as mean_test_errors.
    - title: (Optional) Title for the plot (default is "Optimal Test Error vs. K").
    - save_path: (Optional) Path to save the figure in SVG format. If None, the figure is displayed.
    - nth_N: (Optional) Plot every nth N value to reduce clutter (default is 1, i.e., plot all N values).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter

    # Ensure inputs are numpy arrays for easier indexing
    mean_test_errors = np.array(mean_test_errors)
    std_test_errors = np.array(std_test_errors)

    # Determine the number of lambda values
    num_lam = mean_test_errors.shape[2]

    # Compute optimal errors over lambda
    optimal_errors_numerical = np.min(mean_test_errors, axis=2)  # Shape: (len(NVals), len(KVals))
    optimal_lam_indices = np.argmin(mean_test_errors, axis=2)   # Shape: (len(NVals), len(KVals))
    # Extract the std deviation corresponding to the optimal lambda
    optimal_std_numerical = std_test_errors[np.arange(len(NVals))[:, None], np.arange(len(KVals)), optimal_lam_indices]  # Shape: (len(NVals), len(KVals))

    # Create a single plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Adjusted figure size for better layout
    plt.rc('font', size=10)  # Set default font size

    # Create a color map for N values
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=np.min(NVals), vmax=np.max(NVals))  # Logarithmic color scale for N

    # Select N indices based on nth_N
    selected_N_indices = range(0, len(NVals), nth_N)
    selected_N_list = [NVals[i] for i in selected_N_indices]

    # Loop over selected N values and plot optimal errors
    for idx, NInd in enumerate(selected_N_indices):
        N = NVals[NInd]
        color = cmap(norm(N))

        # Extract optimal errors and std deviations for this N across all K
        optimal_error = optimal_errors_numerical[NInd, :]
        optimal_std = optimal_std_numerical[NInd, :]

        # Plot numerical optimal errors with error bars
        ax.errorbar(
            KVals, optimal_error, yerr=optimal_std, fmt='o-', color=color,
            markersize=3, elinewidth=0.75, capsize=2, label=f'N={N}'
        )

    # Set logarithmic scales for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis labels and title
    ax.set_xlabel('K', fontsize=10)
    ax.set_ylabel('Optimal Test Error', fontsize=10)
    ax.set_title(title, fontsize=10, pad=15)

    # Format x-axis ticks to display integer K values
    ax.set_xticks(KVals)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.get_xaxis().set_minor_formatter(FuncFormatter(lambda x, pos: ''))

    # Format y-axis tick labels to be in decimal form, not scientific notation
    def log_format(x, pos):
        return '{0:g}'.format(x)
    ax.yaxis.set_major_formatter(FuncFormatter(log_format))
    ax.yaxis.set_minor_formatter(FuncFormatter(log_format))

    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Only needed for older versions of matplotlib

    # Add colorbar to the right of the plot
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label='N', pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Adjust legend to be outside the plot to the right
    # To avoid duplicate labels, only include unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        fontsize=8, loc='upper left', bbox_to_anchor=(1.4, 1),
        frameon=False
    )

    # Adjust layout to make room for the legend
    # plt.tight_layout()

    # Save or display the plot
    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
        # plt.close(fig)  # Close the figure to free memory
    else:
        plt.show()


        
def plot_test_errors_with_errorbars_vs_K(KVals, P_list, P_list_theory, lamVals, mean_test_errors, std_test_errors, test_errors_theory=None, title_prefix="Test Errors", nth_lam=1, save_path=None, sharey=False):
    """
    Plot numerical test errors with error bars as a function of K for each P value.
    Each lambda value is represented by a different color, and the color varies continuously with lambda.

    Parameters:
    - KVals: List of K values used in the experiment (x-axis).
    - P_list: List of P values (used for subplot grid).
    - P_list_theory: Theoretical P values (for theory curves).
    - lamVals: Array of lambda values (controls color).
    - mean_test_errors: Mean of numerical test errors across trials.
    - std_test_errors: Standard deviation of numerical test errors across trials.
    - test_errors_theory: Theoretical test errors (optional).
    - title_prefix: Optional prefix for the plot titles (default is "Test Errors").
    - nth_lam: Plot every nth lambda value to reduce clutter (default is 1, i.e., plot all lambda values).
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    - sharey: Whether to share the y-axis among subplots (default is False).
    """
    # Create a color map for lambda values
    cmap = plt.get_cmap('viridis')
    norm = LogNorm(vmin=np.min(lamVals), vmax=np.max(lamVals))  # Logarithmic color scale

    # Create subplots: One for each P value
    fig, axs = plt.subplots(1, len(P_list), figsize=(2.5 * len(P_list), 1.75), sharey=sharey)
    plt.rc('font', size=10)
    
    # Set a title for the figure above the grid of subplots
    fig.suptitle(title_prefix, fontsize=14, fontweight='bold', y=1.35)

    # Handle the case where axs is not a list (i.e., only one subplot)
    if len(P_list) == 1:
        axs = [axs]

    # Adjust spacing between subplots if sharey is False to prevent label overlap
    if not sharey:
        plt.subplots_adjust(wspace=0.2)  # Increase wspace as needed

    for PInd, P in enumerate(P_list):
        ax = axs[PInd]  # Select the current subplot
        for lamInd in range(0, len(lamVals), nth_lam):  # Plot every nth lambda value
            lam = lamVals[lamInd]
            color = cmap(norm(lam))  # Get color corresponding to lambda
            # Mean and standard deviation of numerical results across K for this lambda value
            mean_error = mean_test_errors[:, PInd, lamInd]
            std_error = std_test_errors[:, PInd, lamInd]

            # Plot error bars with markers, no lines connecting the markers
            ax.errorbar(KVals, mean_error, yerr=std_error, fmt='o', color=color, markersize=1.5, elinewidth=.5)

            if test_errors_theory is not None:
                # Plot error bars with markers, no lines connecting the markers
                ax.errorbar(KVals, mean_error, yerr=std_error, fmt='o', color=color, markersize=1.5, elinewidth=.5)
                # Plot the theoretical curve using ax.loglog
                ax.loglog(KVals, test_errors_theory[:, PInd, lamInd], '--', color=color)
            else:
                # Plot error bars with markers, no lines connecting the markers
                ax.errorbar(KVals, mean_error, yerr=std_error, color=color, markersize=.75, elinewidth=.5, lw = 1)

        # Find and plot the optimal regularization for each K value at this P value
        opt_lam_indices = np.argmin(mean_test_errors[:, PInd, :], axis=1)  # Indices of optimal lambda for each K
        opt_mean_errors = mean_test_errors[np.arange(len(KVals)), PInd, opt_lam_indices]  # Optimal numerical errors
        opt_std_errors = std_test_errors[np.arange(len(KVals)), PInd, opt_lam_indices]  # Std dev of optimal numerical errors

        if test_errors_theory is not None:
            # Plot the optimal errors as red markers (for experimental values)
            ax.errorbar(KVals, opt_mean_errors, yerr=opt_std_errors, fmt='o', color='red', markersize=1.5, elinewidth=.5, zorder=5, label='$\lambda$ optimal')
            opt_test_errors_theory = test_errors_theory[np.arange(len(KVals)), PInd, opt_lam_indices]  # Optimal theoretical errors
            # Plot the optimal theoretical curve as a red dotted line
            ax.loglog(KVals, opt_test_errors_theory, 'r--', zorder=5)
            
        else:
            # Plot the optimal errors as red markers (for experimental values)
            ax.errorbar(KVals, opt_mean_errors, yerr=opt_std_errors, color='red', markersize=.75, elinewidth=.5, zorder=5, label='$\lambda$ optimal', lw = 1)

        # Set log scale for both axes
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Adjust axis limits
        ax.set_xlim([.9 * min(KVals), 1.05 * max(KVals)])  # X-axis range

        if sharey:
            y_min = np.min(mean_test_errors) * 0.95
            y_max = np.max(mean_test_errors) * 1.05
            if test_errors_theory is not None:
                y_min = min(y_min, np.min(test_errors_theory)) * 0.95
                y_max = max(y_max, np.max(test_errors_theory)) * 1.05
        else:
            y_min = np.min(mean_test_errors[:, PInd, :]) * 0.95
            y_max = np.max(mean_test_errors[:, PInd, :]) * 1.05
            if test_errors_theory is not None:
                y_min = min(y_min, np.min(test_errors_theory[:, PInd, :])) * 0.95
                y_max = max(y_max, np.max(test_errors_theory[:, PInd, :])) * 1.05

        ax.set_ylim([y_min, y_max])

        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=6)

        # Format y-axis tick labels to be decimals, not scientific notation
        def log_format(x, pos):
            return '{0:g}'.format(x)
        ax.yaxis.set_major_formatter(FuncFormatter(log_format))
        ax.yaxis.set_minor_formatter(FuncFormatter(log_format))

        ax.set_title(f'P = {P}', fontsize=10)
        ax.set_xlabel('K', fontsize=10)
        if PInd == 0:
            ax.set_ylabel('Test Error', fontsize=10)

        ax.tick_params(axis='both', which='major', labelsize=8)

    # Add the colorbar for lambda values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical', label=r'$\lambda$', pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Add the custom legend
    if test_errors_theory is not None:
        handles = [
            plt.errorbar([], [], yerr=0.1, fmt='o', color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),  # Black circle with error bar for experiments
            plt.Line2D([0], [0], color='black', linestyle='--', label='Theory'),  # Black dotted line for theory
            plt.Line2D([0], [0], color='red', linestyle='-', label='$\lambda$ optimal')  # Red solid line for optimal lambda in the legend
        ]
    else:
        handles = [
            plt.errorbar([], [], yerr=0.1, color='black', markersize=2.5, elinewidth=0.75, label='Experiment'),  # Black with error bar for experiments
            plt.errorbar([], [], yerr=0.1, color='red', markersize=2.5, elinewidth=0.75, label=r'$\lambda$ Optimal'),  # red with error bar for optimal lambda
        ]

    # Add the legend to the right of the colorbar, closer than before
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.1, 0.5), fontsize=8, frameon=False)

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.show()

def plot_optimal_errors_vs_K(KVals, P_list, num_trials, optimal_errors_numerical, std_errors_numerical, optimal_errors_theory=None, title="Optimal Test Error vs. K", save_path=None):
    """
    Plot the optimal test errors for varying K values, for both numerical and theoretical results.

    Parameters:
    - KVals: List of K values used in the experiment.
    - P_list: List of P values used in the experiment.
    - num_trials: Number of trials (used for computing SEM).
    - optimal_errors_numerical: Optimal numerical test errors.
    - std_errors_numerical: Standard deviation at the optimal point.
    - optimal_errors_theory: Theoretical optimal test errors (optional).
    - title: Optional title for the plot (default is "Optimal Test Error vs. K").
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    """
    plt.figure(figsize=(2.5, 2))  # Reduced figure size
    plt.rc('font', size=10)  # Set default font size

    # Prepare colors for different P values
    colors = plt.cm.plasma(np.linspace(0, 1, len(P_list)))

    # Create custom legend handles
    legend_handles = []

    # Loop over P values
    for PInd, P in enumerate(P_list):
        color = colors[PInd]

        # Extract optimal errors and error bars
        optimal_error = optimal_errors_numerical[:, PInd]
        std_error = std_errors_numerical[:, PInd]
        error_bars = std_error

        # Plot numerical optimal errors with error bars
        plt.errorbar(KVals, optimal_error, yerr=error_bars, fmt='o', markersize=2, color=color)

        if optimal_errors_theory is not None:
            # Plot theoretical optimal errors
            plt.plot(KVals, optimal_errors_theory[:, PInd], '--', color=color)

        # Add solid line to legend handles
        legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='-', label=f'P={P}'))

    # Set x-axis ticks to integer K values
    plt.xticks(KVals, fontsize=8)
    plt.yticks(fontsize=8)

    plt.xlabel('K', fontsize=10)
    plt.ylabel('Optimal Test Error', fontsize=10)
    plt.title(title, fontsize=10, pad=20)

    # Add the custom legend with solid lines
    plt.legend(handles=legend_handles, fontsize='small', bbox_to_anchor=(1, 1.05), loc='upper left', frameon=False)

    plt.xscale('log')
    plt.yscale('log')

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.show()   
        
def plot_quantity_vs_K(KVals, P_list, quantity, title="Quantity vs. K", ylabel = 'Quantity', save_path=None):
    """
    Plot the optimal test errors for varying K values, for both numerical and theoretical results.

    Parameters:
    - KVals: List of K values used in the experiment.
    - P_list: List of P values used in the experiment.
    - quantity: thing to be plotted (K by len(P_list)).
    - title: Optional title for the plot (default is "Optimal Test Error vs. K").
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    """
    plt.figure(figsize=(2.5, 2))  # Reduced figure size
    plt.rc('font', size=10)  # Set default font size

    # Prepare colors for different P values
    colors = plt.cm.plasma(np.linspace(0, 1, len(P_list)))

    # Create custom legend handles
    legend_handles = []

    # Loop over P values
    for PInd, P in enumerate(P_list):
        color = colors[PInd]

        # Plot theoretical optimal errors
        plt.plot(KVals, quantity[:, PInd], '--', color=color)

        # Add solid line to legend handles
        legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='-', label=f'P={P}'))

    # Set x-axis ticks to integer K values
    plt.xticks(KVals, fontsize=8)
    plt.yticks(fontsize=8)

    plt.xlabel('K', fontsize=10)
    plt.title(title, fontsize=10, pad=20)

    # Add the custom legend with solid lines
    plt.legend(handles=legend_handles, fontsize='small', bbox_to_anchor=(1, 1.05), loc='upper left', frameon=False)

    plt.ylabel(ylabel)
    
    plt.xscale('log')
    plt.yscale('log')

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.show()   
        
def plot_optimal_errors_vs_M(M_list, ell_list, optimal_errors_numerical, std_errors_numerical, optimal_errors_theory, num_trials, title="Optimal Test Error vs. M", save_path=None):
    """
    Plot the optimal test errors for varying M values, for both numerical and theoretical results.

    Parameters:
    - M_list: List of M values used in the experiment.
    - ell_list: List of ell values used in the experiment.
    - optimal_errors_numerical: Optimal numerical test errors (shape: len(M_list) x len(ell_list), optional).
    - std_errors_numerical: Standard deviation at the optimal point (same shape as above, optional).
    - optimal_errors_theory: Theoretical optimal test errors (same shape as above).
    - num_trials: Number of trials (used for computing SEM).
    - title: Optional title for the plot (default is "Optimal Test Error vs. M").
    - save_path: Optional path to save the figure in SVG format. If None, the figure is displayed instead.
    """
    fig, ax = plt.subplots(figsize=(2.5, 2))  # Reduced figure size
    plt.rc('font', size=10)  # Set default font size

    # Prepare a colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=ell_list[0], vmax=ell_list[-1])

    # Loop over ell values
    for ellInd, ell in enumerate(ell_list):
        color = cmap(norm(ell))
        M_vals = M_list

        # Extract optimal errors and error bars
        if optimal_errors_numerical is not None:
            optimal_error = optimal_errors_numerical[:, ellInd]
            std_error = std_errors_numerical[:, ellInd]
            # Plot numerical optimal errors with error bars
            ax.errorbar(M_vals, optimal_error, yerr=std_error, fmt='o', markersize=2, color=color, label=f'Numerical $\ell$={ell:.2f}')

        if optimal_errors_theory is not None:
            # Plot theoretical optimal errors
            ax.plot(M_vals, optimal_errors_theory[:, ellInd], '--', lw=1, color=color, label=f'Theory $\ell$={ell:.2f}')

    # Set x-axis ticks and labels
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('M', fontsize=10)
    ax.set_ylabel('Optimal Test Error', fontsize=10)
    ax.set_title(title, fontsize=10, pad=20)

    # Add colorbar for ell values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for creating a colorbar
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', label=r'$\ell$', pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Add a custom legend to distinguish between theory and numerical results
    handles = []
    if optimal_errors_numerical is not None:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=5, label='Numerical Results'))  # Numerical (black marker)
    if optimal_errors_theory is not None:
        handles.append(plt.Line2D([0], [0], color='black', linestyle='--', label='Theoretical Results'))  # Theoretical (black dashed line)

    # Add the legend next to the colorbar
    if handles:
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.5, 0.5), fontsize=8, frameon=False)

    # Save or show the plot
    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
    else:
        plt.show()
        
def plot_scaling_exponent_vs_ell_with_theory(ell_list, scaling_exponents, alpha, r, title="Scaling Exponent vs. $\ell$", save_path=None, std_scaling_exponents=None):
    """
    Plot the scaling exponent s as a function of ell.
    
    Parameters:
    - ell_list: List of ell values used in the experiment.
    - scaling_exponents: Array of scaling exponents s for each ell.
    - title: Optional title for the plot.
    - save_path: Optional path to save the figure. If None, the figure is displayed instead.
    - alpha: capacity exponent for theory
    - r: source exponent for theory
    """
    bias_exponent = 2 * alpha * ell_list * min(r, 1)
    var_exponent = 1 - ell_list + 2 * alpha * ell_list * min(r, 0.5)
    
    plt.figure(figsize=(2.5, 2))
    plt.rc('font', size=10)

    if std_scaling_exponents is None:
        plt.plot(ell_list, scaling_exponents, 'o', markersize=3, color = 'k', label = r'Empirical: $E_g$')  
    else:
        plt.errorbar(ell_list,  scaling_exponents, yerr=std_scaling_exponents, marker = 'o', ls = '', markersize=2.5, elinewidth=.5, color = 'k', label = r'Empirical: $E_g$')
    plt.plot(ell_list, bias_exponent, ls = '--', color = '#B8860B', label = 'Theory: Bias')
    plt.plot(ell_list, var_exponent, ls = '--', color = '#FF00FF', label = 'Theory: Var')
    
    plt.xlabel(r'$\ell$', fontsize=10)
    plt.ylabel('$s$', fontsize=10)
    plt.title(title, fontsize=10, pad=20)
    
    plt.legend(fontsize='small', bbox_to_anchor=(1, 1.05), loc='upper left', frameon=False)
    #plt.grid(True)
    #plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format='svg', bbox_inches='tight', transparent=True)
        #plt.close()
    else:
        plt.show()