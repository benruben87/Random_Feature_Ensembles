#Ensembling Library for training ensembles of random feature models.  Trimmed down from the general library for "cerebellar ensembles"
#Modified to use the parameterization from Atanasov scaling and renormalization.
#Ridge scaling is altered in order to fit with the conventions of Simon et. al, where the kernel is normailized by the number of features N in each ensemble member, and ridge is not scaled with any other variables.

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
import auxFuncs

import torch
import copy


import time 
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta

from scipy.stats import gamma
from scipy.stats import ortho_group

from itertools import combinations

tz = pytz.timezone('US/Eastern')
torch.set_default_dtype(torch.float64)

def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


def matrix_sqrt(A):
    if torch.max(torch.abs(A))==0:
        return A
    else:
        L,V = torch.linalg.eigh(A)
        DiagL = torch.diag(L)
    return V @ torch.sqrt(DiagL) @ V.transpose(0,1)

def create_projection_matrices(num_matrices, data_dim, output_dim, variance=1.0):
    """
    Creates a list of F matrices (formerly A matrices) with independently drawn Gaussian entries.
    The variance of the Gaussian distribution can be set by the user.
    
    Parameters:
        num_matrices (int): Number of projection matrices.
        data_dim (int): Dimensionality of the data.
        output_dim (int): Dimensionality of the projection space.
        variance (float): Variance of the Gaussian entries.
        
    Returns:
        List[torch.Tensor]: A list of F matrices.
    """
    return [torch.normal(0, variance**0.5, (output_dim, data_dim)).cuda() for _ in range(num_matrices)]

#given raw ensemble outptus, true labels, and a list of ensembling and error functions 
def calcError_fromRawOutAndLabels_cum(rawOutputs, labels_true, ensErrFuncs_cum):

    test_errors = []
    for ensFunc_cum, errFunc_cum in ensErrFuncs_cum:

        labels_cum = ensFunc_cum(rawOutputs, cum=True)  # assuming ensFunc is predefined and processes rawOutputs
        errors_cum = errFunc_cum(labels_cum, labels_true, cum=True)  # assuming errFunc is predefined and compares labels with labels_true
        test_errors.append(errors_cum)

    #stack errors along 0 dimension corr. to ensErrInd
    return torch.stack(test_errors, dim=0)

#given raw ensemble outptus, true labels, and a list of ensembling and error functions 
def calcError_fromRawOutAndLabels(rawOutputs, labels_true, ensErrFuncs):

    test_errors = []
    for ensFunc_cum, errFunc_cum in ensErrFuncs:

        labels_cum = ensFunc_cum(rawOutputs)  # assuming ensFunc is predefined and processes rawOutputs
        errors_cum = errFunc_cum(labels_cum, labels_true)  # assuming errFunc is predefined and compares labels with labels_true
        test_errors.append(errors_cum)

    #stack errors along 0 dimension corr. to ensErrInd
    return torch.stack(test_errors, dim=0)
    
#Takes in inputs and an ensembling function
#Returns raw outputs (P x K x C), single site predictions (P x K x C), and Ensembled Outputs (P x C)
#If ensFunc = None, then only return raw outputs.
def forward(data, F_list, w_list, ensFunc, singleSiteFunc=None, nonlinearity=None, eta=0, sigma_0_root=None):
    """
    Modified forward function to apply nonlinearity after projection.
    
    Parameters:
        data (torch.Tensor): Input data.
        F_list (list): List of F projection matrices.
        w_list (list): List of weight matrices.
        ensFunc (callable): Ensembling function.
        singleSiteFunc (callable): Single site processing function (optional).
        nonlinearity (callable): Nonlinear function applied after projection (optional).
        eta (float): Readout noise level.
        sigma_0_root (torch.Tensor): Noise covariance matrix for data.
    
    Returns:
        torch.Tensor: Raw outputs, single-site outputs, ensemble output.
    """
    K = len(F_list)
    Nrs = [F_list[k].shape[0] for k in range(K)]

    # Check that F_list and w_list have the same length
    assert len(F_list) == len(w_list), "F_list and w_list must have the same length."

    # Check the dimensions
    C = w_list[0].shape[1]
    for w in w_list:
        assert w.shape[1] == C, "All elements of w_list must have the same number of columns."
    for i in range(len(F_list)):
        assert F_list[i].shape[0] == w_list[i].shape[0], "First dimension of F_list and w_list must agree."

    if len(data.shape) == 1:
        data = data.view(1, -1)
    elif data.shape[1] == 1:
        data = data.permute(1, 0)

    P = data.shape[0]
    D = data.shape[1]

    if sigma_0_root is not None:
        standardNormal = torch.normal(torch.zeros(D, P), torch.ones(D, P), device='cuda')
        popNoise = (sigma_0_root @ standardNormal).transpose(0, 1)
        data += popNoise

    projections = []
    for F in F_list:
        if F.is_sparse:
            mask = F.coalesce().indices()[1]
            projections.append(data[:, mask])
        else:
            projection = data @ F.T
            if nonlinearity is not None:
                projection = nonlinearity(projection)
            projections.append(projection)

    if eta > 0:
        rawOutputs = torch.stack([projections[i] @ w_list[i] + eta * torch.randn(P, C).to('cuda') for i in range(len(projections))])
    else:
        rawOutputs = torch.stack([projections[i] @ w_list[i] for i in range(K)])

    rawOutputs = rawOutputs.permute(1, 0, 2)

    if singleSiteFunc is None:
        singleSiteOutputs = None
    else:
        singleSiteOutputs = singleSiteFunc(rawOutputs)

    if ensFunc is None:
        ensembleOutput = None
    else:
        ensembleOutput = ensFunc(rawOutputs)

    return rawOutputs, singleSiteOutputs, ensembleOutput

# Method to fit only the readout weights using linear regression with a single ridge parameter and coupling parameter beta.
# Assumes training loader and test loader store a single (full) batch
def regress_readout(tr_loader, F_list, eta=0, sigma_0_root=None, lam=0, beta=0, r=None, nonlinearity=None):
    K = len(F_list)
    Nrs = [F_list[k].shape[0] for k in range(K)]

    # Default r is to average.
    if r is None:
        r = 1/(K)*torch.ones(K, device='cuda')

    assert len(r) == K, "r should be a vector of length numReadouts."
    # Check that the sum of torch vector r is 1
    assert torch.abs(torch.sum(r)-1) < 1e-6, "Sum of r should be 1."

    C = tr_loader[1].shape[1]  # Number of classes
    P = tr_loader[0].shape[0]  # Number of samples
    D = tr_loader[0].shape[1]  # Data dimensionality

    data = tr_loader[0]

    # Adds test time feature noise if sigma_0_root is not None
    if sigma_0_root is not None:
        standardNormal = torch.randn(D, P, device='cuda')
        popNoise = (sigma_0_root @ standardNormal).T
        # Add presynaptic noise to data.
        data += popNoise

    # Calculate projections efficiently.
    projections = []
    for F in F_list:
        if F.is_sparse:
            mask = F.coalesce().indices()[1]
            projection = data[:, mask]
            if nonlinearity is not None:
                projection = nonlinearity(projection)  # Apply the nonlinearity here
            projections.append(projection)
        else:
            projection = data @ F.T
            if nonlinearity is not None:
                projection = nonlinearity(projection)  # Apply the nonlinearity here
            projections.append(projection)

    w_list = []

    # If cost function is independent, use the regular linear regression
    if beta == 0:
        for i in range(K):
            psi = projections[i].T  # Nr by P matrix of inputs
            psiT = psi.T
            Y = tr_loader[1]
            if eta > 0:
                Y = Y + eta * torch.randn((P, C), device='cuda')  # Add readout noise during training as well!

            if lam == 0:
                w_list.append(torch.linalg.pinv(psiT) @ Y)
            else:
                P = psi.shape[1]
                Nr = psi.shape[0]
                if P <= Nr:
                    W = psi @ torch.linalg.solve(psiT @ psi + lam*Nr*torch.eye(P, device='cuda'), Y)
                else:
                    W = torch.linalg.solve((psi @ psiT + lam*Nr*torch.eye(Nr, device='cuda')), psi @ Y)
                w_list.append(W)

    # If cost function includes some degree of coupling, use joint training procedure:
    elif 0 < beta <= 1:
        
        raise Exception("Coupled training in ridge regression not yet supported -- need to figure out how to scale ridges.") 
        
#         assert nonlinearity is None
        
#         Fbar = torch.vstack([r[k] * F_list[k] for k in range(K)])
#         psi = torch.sparse.mm(Fbar, (tr_loader[0].transpose(0, 1) + shared_noise))  # includes shared feature noise
#         psiT = psi.transpose(0, 1)

#         Omega = psi @ psiT
#         DiagOmega = torch.zeros(Omega.shape[0], Omega.shape[0], device='cuda')
#         startInd = 0
#         for k in range(K):
#             endInd = startInd + Nrs[k]
#             DiagOmega[startInd:endInd, startInd:endInd] = 1/r[k] * Omega[startInd:endInd, startInd:endInd]
#             startInd = endInd
#         TotOmega = (1 - beta) * DiagOmega + beta * Omega

#         Y = tr_loader[1]
#         b = psi @ Y

#         if eta > 0:
#             # make tensor of readout noise values:
#             Xi = eta * torch.normal(torch.zeros(K, P, C), torch.ones(K, P, C)).to('cuda')
#             Xi_tilde = (1 - beta) * Xi + beta * torch.einsum('i,ijk->jk', r, Xi).unsqueeze(0).expand(K, -1, -1)

#             # Create a tensor from the list of Nrs
#             Nrs_tensor = torch.tensor(Nrs).cumsum(0)
#             Nrs_tensor = torch.cat((torch.tensor([0]), Nrs_tensor))

#             # Use tensor slicing and matrix multiplication
#             templist = [psi[Nrs_tensor[k]:Nrs_tensor[k+1], :] @ Xi_tilde[k, :, :] for k in range(K)]

#             # Make new tensor which is templist stacked along the zero'th dimension.
#             b = b - torch.vstack(templist)

#         if lam == 0:
#             W = torch.linalg.pinv(TotOmega) @ b
#         else:
#             W = torch.linalg.solve(TotOmega + lam*P*torch.eye(TotOmega.shape[0], device='cuda'), b)

#         # Convert W back to list of readout weights
#         startInd = 0
#         for i in range(K):
#             endInd = startInd + Nrs[i]
#             w_list.append(W[startInd:endInd, :])
#             startInd = endInd
    else:
        raise Exception('Invalid value for beta.  Beta must be between 0 and 1.')

    return w_list

#Evaluates the training and test error of the model
#ToDo: Modify to calculate training loss, test losos, and single-site training and test errors.
def eval(tr_loader, test_loader, ensFunc, errFunc, F_list, w_list, eta = 0, sigma_0_root = None, nonlinearity = None):

    #single site func will not effect value returned by this function
    singleSiteFunc = None

    #Calculate Training Error
    x = tr_loader[0]
    y = tr_loader[1]
    rawOut, out_reps, out = forward(x, F_list, w_list, ensFunc, singleSiteFunc, nonlinearity, eta = eta, sigma_0_root = sigma_0_root)
    err_train = errFunc(out, y)

    x = test_loader[0]
    y = test_loader[1]
    rawOut_reps, out_reps, out = forward(x, F_list, w_list, ensFunc, singleSiteFunc, nonlinearity, eta = eta, sigma_0_root = sigma_0_root)
    err_test = errFunc(out,y)    
    return (err_train, err_test)

def eval_test(test_loader, ensFunc, errFunc, F_list, w_list, eta = 0, sigma_0_root = None, nonlinearity = None):

    #single site func will not effect value returned by this function
    singleSiteFunc = None
    x = test_loader[0]
    y = test_loader[1]
    rawOut_reps, out_reps, out = forward(x, F_list, w_list, ensFunc, singleSiteFunc, nonlinearity, eta = eta, sigma_0_root = sigma_0_root)
    err_test = errFunc(out,y)    
    return err_test

#Uses the analytical formula for the generalization error in the case of a linear ensemble of predictors.
#Assumes a single class
def calcEg_linear(F_list, w_list, sigma_s, w_star, sigma_eps=0, r = None):
    
    K = len(F_list)
    assert K==len(w_list)
    for w in w_list:
        assert w.shape[1] == 1
    assert K == len(r)

    D = F_list[0].shape[1] #Ambient dimensionality

    Nrs = [F.shape[0] for F in F_list]
    
    w_star = w_star.reshape(-1,1)

    if r is None:
        r = torch.ones(K, device = 'cuda')/K
    
    #Calculate the effective weights
    what = torch.zeros_like(w_star)
    for F, w, r_cur, nr in zip(F_list, w_list, r, Nrs):
        what += r_cur*torch.mm(F.transpose(0,1), w)
        
    what = what.reshape(-1)
    sigma_s = sigma_s.reshape(-1)
    w_star = w_star.reshape(-1)
        
    Eg = torch.sum((what - w_star)**2*sigma_s) + sigma_eps**2

    return Eg


#TODO: fix this function for proper parameterization of the ridge under coupling.
# def error_sweep_lam_beta(tr_loader, test_loader, F_list, lams, betas, ensErrFuncs, eta = 0, sigma_0_root = None, r = None, nonlinearity = None):

#     K = len(F_list)
#     assert K == len(r), "r must be a vector of length numReadouts."
#     D = F_list[0].shape[1] #data dimensionality
#     P = tr_loader[0].shape[0] #Number of samples
#     C = tr_loader[1].shape[1] #Number of classes

#     Nrs = [F_list[k].shape[0] for k in range(K)]
#     # Create a tensor from the list of Nrs
#     Nrs_tensor = torch.tensor(Nrs).cumsum(0)
#     Nrs_tensor = torch.cat((torch.tensor([0]), Nrs_tensor))

#     if r is None:
#         r = 1/(K)*torch.ones(K, device = 'cuda')
#     assert len(r) == K, "r should be a vector of length numReadouts."
#     #check that sum of torch vector r is 1
#     assert torch.abs(torch.sum(r)-1)<1e-6, "Sum of r should be 1."

#     if sigma_0_root == None:
#         shared_noise = 0
#     else:
#         shared_noise = sigma_0_root@torch.randn(D, P).to('cuda')      
    
#     #Calculate covariance matrices:
#     Fbar = torch.vstack([r[k]*F_list[k] for k in range(K)])
#     psi = torch.sparse.mm(Fbar, (tr_loader[0].transpose(0,1)+shared_noise)) #includes shared feature noise
#     psiT = psi.transpose(0,1)
    
#     Omega = psi@psiT
#     DiagOmega = torch.zeros(Omega.shape[0], Omega.shape[0], device = 'cuda')
#     for k in range(K):
#         DiagOmega[Nrs_tensor[k]:Nrs_tensor[k+1], Nrs_tensor[k]:Nrs_tensor[k+1]] = 1/r[k]*Omega[Nrs_tensor[k]:Nrs_tensor[k+1], Nrs_tensor[k]:Nrs_tensor[k+1]]

#     Y = tr_loader[1]
#     b_base = psi@Y

#     Xi = eta*torch.randn(K, P, C).to('cuda')
#     Xi_tilde_beta0 = Xi
#     Xi_tilde_beta1 = torch.einsum('i,ijk->jk', r, Xi).unsqueeze(0).expand(K, -1, -1)

#     if eta>0:

#         # Use tensor slicing and matrix multiplication
#         templist_beta0 = [psi[Nrs_tensor[k]:Nrs_tensor[k+1], :] @ Xi_tilde_beta0[k, :, :] for k in range(K)]
#         templist_beta1 = [psi[Nrs_tensor[k]:Nrs_tensor[k+1], :] @ Xi_tilde_beta1[k, :, :] for k in range(K)]

#         b_modifier_beta0 = torch.vstack(templist_beta0)
#         b_modifier_beta1 = torch.vstack(templist_beta1)


#     #Calculate the errors for each value of lam and beta and each ensembling/error function.
#     tr_errors = np.empty((len(lams), len(betas), len(ensErrFuncs)))
#     test_errors = np.empty((len(lams), len(betas), len(ensErrFuncs)))

#     for betaInd, beta in enumerate(betas):

#         # #if K=1, then beta is irrelevant.
#         # if K==1 and betaInd>0: 
#         #     tr_errors[:, betaInd, :] = tr_errors[:, 0, :]
#         #     test_errors[:, betaInd, :] = test_errors[:, 0, :]
#         #     continue

#         TotOmega = (1-beta)*DiagOmega + beta*Omega
#         b = b_base
#         if eta>0:
#             b = b - (1-beta)*b_modifier_beta0 - beta*b_modifier_beta1

#         for lamInd, lam in enumerate(lams):
#             if lam==0:
#                 W = torch.linalg.pinv(TotOmega)@b
#             else:
#                 W = torch.linalg.solve(TotOmega + lam*P*torch.eye(TotOmega.shape[0], device = 'cuda'), b)

#             #convert W back to list of readout weights
#             startInd = 0
#             for i in range(K):
#                 endInd = startInd + Nrs[i]
#                 w_list[i] = W[startInd:endInd, :]

#                 startInd = endInd

#             #Calculate the errors for each ensembling/error function.
#             for ensErrInd, ensErrFunc in enumerate(ensErrFuncs):
#                 ensFunc, errFunc = ensErrFunc
#                 curErrors = eval(tr_loader, test_loader, ensFunc, errFunc, F_list, w_list, eta = eta, sigma_0_root = sigma_0_root)
#                 tr_errors[lamInd, betaInd, ensErrInd] = curErrors[0].item()
#                 test_errors[lamInd, betaInd, ensErrInd] = curErrors[1].item()

#     return tr_errors, test_errors

def error_sweep_lam(tr_loader, test_loader, F_list, lam_list, ensErrFuncs, r=None, nonlinearity=None, calc_train_err=False, useLinEg=False, sigma_s = None, w_star = None, sigma_eps=None, cum = False):
    
    start_time = time.time()

    # First check if we are to calculate error using a test loader:
    if useLinEg == False:
        assert test_loader is not None, "test_loader cannot be None if not using linear error formula."

    K = len(F_list)
    Nrs = [F_list[k].shape[0] for k in range(K)]
    Nrs_tensor = torch.tensor(Nrs).cumsum(0)
    Nrs_tensor = torch.cat((torch.tensor([0]), Nrs_tensor))

    if r is None:
        r = 1/(K)*torch.ones(K, device='cuda')
    assert len(r) == K, "r should be a vector of length numReadouts."
    assert torch.abs(torch.sum(r)-1) < 1e-6, "Sum of r should be 1."

    P = tr_loader[0].shape[0]
    M = tr_loader[0].shape[1] # Number of features

    tr_features = tr_loader[0]
    tr_projections = []
    for F in F_list:
        if F.is_sparse:
            mask = F.coalesce().indices()[1]
            projection = tr_features[:, mask]
            if nonlinearity is not None:
                projection = nonlinearity(projection)  # Apply the nonlinearity
            tr_projections.append(projection)
        else:
            projection = tr_features @ F.T
            if nonlinearity is not None:
                projection = nonlinearity(projection)  # Apply nonlinearity
            tr_projections.append(projection)

    psi_list = [tr_projections[k].T for k in range(K)]

    # Construct list of covariance matrices.
    Omega_list = []
    for psi, nr in zip(psi_list, Nrs):
        if P > nr:
            Omega_list.append(psi @ psi.T)
        else:
            Omega_list.append(psi.T @ psi)
        
    #Calculate projections of test samples if necessary
    if useLinEg == False:
        test_features = test_loader[0]
        test_projections = []
        for F in F_list:
            if F.is_sparse:
                mask = F.coalesce().indices()[1]
                projection = test_features[:, mask]
                if nonlinearity is not None:
                    projection = nonlinearity(projection)  # Apply the nonlinearity
                test_projections.append(projection)
            else:
                projection = test_features @ F.T
                if nonlinearity is not None:
                    projection = nonlinearity(projection)  # Apply nonlinearity
                test_projections.append(projection)

        psi_list_test_err = [test_projections[k].T for k in range(K)]

    Y = tr_loader[1]

    tr_errors = torch.empty((len(lam_list), len(ensErrFuncs), K)) if calc_train_err else None
    if useLinEg:
        test_errors = torch.empty(len(lam_list))
    elif cum:
        test_errors = torch.empty((len(lam_list), len(ensErrFuncs), K))
    else:
        test_errors = torch.empty((len(lam_list), len(ensErrFuncs)))

    eigenvalues_list = []
    eigenvectors_list = []
    for k in range(K):
        eigenvalues, eigenvectors = torch.linalg.eigh(Omega_list[k])
        eigenvalues_list.append(eigenvalues)
        eigenvectors_list.append(eigenvectors)

    for lamInd, lam in enumerate(lam_list):
        regularized_inverse_matrix_list = []  # Regularized inverse
        for k in range(K):
            eigenvalues = eigenvalues_list[k]
            eigenvectors = eigenvectors_list[k]
            if lam == 0:
                inverted_eigenvalues = torch.where(eigenvalues > 1e-10, 1.0 / eigenvalues, torch.zeros_like(eigenvalues))
                scaled_eigenvectors = eigenvectors * inverted_eigenvalues.unsqueeze(0)
                pseudoinverse_matrix = scaled_eigenvectors @ eigenvectors.T
            else:
                adjusted_eigenvalues = eigenvalues + Nrs[k]*lam #Scale ridge with N
                inverted_eigenvalues = 1.0 / adjusted_eigenvalues
                scaled_eigenvectors = eigenvectors * inverted_eigenvalues.unsqueeze(0)
                pseudoinverse_matrix = scaled_eigenvectors @ eigenvectors.T
            regularized_inverse_matrix_list.append(pseudoinverse_matrix)

        w_list = []  # List of readout weights for each ensemble member
        for k in range(K):
            if P > Nrs[k]:
                W = regularized_inverse_matrix_list[k] @ (psi_list[k] @ Y)
            else:
                W = psi_list[k] @ regularized_inverse_matrix_list[k] @ Y

            w_list.append(W)

        
        if useLinEg:
            test_errors[lamInd] = calcEg_linear(F_list, w_list, sigma_s, w_star, sigma_eps=sigma_eps, r = r)
        else:
            # Calculate raw outputs
            rawOut_test = torch.stack([psi_list_test_err[i].T @ w_list[i] for i in range(K)])
            rawOut_test = rawOut_test.permute(1, 0, 2)

            true_labels = test_loader[1]

            # Now calculate test errors
            for ensErrInd, ensErrFunc in enumerate(ensErrFuncs):
                if cum:
                    test_errors[lamInd, ensErrInd, :] = calcError_fromRawOutAndLabels_cum(rawOut_test, true_labels, [ensErrFunc])
                else:
                    test_errors[lamInd, ensErrInd] = calcError_fromRawOutAndLabels(rawOut_test, true_labels, [ensErrFunc])

    total_time = time.time() - start_time

    return test_errors