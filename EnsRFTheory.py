#File with functions for calculating theory curves for random features regression from the eigenvalues of the deterministic kernel and task structure.
#Using parameterization where kernel is normalized by the number of features.  Equivalent to using unnormalized kernel and scaling ridge with number of features N.  Note that the effective ridge scaling therefore depends on K when M = KN is held fixed and K varies.

import numpy as np
import scipy as sp

def calc_Df1(Sigma, kappa):
    return np.sum(Sigma/(Sigma + kappa))
    
def calc_Df2(Sigma, kappa):
    return np.sum(Sigma**2/(Sigma + kappa)**2)
    
def calc_tf(Sigma, wbar, kappa):
    return np.sum(wbar**2*Sigma/(Sigma + kappa))
        
def calc_tfprime(Sigma, wbar, kappa):
    return -1*np.sum(wbar**2*Sigma/(Sigma + kappa)**2)

#Calculate kappa for kernel regression with deterministic kernel eigenvalues Sigma
#kappa = lam/(1-Df1/P)
def calc_kappa_krr(Sigma, wbar, lam, P):
    
    tol = 1e-16

    D = Sigma.shape[0]
    
    # if lam==0 and P>D:
    #     return 0

    kappa_max = lam+1/P*np.sum(Sigma) + 10*tol
    kappa_min = lam
    
    #First find kappa such that Df1 is about P
    if P<D:
        kappa_0 = sp.optimize.bisect(lambda kappa_0: P - calc_Df1(Sigma, kappa_0), 0, kappa_max, xtol=tol, maxiter=1000)
    
        #First make sure that Df1 is not greater than P, which would be a major party foul.
        while(calc_Df1(Sigma, kappa_0)>P):
            kappa_0 = kappa_0+10*tol
            
        kappa_min = max(kappa_0, lam)
        
    
    #At zero ridge kappa_min is kappa
    if lam==0:
        return kappa_min

    # print(kappa_min, kappa_max)
    # print(kappa_min - lam/(1-D/P*df1_multiscale(Dks, etaks, kappa_min)), kappa_max - lam/(1-D/P*df1_multiscale(Dks, etaks, kappa_max)))
    
    kappa = sp.optimize.bisect(lambda kappa: kappa - lam/(1-1/P*calc_Df1(Sigma, kappa)), kappa_min, kappa_max, xtol=tol, maxiter=1000)
    return kappa


def calc_kappas_rf_krr(Sigma, wbar, lam, P, N):

    tol = 1e-16

    D = Sigma.shape[0]
    
    ### Switch to not scaling the ridge with any variables.  In alex's stuff ridge scales with P.
    lam = lam/P
     
    #First determine kappa for zero ridge.  This will serve as a lower bound for the bisection method if lam is greater than zero
    kappa_2_min = lam
    kappa_2_max = lam + (P+N)/(P*N)*np.sum(Sigma)
    
    B = min(N, P) #Bottleneck
    
    if B<D:
        kappa_2_0_min= 0
        kappa_2_0_max = np.sum(Sigma)/B
        kappa_2_0 = sp.optimize.bisect(lambda x: calc_Df1(Sigma, x) - B, kappa_2_0_min, kappa_2_0_max, xtol=tol, maxiter=1000)
        
        #Make sure that Df1<B
        while(calc_Df1(Sigma, kappa_2_0)>B):
            kappa_2_0 = kappa_2_0+tol
    
        kappa_2_min = max(kappa_2_0, lam)
    
    
    #Handle the ridgeless cases
    if lam==0:
        kappa_2 = kappa_2_min
        if B>D:
            kappa_1 = 0
            kappa_2 = 0
        elif N<D and P>=D:
            kappa_1 = 0
        elif N>=D and P<D:
            kappa_1 = kappa_2
        else:
            kappa_1 = kappa_2*(1-1/N*calc_Df1(Sigma, kappa_2))
            
        return kappa_1, kappa_2

    #if lam is not zero, we solve the equation kappa*(1-1/P*Df1)*(1-1/N*Df1) = lam
    kappa_2_min = max(kappa_2_min, lam)
    
    #There are rare cases where the minimum value of kappa_2 found by solving Df1(Sigma, kappa_2_0) == B is slightly too high.  To alleviate this, we lower kappa_2_min until we are in the clear
    while kappa_2_min*(P-calc_Df1(Sigma, kappa_2_min))*(N-calc_Df1(Sigma, kappa_2_min)) > lam*P*N:
        kappa_2_min = kappa_2_min-tol
        
    if calc_Df1(Sigma, kappa_2_0)>B:
        raise Exception('The tolerance has jumped us back :(')
        
    #Similarly, we will raise the maximum value to ensure that it is in the clear
    while kappa_2_max*(P-calc_Df1(Sigma, kappa_2_max))*(N-calc_Df1(Sigma, kappa_2_max)) < lam*P*N:
        kappa_2_ax = kappa_2_max+tol
    
    # #Diagnostic: Print bounds for min and max
    # print('kappa_min gives: ' + str(kappa_2_min*(P-calc_Df1(Sigma, kappa_2_min))*(N-calc_Df1(Sigma, kappa_2_min)) - lam*P*N))
    # print('Df1 kappa_min: ' + str(calc_Df1(Sigma, kappa_2_min)))
    # print('lam: ' + str(lam) + ' kappa_min: ' + str(kappa_2_min))
    # print('N, P, D: ' + str(N) + ', ' + str(P) + ', ' + str(D))
    # print('kappa_max gives: ' + str(kappa_2_max*(P-calc_Df1(Sigma, kappa_2_max))*(N-calc_Df1(Sigma, kappa_2_max)) - lam*P*N))
    
    #Note: Bisection method throws error automatically if it does not converge in maxiter iterations!
    kappa_2 = sp.optimize.bisect(lambda kappa_2: kappa_2*(P-calc_Df1(Sigma, kappa_2))*(N-calc_Df1(Sigma, kappa_2)) - lam*P*N, kappa_2_min, kappa_2_max, xtol=tol, maxiter=1000)
    
    kappa_1 = lam/(1-1/P*calc_Df1(Sigma, kappa_2))
    
    return kappa_1, kappa_2


## calculate generalization error.
def Eg_KInf(Sigma, wbar, lam, P, N, sigma_eps = 0, verbose = False):
    
    assert lam>0, "ridge=0 is not supported"
    
    D = Sigma.shape[0]
    kappa_1, kappa_2 = calc_kappas_rf_krr(Sigma, wbar, lam, P, N)
    Df2 = calc_Df2(Sigma, kappa_2)
    gamma_2 = 1/P*Df2
    tfprime = calc_tfprime(Sigma, wbar, kappa_2)
    Eg = -kappa_2**2/(1-gamma_2)*tfprime + sigma_eps**2/(1-gamma_2)

    #print all variables if verbose:
    if verbose:
        print("kappa_2 = ", kappa_2)
        print("Df2 = ", Df2)
        print("gamma_2 = ", gamma_2)
        print("tfprime = ", tfprime)
    return Eg

def Eg_KInf_fromKappa2(Sigma, wbar, kappa_2, P, N, sigma_eps):
    D = Sigma.shape[0]
    Df2 = calc_Df2(Sigma, kappa_2)
    gamma_2 = 1/P*Df2
    tfprime = calc_tfprime(Sigma, wbar, kappa_2)
    Eg = -kappa_2**2/(1-gamma_2)*tfprime + sigma_eps**2/(1-gamma_2)
    return Eg

def Eg_K1(Sigma, wbar, lam, P, N, sigma_eps = 0, verbose = False):
    
    assert lam>0, "ridge=0 is not supported"
    
    D = Sigma.shape[0]
    kappa_1, kappa_2 = calc_kappas_rf_krr(Sigma, wbar, lam, P, N)
    Df1 = calc_Df1(Sigma, kappa_2)
    Df2 = calc_Df2(Sigma, kappa_2)

    tf = calc_tf(Sigma, wbar, kappa_2)
    tfprime = calc_tfprime(Sigma, wbar, kappa_2)
    
    
    # if lam==0 and N>P:
    #     signal = -kappa2**2*tfprime/(1-D/P*df2) + kappa2*tf*(1-N/D)/(1-P/D)*(P/N)/(1-P/N)
    #     noise = sigma_eps**2*(D/P*df2/(1-D/P*df2) + (1-N/D)/(1-P/D)*(P/N)/(1-P/N))
    #     Eg = signal + noise
    #     if verbose:
    #         print('special case: lam=0, N>=P')
    #         print("kappa2 = ", kappa2)
    #         print("df1 = ", df1)
    #         print("df2 = ", df2)
    #         print("tf = ", tf)
    #         print("tfprime = ", tfprime)
    #     return Eg

    dlogdlog = (N-Df1)/(N-Df2)

    gamma_1 = 1/P*(Df1-dlogdlog*(Df1-Df2))
    Eg = -kappa_2**2*tfprime/(1-gamma_1)*dlogdlog + kappa_2*tf/(1-gamma_1)*(1-dlogdlog) + sigma_eps**2/(1-gamma_1)
    #Print all variables:
    if verbose:
        print("kappa2 = ", kappa_2)
        print("kappa1 = ", kappa_1)
        print("Df1 = ", Df1)
        print("Df2 = ", Df2)
        print("tf = ", tf)
        print("tfprime = ", tfprime)
        print("dlogdlog = ", dlogdlog)
        print("gamma_1 = ", gamma_1)
    return Eg

def Eg_K1_fromKappa2(Sigma, wbar, kappa_2, P, N, sigma_eps = 0):
    
    D = Sigma.shape[0]

    Df1 = calc_Df1(Sigma, kappa_2)
    Df2 = calc_Df2(Sigma, kappa_2)

    tf = calc_tf(Sigma, wbar, kappa_2)
    tfprime = calc_tfprime(Sigma, wbar, kappa_2)

    dlogdlog = (N-Df1)/(N-Df2)

    gamma_1 = 1/P*(Df1-dlogdlog*(Df1-Df2))
    Eg = -kappa_2**2*tfprime/(1-gamma_1)*dlogdlog + kappa_2*tf/(1-gamma_1)*(1-dlogdlog) + sigma_eps**2/(1-gamma_1)

    return Eg


def Eg_K(Sigma, wbar, lam, P, N, K, sigma_eps=0, verbose=False, returnBiasVariance = False):
    """
    Compute the generalization error as a function of K for a random feature model.

    Parameters:
    - Sigma: Array of eigenvalues of the kernel
    - wbar: Array of wbar coefficients
    - lam: Regularization parameter (ridge)
    - P: Number of training samples
    - N: Number of features per ensemble member
    - K: Either a single positive number or a list/array of positive numbers representing the number of ensemble members
    - sigma_eps: Noise variance (default 0)
    - verbose: If True, print debug information (default False)
    - returnBiasVaraince: boolean to return bias and variance as well as error.

    Returns:
    - Generalization error as a function of K
    """
    
    # Compute the bias term (K -> infinity)
    Bias = Eg_KInf(Sigma, wbar, lam, P, N, sigma_eps, verbose)
    
    # Compute the variance term (K = 1) minus the bias to get the variance component
    Variance = Eg_K1(Sigma, wbar, lam, P, N, sigma_eps, verbose) - Bias
    
    # Check if K is a single positive number (can be float or int)
    if np.isscalar(K) and K > 0:
        if returnBiasVariance:
            return Bias + 1/K * Variance, Bias, Variance
        return Bias + 1/K * Variance
    
    # Check if K is a list or array of positive numbers
    elif isinstance(K, (list, np.ndarray)) and all(k > 0 for k in K):
        if returnBiasVariance:
            return [Bias + 1/k * Variance for k in K], Bias, Variance
        return [Bias + 1/k * Variance for k in K]
    
    # Throw an exception if K is neither a positive number nor a valid list/array of positive numbers
    else:
        raise ValueError("K must be a positive number or a list/array of positive numbers.")
