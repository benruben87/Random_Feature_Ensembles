import torch
import pytz
from datetime import datetime
from dateutil.relativedelta import relativedelta
tz = pytz.timezone('America/New_York')
import numpy as np
from scipy.stats import gamma

#Linear prediction single site and ensembling functions
def identity(x):
    return x

def mean(x, cum = False):

    if cum:
        return mean_cum(x)

    return torch.mean(x, dim=1)

# Calculates running average over dimension dim.  default dim=1
def mean_cum(tensor, dim=1):
    # Compute cumulative sum along the specified dimension
    cumsum = torch.cumsum(tensor, dim=dim)
    
    # Create a tensor of index counts
    count = torch.arange(1, tensor.size(dim) + 1, device=tensor.device, dtype=tensor.dtype)
    shape = [1] * tensor.dim()
    shape[dim] = -1
    count = count.view(shape)
    
    # Compute the cumulative mean
    cummean = cumsum / count

    return cummean


#score average ensembling functions
def scoreAverage(x, cum = False):
    if cum:
        return scoreAverage_cum(x)
    
    return torch.sign(torch.mean(x, dim=1))

def scoreAverage_cum(x):
    return torch.sign(torch.cumsum(x, dim=1))

#Binary classification single site and ensembling functions
def sign(x):
    return torch.sign(x)

def majorityVote(predictions, cum = False):

    if cum:
        return majorityVote_cum(predictions)

    return torch.sign(torch.sum(torch.sign(predictions), dim=1))

def majorityVote_cum(predictions):
    return torch.sign(torch.cumsum(torch.sign(predictions), dim=1))

def median(x, cum=False, dim=1):
    if cum:
        return median_cum(x, dim=dim)
    return torch.median(x, dim=dim).values


#One-Hot Classification single site and ensembling functions.
#Single site function for one-hot classification
def oneHotArgMax(predictions):
    """
    Given a tensor `predictions` of shape [P, K, C],  where P is a number of samples, K is an ensemble size, and C is the number of classes, return a new tensor of shape [P, K, C] where each row [p, k, :] contains a one-hot encoding of the class with the maximum predicted score for each ensemble member for sample p.

    Parameters:
    - predictions (torch.Tensor): A tensor of shape [P, K, C] representing the predicted class scores 
                                  for K ensemble members and C classes.

    Returns:
    - torch.Tensor: A tensor of shape [P, K, C] consisting of one-hot encodings for the maximum 
                    classes along the third dimension.
    """
        
    # Find the indices of the maximum values along each row (axis 1)
    _, max_indices = torch.max(predictions, dim=2)
    
    # Create one-hot encodings using the max indices
    one_hot_maxes = F.one_hot(max_indices, num_classes=predictions.size(2)).to(predictions.device)
    
    return one_hot_maxes

#Ensembling function for one-hot classification
def majorityVoteOneHot(predictions):
    """
    Given a tensor `predictions` of shape [P, K, C], where P is the number of data points, 
    K is the number of ensemble members, and C is the number of classes, return a new 
    tensor of shape [P, C] where each row contains a one-hot encoding of the class 
    receiving the majority vote from the ensemble members for each data point.

    Parameters:
    - predictions (torch.Tensor): A tensor of shape [P, K, C] representing the predicted 
                                  class scores for P data points from K ensemble members 
                                  across C classes.

    Returns:
    - torch.Tensor: A tensor of shape [P, C] consisting of one-hot encodings for the 
                    majority-voted class for each data point.
    """
    
    # Calculate the argmax along the K dimension for each data point, resulting in a tensor of shape [P, K]
    _, max_indices = torch.max(predictions, dim=2)
    
    # Count the occurrences of each class index for each data point
    counts = torch.zeros(predictions.shape[0], predictions.shape[2], device=predictions.device)
    for i in range(predictions.shape[1]):
        counts += F.one_hot(max_indices[:, i], num_classes=predictions.size(2))
    
    # Find the indices of the maximum counts for each data point
    _, majority_indices = torch.max(counts, dim=1)
    
    # Create one-hot encodings using the majority indices
    one_hot_majority = F.one_hot(majority_indices, num_classes=predictions.size(2))
    
    return one_hot_majority


#Error function for one-hot classification
def MultiClassErrorRate(out, y):
    #out: [P, C] one-hot label predictions
    #y: [P, C] one-hot ground truth labels
    return torch.mean(torch.sum(torch.abs(out-y), dim = 1).to(float))/2

##### Angular Variable functions:
#Single site function for angles.
def normalizeAngle(predictions):
    return F.normalize(predictions, p=2, dim=2)

#Ensembling Function for angular variables represented with sin, cos
#Analog of a majority vote for angular variables, where angles are normalized, averaged, then normalized.
def circularMean(predictions):
    vecs_norm = F.normalize(predictions, p=2, dim=2)
    vecs_mean = torch.mean(vecs_norm, dim = 1)
    #Return normalized angle predictions
    return F.normalize(vecs_mean, p=2, dim=1)

#Analog of score average for angular variables, where angles are normalized, averaged, then normalized.
def circularScoreAve(predictions):
    vecs_mean = torch.mean(predictions, dim = 1)
    #Return normalized angle predictions
    return F.normalize(vecs_mean, p=2, dim=1)
#       """
#     Given a tensor `predictions` of shape [P, K, 2],  where P is a number of samples, K is an ensemble size, and the last dimension has length 2 for Sin and Cos, return a new tensor of shape [P, K, 2] where each row [p, k, :] contains a Sin, Cos pair that has been corrected to the closest normalized value.

#     Parameters:
#     - predictions (torch.Tensor): A tensor of shape [P, K, 2] representing the predicted angular scores 
#                                   for K ensemble members.

#     Returns:
#     - torch.Tensor: A tensor of shape [P, 2] consisting of noramlized angular encodings for the maximum 
#                     classes along the third dimension.
#     """



#Error functions for angles.  Error is bounded above by pi

#Auxilary function that converts from sin cos representation of predicted and true angles to angle differences from 0 to pi
def sinCosToDiff(out, y):
    angles_out = torch.atan2(out[:, 0], out[:, 1])
    angles_y = torch.atan2(y[:, 0], y[:, 1])

    # Compute angular differences
    angular_diffs = torch.abs(angles_out - angles_y)

    # Handle wrap-around for angles > pi
    angular_diffs = torch.where(angular_diffs > torch.pi, 2*torch.pi - angular_diffs, angular_diffs)
    return angular_diffs

#out: [P,2] angular encoding predictions as sin, cos
#y: [P, 2] anglular encoding labels as sin, cos   
def meanAngularDiff(out, y):
    angular_diffs = sinCosToDiff(out, y)
    # Return mean angular difference
    return torch.mean(angular_diffs)

#Alternative error function for angles which uses median to be more robust to outliers in the prediction error.
#It turns out this is equivalent to measuring the "discrimination threshold!!!!"
def medianAngularDiff(out, y):
    angular_diffs = sinCosToDiff(out, y)
    return torch.median(angular_diffs)

#Error Functions:

#mean square error, summed over output classes to scale with C.
def SquareError(out,y, cum = False):
    
    if cum:
        return SquareError_cum(out, y)
    
    return torch.mean(torch.sum(torch.square(out-y), dim = 1))

#Assumes out_cum contains a cumulative prediction according to the ensembling function over dimension 1.
def SquareError_cum(out_cum, y, cum=True):
    
    assert cum==True, 'cum=False passed into SquareError_cum'
    
    y_replicated = y.unsqueeze(1).expand_as(out_cum) #P by K by C

    # Compute the square difference between out_cum and replicated y
    square_diff = torch.square(out_cum - y_replicated)

    # Take the mean over dim=0
    mean_square_error_cum = out_cum.shape[2]*square_diff.mean(dim=(0,2)) #Scale mean by C to match the scale of the other error functions

    return mean_square_error_cum

#mean square error, summed over output classes to scale with C.
#out is P by C
#y is P by C
def SquareError_rel(out,y, cum = False):
    
    if cum:
        return SquareError_rel_cum(out, y)

    y_sq_tot = torch.mean(torch.sum(torch.square(y), dim = 1))
    
    return torch.mean(torch.sum(torch.square(out-y), dim = 1))/y_sq_tot

#out_cum is P by K by C
#y is P by C
def SquareError_rel_cum(out_cum, y, cum=True):
    
    assert cum==True, 'cum=False passed into SquareError_rel_cum'
    
    y_replicated = y.unsqueeze(1).expand_as(out_cum) #P by K by C

    # Compute the square difference between out_cum and replicated y
    square_diff = torch.square(out_cum - y_replicated)

    square_diff_tot = torch.sum(square_diff, dim = 2) #Sum over classes

    #Compute mean square target values
    y_sq_tot = torch.mean(torch.sum(torch.square(y), dim = 1))

    square_diff_rel  = square_diff/y_sq_tot

    # Take the mean over dim=0, 2
    mean_square_error_rel_cum = square_diff_rel.mean(dim=(0,2))

    return mean_square_error_rel_cum

def SgnErrorRate(out, y, cum = False):
    
    if cum:
        return SgnErrorRate_cum(out, y)
    
    return 1/2*torch.mean(torch.abs(torch.sign(out)-torch.sign(y)))

def SgnErrorRate_cum(out_cum,y, cum=True):
    
    assert cum==True, 'cum=False passed into SgnErrorRate_cum'
    
    sgn_y_replicated = torch.sgn(y).unsqueeze(1).expand_as(out_cum)
    errors = 1/2*torch.abs(torch.sgn(out_cum) - sgn_y_replicated)    
    return torch.mean(errors, dim=(0,2))

#Functions for Timing Operations
def time_now():
    return datetime.now(tz).strftime("%m-%d_%H-%M")

def time_diff(t_a, t_b):
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

#Matrix square root.
def matrix_sqrt(A):
    L,V = torch.linalg.eigh(A)
    DiagL = torch.diag(L)
    return V @ torch.sqrt(DiagL) @ V.T

def makeCorrelatedMatrix(M, c):
    return (1-c)*torch.eye(M)+c*torch.ones(M)


#Gamma distribution subsampling:

#Generate K gamma distributed fractions with mean nu_mean and a standard deviation which is StdFrac times nu_mean
def drawGammaDistFracs(K, nu_mean, stdFrac, constrain = True, ceil = 1, floor = 0.002):
    if stdFrac==0:
        return nu_mean*np.ones(K)
    
    variance = (stdFrac*nu_mean)**2
    
    # Calculate the shape and scale parameters based on mean and variance
    shape = nu_mean ** 2 / variance
    scale = variance / nu_mean

    # Generate random numbers from the gamma distribution
    random_numbers = gamma.rvs(shape, scale=scale, size=K)
    
    #Constrain to sum to the correct number of synapses before applying cieling and floor
    if constrain:
        random_numbers = nu_mean*K*random_numbers/np.sum(random_numbers)
        
    #Numbers are capped at ceil
    if ceil is not None:
        random_numbers[random_numbers>ceil] = ceil
        
    if floor is not None:
        random_numbers[random_numbers<floor] = floor
        
    return random_numbers


    #Cerebellum Dataset Maker: