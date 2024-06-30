# This notebook centrally defines the various functions that are used for plotting, handling data, and calculating OT distances.

# Overview of functions (note that not all functions are used in this analysis):
# 
# **OT calculating functions:**
# 
# *   `calcGroundCostMatrix()`
#     -   `calcGroundCostMatrix_1DpT()`
#     -   `calcGroundCostMatrix_2D()`
#     -   `calcGroundCostMatrix_3D()`
# *   `calcOTDistance()`
# *   `calcOTDistance_non_square()`
# *   `parallel_OT_non_square()`
# *   `event_to_ensemble_dist()`
# *   `calcIndividualOTScores()`
# *   `checkSCHEMES()`
# *   `getMasses()`
# 
# **Data handling functions:**
# 
# *   `roundToSigFig()`
# *   `randomDataSample()`
# *   `calcROCmetrics()`
# *   `calcAUC()`
# *   `indxOfCertainTPR()`
# *   `getRepeatAvStd()`
# *   `calcWeightedComboOTscores()`
# *   `getFractionsOfMax()`
# *   `maxIndividualOTScore()`
# *   `calcMultiplicityData()`
# *   `calcMaxAvSI()`
# *   `calcMaxAvSI_helper()`
# *   `calcMaxAvF1()`
# *   `calcMaxAvF1_helper()`
# 
# **Background data augmentation functions:**
# 
# *   `check_data()`
# *   `add_one_hot_columns()`
# *   `format_for_analysis()`
# *   `augmentAndSaveData()`
# *   `neg_augs()`
# *   `add_objects()` (augmentation 1)
# *   `add_objects_constptmet()` (augmentation 2)
#       - `get_std_rivet()`
#       - `etaphi_smear_events()`
#       - `collinear_fill_e_mu()`
#       - `collinear_fill_jets()`
# *   `shift_met_or_pt()` (augmentation 3)
# *   `aug_2_jetConditions()`
# *   `aug_2_electronConditions()`
# *   `aug_2_muonsConditions()`
# *   `aug_2_conditions()`
# 
# **Plotting functions:**
# 
# *   Misc. helper functions
#     - `RGBAtoRGBAtuple()`
# *   Data plotting functions
#     - `plotDataHists()`
#     - `plotMultiplicityData()`
#     - `plotDataAugHists()`
# *   OT results plotting functions
#     - `plotScoreHists()`
#     - `plotMaxIndividualOTScoresPerEvent()`
# *   Performance metric plotting functions
#     - Repeat runs
#       - `plotROCcurve_errorBand()`
#       - `plotROCcurve_errorBand_specificMethod()`
#       - `plotInvROCcurve_errorBand()`
#       - `plotSIcurve_errorBand()`
#       - `plotSIcurve_errorBand_specificMethod()`
#     - Single test run
#       - `plotROCcurve()`
#       - `plotInvROCcurve()`
#       - `plotSIcurve()`
# 
# 
# **Machine learning functions:**
# 
# * SVM Classification
#    - `SVM_ROC_Metrics()`
#    - `SVM_Classification_With_Best_Hyperparameters()`
# * kNN Classification
#    - `kNN_with_score_list()`
#    - `kNN_with_distance_matrix()`
#    - `kNN_cross_validation()`
#    - `rNN_with_distance_matrix()`
#    - `rNN_cross_validation()`
#    - `SVM_with_distance_matrix()`
#    - `SVM_cross_validation()`
#    - `OneClassSVM_with_distance_matrix()`
#    - `kNN_ROC_metrics()`

# # Import libraries


import numpy as np
from numpy.random import RandomState
import numpy.ma as ma

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import h5py
import ot  # PyOT library: https://pythonot.github.io/index.html
from numpy.random import Generator, PCG64
from sklearn import metrics
import itertools

from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from sklearn import svm
import concurrent.futures as cf

# # OT calculating functions

# #### `calcGroundCostMatrix()`
# 
# Also contains
# - `calcGroundCostMatrix_1DpT()`
# - `calcGroundCostMatrix_2D()`
# - `calcGroundCostMatrix_3D()`

def calcGroundCostMatrix(xs, xt, COSTSCHEME):
  """
  Calculate ground cost matrix between xs and xt

  Inputs:
             xs:  3D (pT, eta, phi) coordinates of nx source particles (shape=(nx,3));
                  nx = number of entries in xEvents
             xt:  3D (pT, eta, phi) coordinates of ny source particles (shape=(ny,3));
                  ny = number of entries in yEvents
     COSTSCHEME:  Determines what scheme will be used to calculate the ground cost matrix. Options are:
                  - 1DpT: Ground space is pT only
                  - 2D:   Ground space is 2D (eta,phi); note mass is pT
                  - 3D:   Ground space is 3D (pT,eta,phi); note mass is uniform

  Output:
     Returns matrix of pair-wise |x - y|^2 distances

  """
  if COSTSCHEME=='1DpT':
    return calcGroundCostMatrix_1DpT(xs, xt)
  elif COSTSCHEME=='2D':
    return calcGroundCostMatrix_2D(xs, xt)
  elif COSTSCHEME=='3D':
    return calcGroundCostMatrix_3D(xs, xt)
  else:
    print("Error: Invalid COSTSCHEME")
    return 0

# ##### Auxiliary functions

def calcGroundCostMatrix_1DpT(xs, xt):
  """
  Auxiliary function, assumes ground space is pT only
  """
  deltaPT      = xs[:,0,None] - xt[:,0]
  return deltaPT**2

# **Note:** The following functions were modified from Tianji Cai's code below:
# 
# ```
# d_phis     = np.pi - np.abs(np.pi - np.abs(jet1_coords[:,1,None]-jet2_coords[:,1]))
# squareDist = (jet1_coords[:,0,None]-jet2_coords[:,0])**2 + d_phis**2
# ```
# 
# But also note that according to [this thread](https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point), this way of calculating modular differences may fail if deltaPhi_raw > 360 deg (2 pi).
# However, since the input is between -pi and pi, the max possible is 2pi so it shouldn't be a problem. It is also unsigned, however since we will square it that doesn't matter.

def calcGroundCostMatrix_2D(xs, xt):
  """
  Auxiliary function, assumes ground space is eta,phi.
  phi is 2pi modular which must be taken into account when computing the matrix of
  pair-wise |x - y|^2 distances
  """
  deltaEta     = xs[:,1,None] - xt[:,1]
  deltaPhi_raw = xs[:,2,None] - xt[:,2]
  deltaPhi     = np.pi - np.abs(np.pi - np.abs(deltaPhi_raw))
  return deltaEta**2 + deltaPhi**2

def calcGroundCostMatrix_3D(xs, xt):
  """
  Auxiliary function, assumes ground space is pT,eta,phi.
  phi is 2pi modular which must be taken into account when computing the matrix of
  pair-wise |x - y|^2 distances
  """
  deltaPT      = xs[:,0,None] - xt[:,0]
  deltaEta     = xs[:,1,None] - xt[:,1]
  deltaPhi_raw = xs[:,2,None] - xt[:,2]
  deltaPhi     = np.pi - np.abs(np.pi - np.abs(deltaPhi_raw))
  return deltaPT**2 + deltaEta**2 + deltaPhi**2

# #### `calcOTDistance()`

# **Note:** We're using the following references from the POT library ([here](https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html#sphx-glr-auto-examples-plot-ot-2d-samples-py) and [here](https://pythonot.github.io/quickstart.html#solving-optimal-transport))

def calcOTDistance(xEvents, yEvents, OTSCHEME, COSTSCHEME, kwargs={}, Matrix = False):
  """
  Solve the optimal transport problem and find the 2-Wasserstein distance
  for a set of source events (xEvents) and target events (yEvents) for a
  given ground cost function.

  Inputs:
        xEvents:  Array of sample of x-type events; shape (N, 19, 3)
        yEvents:  Array of sample of y-type events; shape (N, 19, 3)
       OTSCHEME:  Determines what scheme will be used to calculate the OT distance.
                  Note the exact meaning varies somewhat depending on the choice of COSTSCHEME.
                  It's a dictionary of 3 booleans cooresponding to whether the PT is normalized,
                  whether the OT calculation is balanced, and whether the zero padding should be
                  removed. Namely,
                  OTSCHEME['normPT']      ==True:  Means that the pT should be normalized;
                                         ==False:  Means that the pT should be unnormalized
                  OTSCHEME['balanced']    ==True:  Means that the OT calculation should be balanced;
                                         ==False:  Means that the OT calculation should be unbalanced
                  OTSCHEME['noZeroPad']   ==True:  Means that the zero padding should be removed;
                                         ==False:  Means that the zero padding should be kept
                  OTSCHEME['individualOT']==True:  Means that the OT calculation is done on each species separately;
                                         ==False:  Means that the OT calculation is done ignoring species type
     COSTSCHEME:  Determines what scheme will be used to calculate the ground cost matrix. Options are:
                  - 1DpT: Ground space is pT only
                  - 2D:   Ground space is 2D (eta,phi); note mass is pT
                  - 3D:   Ground space is 3D (pT,eta,phi); note mass is uniform
         kwargs:  Dictionary containing keyword arguments for unbalanced OT calculation; default is an empty dictionary
         MATRIX:  Whether to return wXY as an (N,N) array or as a (N*N,) array

  Outputs
     wXY:  Array of the corresponding (squared) 2-Wasserstein distances
  """

  #-- Sanity checks on inputs --#

  # Check schemes
  assert checkSCHEMES(OTSCHEME, COSTSCHEME)==True
  if OTSCHEME['balanced']==False:
    assert all([x in list(kwargs.keys()) for x in ['div', 'reg_m'] ])

  # Get number of signal and background events #! Make more general later to handle different number of x and y events
  assert(xEvents.shape[0] == yEvents.shape[0])
  N = xEvents.shape[0]

  #-- Preliminaries --#
  # Create objects for outputs
  wXY = np.zeros(shape=(N,N))

  #-- Loop over pairs of events --#
  # We use itertools to make looping more efficient (i.e. do double for loops)
  dummyArr = np.arange(N)
  wXY_list = []
  for (i, j) in tqdm(itertools.product(dummyArr, dummyArr), total=len(dummyArr)**2): #! Implement diagonal+upper-triangle restriction to save on computation time
    #-- Get source and target data points--#
    # Remove zero-padding if specified
    if OTSCHEME['noZeroPad']==True:
      sMask = ~np.all(xEvents[i, :, :] == 0., axis=1) # Mask to remove zero rows
      tMask = ~np.all(yEvents[j, :, :] == 0., axis=1)

      xs = xEvents[i, :, :][sMask] # "source data points"
      xt = yEvents[j, :, :][tMask] # "target data points"
    else:
      xs = xEvents[i, :, :] # "source data points"
      xt = yEvents[j, :, :] # "target data points"

    #-- Get masses --#
    a,b = getMasses(xs, xt, OTSCHEME, COSTSCHEME)

    #-- Normalize pT coordinate if specified --#
    #! Might be more efficient to do this before getMasses() but then will need to change getMasses()
    if OTSCHEME['normPT']==True:
      totPTs, totPTt = xs[:,0].sum(), xt[:,0].sum()
      xs[:,0], xt[:,0] = xs[:,0]/totPTs, xt[:,0]/totPTt

    #-- Get cost function and append to list --#
    cxy = calcGroundCostMatrix(xs, xt, COSTSCHEME)

    #-- Calculate the unbalanced Wasserstein distance --#
    if OTSCHEME['balanced']==False:
      if j >= i:
        wXY[i,j] = ot.unbalanced.mm_unbalanced2(a, b, cxy, reg_m=kwargs['reg_m'], div=kwargs['div'])
        wXY_list.append(wXY[i,j])
      else:
        wXY[i,j] = wXY[j,i]
    else:
      if j >= i:
        wXY[i,j] = ot.emd2(a, b, cxy)
        wXY_list.append(wXY[i,j])
      else:
        wXY[i,j] = wXY[j,i]

  if Matrix == True:
    return wXY

  else:
    return np.asarray(wXY_list)

# #### `calcOTDistance_non_square()`

def calcOTDistance_non_square(xEvents, yEvents, OTSCHEME, COSTSCHEME, kwargs={}, Matrix = False):
  """
  Solve the optimal transport problem and find the 2-Wasserstein distance
  for a set of source events (xEvents) and target events (yEvents) for a
  given ground cost function.

  Inputs:
        xEvents:  Array of sample of x-type events; shape (N, 19, 3)
        yEvents:  Array of sample of y-type events; shape (M, 19, 3)
       OTSCHEME:  Determines what scheme will be used to calculate the OT distance.
                  Note the exact meaning varies somewhat depending on the choice of COSTSCHEME.
                  It's a dictionary of 3 booleans cooresponding to whether the PT is normalized,
                  whether the OT calculation is balanced, and whether the zero padding should be
                  removed. Namely,
                  OTSCHEME['normPT']      ==True:  Means that the pT should be normalized;
                                         ==False:  Means that the pT should be unnormalized
                  OTSCHEME['balanced']    ==True:  Means that the OT calculation should be balanced;
                                         ==False:  Means that the OT calculation should be unbalanced
                  OTSCHEME['noZeroPad']   ==True:  Means that the zero padding should be removed;
                                         ==False:  Means that the zero padding should be kept
                  OTSCHEME['individualOT']==True:  Means that the OT calculation is done on each species separately;
                                         ==False:  Means that the OT calculation is done ignoring species type
     COSTSCHEME:  Determines what scheme will be used to calculate the ground cost matrix. Options are:
                  - 1DpT: Ground space is pT only
                  - 2D:   Ground space is 2D (eta,phi); note mass is pT
                  - 3D:   Ground space is 3D (pT,eta,phi); note mass is uniform
         kwargs:  Dictionary containing keyword arguments for unbalanced OT calculation; default is an empty dictionary
         MATRIX:  Whether to return wXY as an (N,M) array or as a (N*M,) array

  Outputs

     wXY:  Matrix of the corresponding (squared) 2-Wasserstein distances, or list for default
  """

  #-- Sanity checks on inputs --#

  # Check schemes
  assert checkSCHEMES(OTSCHEME, COSTSCHEME)==True
  if OTSCHEME['balanced']==False:
    assert all([x in list(kwargs.keys()) for x in ['div', 'reg_m'] ])

  N = xEvents.shape[0]
  M = yEvents.shape[0]

  #-- Preliminaries --#
  # Create objects for outputs
  wXY = np.zeros(shape=(N,M))

  #-- Loop over pairs of events --#
  # We use itertools to make looping more efficient (i.e. do double for loops)
  dummyArr_1 = np.arange(N)
  dummyArr_2 = np.arange(M)
  wXY_list = []
  for (i, j) in itertools.product(dummyArr_1, dummyArr_2):
    #-- Get source and target data points--#
    # Remove zero-padding if specified
    if OTSCHEME['noZeroPad']==True:
      sMask = ~np.all(xEvents[i, :, :] == 0., axis=1) # Mask to remove zero rows
      tMask = ~np.all(yEvents[j, :, :] == 0., axis=1)

      xs = xEvents[i, :, :][sMask] # "source data points"
      xt = yEvents[j, :, :][tMask] # "target data points"
    else:
      xs = xEvents[i, :, :] # "source data points"
      xt = yEvents[j, :, :] # "target data points"

    #-- Get masses --#
    a,b = getMasses(xs, xt, OTSCHEME, COSTSCHEME)

    #-- Normalize pT coordinate if specified --#
    #! Might be more efficient to do this before getMasses() but then will need to change getMasses()
    if OTSCHEME['normPT']==True:
      totPTs, totPTt = xs[:,0].sum(), xt[:,0].sum()
      xs[:,0], xt[:,0] = xs[:,0]/totPTs, xt[:,0]/totPTt

    #-- Get cost function and append to list --#
    cxy = calcGroundCostMatrix(xs, xt, COSTSCHEME)

    #-- Calculate the unbalanced Wasserstein distance --#
    if OTSCHEME['balanced']==False:
      wXY[i,j] = ot.unbalanced.mm_unbalanced2(a, b, cxy, reg_m=kwargs['reg_m'], div=kwargs['div'])
      wXY_list.append(wXY[i,j])
    else:
      wXY[i,j] = ot.emd2(a, b, cxy)
      wXY_list.append(wXY[i,j])

  if Matrix == True:
    return wXY

  else:
    return np.asarray(wXY_list)

# #### `parallel_OT_non_square()`

def parallel_OT_non_square(event_set1, event_set2, OTSCHEME, COSTSCHEME, kwargs={}, num_cores = None):
    """
    Calls calcOTDistance_non_square() in parallel

    Inputs:
     event_set1:  Array of sample of type-1 events; shape (N, 19, 3)
     event_set2:  Array of sample of type-2 events; shape (M, 19, 3)
       OTSCHEME:  Determines what scheme will be used to calculate the OT distance.
                  Note the exact meaning varies somewhat depending on the choice of COSTSCHEME.
                  It's a dictionary of 3 booleans cooresponding to whether the PT is normalized,
                  whether the OT calculation is balanced, and whether the zero padding should be
                  removed. Namely,
                  OTSCHEME['normPT']      ==True:  Means that the pT should be normalized;
                                         ==False:  Means that the pT should be unnormalized
                  OTSCHEME['balanced']    ==True:  Means that the OT calculation should be balanced;
                                         ==False:  Means that the OT calculation should be unbalanced
                  OTSCHEME['noZeroPad']   ==True:  Means that the zero padding should be removed;
                                         ==False:  Means that the zero padding should be kept
                  OTSCHEME['individualOT']==True:  Means that the OT calculation is done on each species separately;
                                         ==False:  Means that the OT calculation is done ignoring species type
     COSTSCHEME:  Determines what scheme will be used to calculate the ground cost matrix. Options are:
                  - 1DpT: Ground space is pT only
                  - 2D:   Ground space is 2D (eta,phi); note mass is pT
                  - 3D:   Ground space is 3D (pT,eta,phi); note mass is uniform
         kwargs:  Dictionary containing keyword arguments for unbalanced OT calculation; default is an empty dictionary
      num_cores:  Number of cores to use in parallelization
    """
    if num_cores is None:
      import os
      num_cores = os.cpu_count()
    num_events = len(event_set1)
    assert num_events%num_cores == 0, 'Number of type-1 events must be divisible by number of cores'
    num_per_core = num_events//num_cores
    with cf.ProcessPoolExecutor() as executor:

        futures = [executor.submit(calcOTDistance_non_square, event_set1[num_per_core*i:num_per_core*(i+1)], event_set2, OTSCHEME, COSTSCHEME, kwargs=kwargs, Matrix = True) for i in range(num_cores)]

    results = [f.result() for f in futures]

    distance_matrix = np.vstack(results)

    return distance_matrix

# #### `event_to_ensemble_dist()`

def event_to_ensemble_dist(wXY, EVENT_TO_ENSEMBLE_TYPE, AXIS=0):
    """
    Calculate the event to ensemble distance.
    wXY should be an NxN matrix where wXY[i,j] is the OT distance between X[i] and Y[j]
    AXIS=0 assumes that X events are the reference population (i.e. ensemble)
    """
    if EVENT_TO_ENSEMBLE_TYPE=='MAX':
        return np.max(wXY, axis=AXIS)
    elif EVENT_TO_ENSEMBLE_TYPE=='MIN':
        return np.min(wXY, axis=AXIS)
    elif EVENT_TO_ENSEMBLE_TYPE=='MEAN':
        return np.mean(wXY, axis=AXIS)
    else:
        print("ERROR: Invalid EVENT_TO_ENSEMBLE_TYPE ",EVENT_TO_ENSEMBLE_TYPE)
        return 0

# #### `calcIndividualOTScores()`

def calcIndividualOTScores(trimmedDataDict, sigAliasList, OTSCHEME, COSTSCHEME, kwargs={}, speciesList=['MET', 'e', 'mu', 'jet']):
  """
  Calculate individual OT scores and store them in a score dictionary.

  Inputs:
    trimmedDataDict:  Dictionary of data trimmed to size (N, 19, 3); contains
                      two samples of background data ('bkgEvents1' and 'bkgEvents2')
                      and one sample of signal data for each signal type ('sig_A',
                      'sig_h0', 'sig_hch', 'sig_LQ')
    sigAliasList:     List of signal type aliases; i.e. ['sig_A', 'sig_h0', 'sig_hch', 'sig_LQ']
    speciesList:      List of species; default ['MET', 'e', 'mu', 'jet']

  Outputs:
    scoreDict: Dictionary of scores for OT on each particle species
  """
  scoreDict = {}

  #-- Loop over particle species
  for species in speciesList:
    scoreDict[species] = {}

    #-- Set background data according to species type --#
    if species == 'MET':
      B1_data = trimmedDataDict['bkgEvents1'][:, 0, :].reshape(-1,1,3)
      B2_data = trimmedDataDict['bkgEvents2'][:, 0, :].reshape(-1,1,3)
    elif species == 'e':
      B1_data = trimmedDataDict['bkgEvents1'][:, 1:5, :]
      B2_data = trimmedDataDict['bkgEvents2'][:, 1:5, :]
    elif species == 'mu':
      B1_data = trimmedDataDict['bkgEvents1'][:, 5:9, :]
      B2_data = trimmedDataDict['bkgEvents2'][:, 5:9, :]
    elif species == 'jet':
      B1_data = trimmedDataDict['bkgEvents1'][:, 9:, :]
      B2_data = trimmedDataDict['bkgEvents2'][:, 9:, :]

    #-- Calculate OT distance for background-to-background case --#
    _, scoreDict[species]['wBB'] = calcOTDistance(B1_data, B2_data, OTSCHEME=OTSCHEME, COSTSCHEME=COSTSCHEME, kwargs=kwargs)

    #-- Loop over signal cases to calculate OT distance in background-to-signal case --#
    for alias in sigAliasList:

      #-- Set signal data according to species type --#
      if species == 'MET':
        S_data = trimmedDataDict[alias][:, 0, :].reshape(-1,1,3)
      elif species == 'e':
        S_data = trimmedDataDict[alias][:, 1:5, :]
      elif species == 'mu':
        S_data = trimmedDataDict[alias][:, 5:9, :]
      elif species == 'jet':
        S_data = trimmedDataDict[alias][:, 9:, :]

      #-- Calculate OT distance for background-to-signal case --#
      name_w = 'wBS_'+alias
      _, scoreDict[species][name_w] = calcOTDistance(B1_data, S_data, OTSCHEME=OTSCHEME, COSTSCHEME=COSTSCHEME, kwargs=kwargs)

  return scoreDict

# #### `checkSCHEMES()`

def checkSCHEMES(OTSCHEME, COSTSCHEME):
  """
  Auxiliary function to make calcOTDistance() a bit tidier while still checking that the schemes make sense

  See calcOTDistance for definition of OTSCHEME, COSTSCHEME
  """

  #-- Check that SCHEMES contain valid entries --#
  assert (COSTSCHEME in ['1DpT','2D', '3D']), "Error: Invalid COSTSCHEME"
  for key in OTSCHEME.keys():
    assert(OTSCHEME[key] in [True, False]), "Error: Invalid OTSCHEME"


  #-- Check that SCHEME combinations make sense --#

  if COSTSCHEME == '1DpT':

    # pT assumed to be unnormalized
    assert (OTSCHEME['normPT']==False), "Error: Invalid OTSCHEME['normPT'] for COSTSCHEME==%s"%(COSTSCHEME)

    # either balanced or unbalanced is fine
    # => nothing to check

    # if doing balanced then either removing or keeping zero padding is fine => nothing to check;
    # if doing unbalanced then zero padding should be removed
    if OTSCHEME['balanced']==False:
      assert (OTSCHEME['noZeroPad']==True), "Error: Invalid OTSCHEME['noZeroPad'] for COSTSCHEME==%s"%(COSTSCHEME)

    # check that we're not doing individual OT
    assert (OTSCHEME['individualOT']==False), "Error: Invalid OTSCHEME['individualOT'] for COSTSCHEME==%s"%(COSTSCHEME)

  elif COSTSCHEME == '2D':

    # either normalized or unnormalized pT is fine
    # => nothing to check

    # if pT is unnormalized we must be doing unbalanced
    # if pT is normalized we assume we're doing balanced
    if OTSCHEME['normPT']==False:
      assert (OTSCHEME['balanced']==False), "Error: Invalid OTSCHEME['balanced'] for COSTSCHEME==%s"%(COSTSCHEME)
    elif OTSCHEME['normPT']==True:
      assert (OTSCHEME['balanced']==True), "Error: Invalid OTSCHEME['balanced'] for COSTSCHEME==%s"%(COSTSCHEME)

    # check that we're not doing individual OT
    assert (OTSCHEME['individualOT']==False), "Error: Invalid OTSCHEME['individualOT'] for COSTSCHEME==%s"%(COSTSCHEME)

  elif COSTSCHEME == '3D':

    # either normalized or unnormalized pT is fine
    # => nothing to check

    # either balanced or unbalanced is fine
    # => nothing to check

    # either removing or keeping zero padding is fine
    # => nothing to check

    # if we're doing individual species we want balanced OT with unnormalized pT and zero padding intact
    if OTSCHEME['individualOT'] == True:
      assert (OTSCHEME['balanced']==True),   "Error: Invalid OTSCHEME['balanced'] for individualOT and COSTSCHEME==%s"%(COSTSCHEME)
      assert (OTSCHEME['normPT']==False),    "Error: Invalid OTSCHEME['normPT'] for individualOT and COSTSCHEME==%s"%(COSTSCHEME)
      assert (OTSCHEME['noZeroPad']==False), "Error: Invalid OTSCHEME['noZeroPad'] for individualOT and COSTSCHEME==%s"%(COSTSCHEME)


  # If all assert statements passed without issue then return True
  return True


# #### `getMasses()`

def getMasses(xs, xt, OTSCHEME, COSTSCHEME):
  """
  Auxiliary function to get probability masses based on OTSCHEME and COSTSCHEME.
  There are 5 different options.
  """
  #-- Option 1:  m_i = 1/19 --#
  # There are 3 combinations that give this
  # They assume 1D or 3D, balanced, and zero-padded
  if (COSTSCHEME != '2D') and (OTSCHEME['balanced']==True) and (OTSCHEME['noZeroPad']==False) and (OTSCHEME['individualOT']==False):
    a = np.ones((19,)) / 19
    b = a

  #-- Option 2: m_i = 1, 1/4, or 1/10 depending on individual species type --#
  elif (COSTSCHEME != '2D') and (OTSCHEME['balanced']==True) and (OTSCHEME['noZeroPad']==False) and (OTSCHEME['individualOT']==True):
    nx, ny = xs.shape[0], xt.shape[0]
    a = np.ones((nx,)) / nx
    b = np.ones((ny,)) / ny

  #-- Option 3: m_i = 1/N, N=number of particles in the event --#
  # There are 3 combinations that give this
  # They assume 1D or 3D, balanced, and zero-padding removed
  elif (COSTSCHEME != '2D') and (OTSCHEME['balanced']==True) and (OTSCHEME['noZeroPad']==True):
    nx, ny = xs.shape[0], xt.shape[0]
    a = np.ones((nx,)) / nx
    b = np.ones((ny,)) / ny

  #-- Option 4: m_i = pT --#
  # There is only 1 combination that gives this
  elif (COSTSCHEME == '2D') and (OTSCHEME['normPT']==False):
    a = xs[:,0]
    b = xt[:,0]

  #-- Option 5: m_i = pT/sum(pT) --#
  # There is only 1 combination that gives this
  elif (COSTSCHEME == '2D') and (OTSCHEME['normPT']==True):
    totalpT_xs = np.sum(xs[:,0]) # Total pT in each x-event
    totalpT_xt = np.sum(xt[:,0]) # Total pT in each y-event

    a = np.ascontiguousarray(xs[:,0]/totalpT_xs) # POT error will result if not C-contiguous
    b = np.ascontiguousarray(xt[:,0]/totalpT_xt)

    # Assuming balanced OT, both total masses should be the same (up to some precision)
    # Using same precision considerations as POT library which is the default for np.testing.assert_almost_equal() (decimal=7)
    np.testing.assert_almost_equal(a.sum(0), b.sum(0,keepdims=True), err_msg='a and b vector must have the same sum')

  #-- Option 6: m_i = 1 --#
  else:
    nx, ny = xs.shape[0], xt.shape[0]
    a = np.ones((nx,))
    b = np.ones((ny,))

  return a, b

# # Data handling functions

# ##### `roundToSigFig()`

#-- Round num to n significant digits --#
# Note that half is not always rounded up (better strategy for scientific applications)
#   Ex. roundToSigFig(0.090215, 4)  ->   0.09022
#       roundToSigFig(0.090225, 4)  ->   0.09022
# https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
def roundToSigFig(num, n):
  return '{:g}'.format(float('{:.{p}g}'.format(num, p=n)))

# ##### `loadJSONFile()`

def loadJSONFile(filepath, NREPEAT=5, INVERTED=False):
  """
  Loads JSON file with assumed structure, and returns the dictionary.

  Structure of JSON:
    'repeat0'
       'sig_A'
          List of 'auc', 'fpr', 'tpr', 'SI', 'fprInv', 'F1' in that order.
       'sig_h0'
          ...
       'sig_hch'
          ...
       'sig_LQ'
          ...
    'repeat1'
    ...
    repeatN

  Returned dictionary structure expands lists into separate keys with the corresponding names.
  """

  #-- Open file and load dictionary --#
  f    = open(filepath)
  data = json.load(f)

  #-- Create new dictionary --#
  ROCmetricList = ['auc', 'fpr', 'tpr', 'SI', 'fprInv', 'F1']
  newDict = {}
  if INVERTED:
    REPEATLIST = ['repeat%d'%i for i in range(NREPEAT)] # ['repeat0', ...,]
  else:
    REPEATLIST = list(data.keys()) # ['repeat0', ...,]

  for key in REPEATLIST:
    newDict[key] = {}

    for subkey in ['sig_A', 'sig_h0', 'sig_hch', 'sig_LQ']:
      newSubkeyName = 'ROC_metric_'+subkey
      newDict[key][newSubkeyName] = {}
      for i in range(len(ROCmetricList)):

        if INVERTED:
          ROCmetric = np.array(data[subkey][key][i], dtype=np.float64)
        else:
          ROCmetric = np.array(data[key][subkey][i], dtype=np.float64)

        if ROCmetricList[i] == 'fprInv':
          # None entries come from division by zero, so for these cases we impose a regulator (inf approx 1/0.0001)
          ROCmetric = np.nan_to_num(ROCmetric, nan=1/0.0001)
          # maskNones = ROCmetric==None
          # ROCmetric[maskNones] = np.repeat(1/0.0001, ROCmetric[maskNones].shape[0])

        newDict[key][newSubkeyName][ROCmetricList[i]] = ROCmetric

  return newDict

# ##### `randomDataSample()`

def randomDataSample(data, nEvents, random_state):
  """
  Generate a random sample of data by generating a 1D array of nEvents uniform
  random integers and returning the events corresponding to these integers.

  Inputs:
    data:          Total data array of shape (nTotal, 19, 3)
    nEvents:       Number of events to sample
    random_state:  The generator to use to generate the samples (for reproducibility)

  Outputs:
    Selected events; shape (nEvents, 19, 3)
  """
  #! Pretty slow in practice, make it faster later

  nTotal = data.shape[0]
  randomIntArray = random_state.integers(0,nTotal,nEvents)

  return data[randomIntArray, :, :]

# ##### `calcROCmetrics()`

def calcROCmetrics(scoreBkg, scoreSigList, SIreg=0.0001, INTERPOLATE=True):
    """
    Calculate 4 metrics related to ROC curve components:
        - AUC
        - Background efficiency (aka FPR or eps_B), Signal efficiency (aka TPR or eps_S)
        - Significance Improvement (SI) defined as eps_S/sqrt(eps_B + SIreg) <=> TPR/sqrt(FPR + SIreg)
          where SIreg is a regulator for statistical fluctuations at low efficiency; SIreg=0.01% by default
          Reference: https://arxiv.org/pdf/2001.05001.pdf
        - Inverse Background efficiency (aka FPR^{-1} or eps_B^{-1}); division by zero is masked

    Inputs:
      scoreBkg:      Anomaly score values for N background events; shape (N,)
      scoreSigList:  List of anomaly score values for each signal type case; Length=nCases
                    scoreSigList[i] is the anomaly score for M signal events of the ith signal type case; shape (M,)
      SIreg:         Regulator to prevent division by zero when calculating SI metric; default 0.01%
      INTERPOLATE:   Whether to interpolate to a standard array of TPR values

    Outputs:
      aucList:       List of AUC scores for each background to signal type pairing; Length=nCases
      fprList:       List of FPR arrays for each background to signal type pairing; Length=nCases
                    fprList[i] is an array of shape (Q,) with Q>2
      tprList:       List of TPR arrays for each background to signal type pairing; Length=nCases
                    tprList[i] is an array of shape (Q,) with Q>2
      SIList:        List of SI arrays for each background to signal type pairing; Length=nCases
                    SIList[i] is an array of shape (Q,) with Q>2
      fprInvList:    List of inverse FPR arrays for each background to signal type pairing; Length=nCases
                    fprInvList[i] is a masked array of shape (Q,) with Q>2 with division by zero cases masked
      F1List:        List of F1 score arrays for each background to signal type pairing; Length=nCases
                    F1List[i] is an array of shape (Q,) with Q>2
    """

    #-- Preliminaries --#
    nCases    = len(scoreSigList) # Number of signal cases
    aucList, fprList, tprList, SIList, fprInvList, F1List = [], [], [], [], [], [] # Create lists to store results

    #-- Loop over signal cases --#
    for i in range(nCases):
      scoreSig = scoreSigList[i]

      #-- Calculate AUC (and, internally, label and score inputs for sklearn's function) --#
      auc, labels, scores = calcAUC(scoreBkg, scoreSig)
      aucList.append(auc)

      #-- Calculate other ROC curve metrics --#
      fpr_raw, tpr_raw, _ = metrics.roc_curve(labels, scores)

      if INTERPOLATE:
        #https://stats.stackexchange.com/questions/186337/average-roc-for-repeated-10-fold-cross-validation-with-probability-estimates
        base_tpr = np.linspace(0, 1, 101) # 0.00, 0.01, ..., 1.0
        tpr = base_tpr
        fpr = np.interp(base_tpr, tpr_raw, fpr_raw)
      else:
        fpr = fpr_raw
        tpr = tpr_raw

      fprList.append(fpr)
      tprList.append(tpr)

      #-- Calculate SI metric --#
      # Def: eps_S/sqrt(eps_B + SIreg) <=> TPR/sqrt(FPR + SIreg)
      fpr_sqrt = np.sqrt(fpr + SIreg)
      SI = tpr/fpr_sqrt
      SIList.append(SI)

      #-- Calculate inverse FPR metric --#
      #! Should we also use regulator here? Is this typical?
      fpr_masked = ma.masked_where(fpr==0., fpr) # Get rid of possibility of dividing by zero
      fprInv = 1./fpr_masked
      fprInvList.append(fprInv)

      #-- Calculate F1 score metric --#
      # Recall that
      #   P   is the number of signal and N is the number of background
      #   TP  is the number of true positives, FP is the number of false positives, and FN is the number of false negatives
      #   TPR = TP/P, FPR = FP/N, and FNR = FN/P
      #   FNR = 1 - TPR
      #
      # Def of F1: (2*TP)/(2*TP + FP + FN)
      P   = scoreSig.shape[0] # Number of signal
      N   = scoreBkg.shape[0] # Number of background
      tp  = P*tpr
      fp  = N*fpr
      fn  = P*(1-tpr) # P*fnr
      F1  = (2*tp)/(2*tp + fp + fn)
      F1List.append(F1)

    return aucList, fprList, tprList, SIList, fprInvList, F1List

# ##### `calcAUC()`

def calcAUC(scoreBkg, scoreSig):
  """
  Given the anomaly scores for background and signal events, calculate the AUC.

  Inputs:
    scoreBkg:   Anomaly score values for N background events; shape (N,)
    scoreSig:   Anomaly score values for M signal events; shape (M,)

  Outputs:
    auc:        Area Under the Curve (AUC) value; float
    labels:     Numeric background/signal labels; 0 for background, 1 for signal
                (necessary for ROC and AUC calculations); shape (N+M,)
    scores:     Concatenated anomaly scores from background and signal events;
                (necessary for ROC and AUC calculations); shape (N+M,)

  NOTE: Using the sklearn.metrics.roc_auc_score() function
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
  """

  labelsB = np.repeat(0, scoreBkg.shape[0]) # Background labels
  labelsS = np.repeat(1, scoreSig.shape[0]) # Signal labels
  labels  = np.concatenate((labelsB, labelsS))

  scores  =  np.concatenate((scoreBkg, scoreSig))

  auc = metrics.roc_auc_score(labels, scores)

  return auc, labels, scores

# ##### `indxOfCertainTPR()`

def indxOfCertainTPR(tprList, TPRval = 0.3):
  """
  For each TPR array in tprList, get the index corresponding to the TPR value
  which is closest to TPRval. This can be used to examine the other metrics at
  a certain TPR (signal efficiency, \eps_S) value.

  Inputs:
    tprList:   List of TPR arrays for each background to signal type pairing; Length=number of signal cases
               tprList[i] is an array of shape (Q,) with Q>2
    TPRval:    Fixed reference TPR value; default is TPR = \eps_S = 30%

  Outputs:
    indxList:  List of indices corresponding to TPR ~= TPRval; length = number of signal cases
  """
  indxList = []
  nCases = len(tprList) # Number of signal cases

  for i in range(nCases):
    tprArr = tprList[i]
    difference_array = np.absolute(tprArr-TPRval)
    indx = difference_array.argmin()
    indxList.append(indx)

  return indxList

# ##### `getRepeatAvStd()`

def getRepeatAvStd(scoreDict, sigAliasList=['sig_A', 'sig_h0', 'sig_hch', 'sig_LQ'], NREPEAT=5):
  """
  Calculates average and standard deviation over repeats for the various ROC metrics: auc, fpr, SI, fprInv, F1.
  For auc, the av and std are each a single number. Whereas, for fpr, SI, fprInv, F1 the av and std will each be arrays of the same respective size.

  Inputs:
      scoreDict:  Dictionary containing ROC metrics (auc, fpr, SI, fprInv, F1) for all signal cases. There are NREPEAT copies.
                  Structure is assumed to be the following:
                      scoreDict['repeat0']
                                          ['ROC_metric_%s']%sigAliasList[0]
                                                          ['auc']    # Shape = (NREPEAT,)
                                                          ['fpr']    # Shape = (NREPEAT,n_Thresholds)
                                                          ['SI']     # Shape = (NREPEAT,n_Thresholds)
                                                          ['fprInv'] # Shape = (NREPEAT,n_Thresholds)
                                                          ['F1']     # Shape = (NREPEAT,n_Thresholds)
                                          ...
                                          ['ROC_metric_%s']%sigAliasList[3]
                      scoreDict['repeat1']
                      ...
                      scoreDict['repeat%s'%NREPEAT]

      sigAliasList:  List of signal type alias
          NREPEAT:  Number of repeated test sample sets.

  NOTE: Assumes that calcROCmetrics() was calculated with INTERPOLATE=True => TPR is fixed to baseTPR.
          This means n_Thresholds=101, corresponding to using baseTPR = np.linspace(0,1,101).

  Outputs:
      N/a, updates scoreDict with values under key 'avStdQuantities' for each signal type. For example,
          scoreDict['avStdQuantities']['sig_A']['auc']['mean']
                                                      ['std']
  """

  REPEATLIST = ['repeat%d'%i for i in range(NREPEAT)]
  scoreDict['avStdQuantities'] = {}

  # Loop over signal types
  for alias in sigAliasList:
    print("Analyzing signal type = %s "%alias)
    name = 'ROC_metric_%s'%(alias)
    scoreDict['avStdQuantities'][alias]                   = {}

    # Get average and std of desired quantities
    for quantity in ['auc', 'fpr', 'SI', 'fprInv', 'F1']:

      qArr = np.array([scoreDict[key][name][quantity] for key in REPEATLIST ])
      scoreDict['avStdQuantities'][alias][quantity]         = {}
      scoreDict['avStdQuantities'][alias][quantity]['mean'] = qArr.mean(axis=0)
      scoreDict['avStdQuantities'][alias][quantity]['std']  = qArr.std(axis=0)

# ##### `calcWeightedComboOTscores()`

def calcWeightedComboOTscores(scoreDict, wList=[1., 1., 1., 1.]):
  """
  calculate a combination of individual species OT scores for each subkey case

  Inputs:
    scoreDict:  Dictionary of scores for OT on each particle species; e.g. scoreDict['MET']['wBB']
    wList:      How much to weight 'MET', 'e', 'mu', 'jet' information, respectively, in the sum;
                default equal weighting wList=[1., 1., 1., 1.]
  Outputs:
    Updated scoreDict
  """
  #-- Calculate and store combination (sum) --#
  scoreDict['combo'] = {}
  for subkey in scoreDict['MET'].keys():
    nameCombo = 'combo_'+str(subkey)
    scoreDict['combo'][nameCombo]  = wList[0]*scoreDict['MET'][subkey] + wList[1]*scoreDict['e'][subkey] + wList[2]*scoreDict['mu'][subkey] + wList[3]*scoreDict['jet'][subkey]

  return scoreDict

# ##### `getFractionsOfMax()`

def getFractionsOfMax(indxs, val):
  """
  Calculates the fraction of indxs entries that equal val
  """

  indxs_masked = ma.masked_where(indxs==val, indxs)
  total        = indxs.shape[0]
  fraction     = float(np.sum(indxs_masked.mask))/ float(total)

  return fraction

# ##### `maxIndividualOTScore()`

def maxIndividualOTScore(scoreDict, alias):
  """
  Calculates which individual OT score is the largest for each event in signal type alias
  """

  # Get individual OT scores from dictionary
  metArr = scoreDict['MET'][alias].reshape(-1, 1)
  eArr   = scoreDict['e'][alias].reshape(-1, 1)
  muArr  = scoreDict['mu'][alias].reshape(-1, 1)
  jetArr = scoreDict['jet'][alias].reshape(-1, 1)

  comboArr = np.concatenate((metArr, eArr, muArr, jetArr), axis=1)

  # Get index of max individual OT score for each event
  # 0=met, 1=e, 2=mu, 3=jet
  indxs  = np.argmax(comboArr, axis=1)
  maxArr = np.max(comboArr, axis=1)

  # Print fractions
  print("Signal Type = ", alias)
  print("Fraction of times that each individual OT score was the maximum for a given event:")
  print("   MET: %s "%str(getFractionsOfMax(indxs, 0)*100))
  print("     e: %s "%str(getFractionsOfMax(indxs, 1)*100))
  print("    mu: %s "%str(getFractionsOfMax(indxs, 2)*100))
  print("   jet: %s "%str(getFractionsOfMax(indxs, 3)*100))

  return indxs, maxArr

# ##### `calcMultiplicityData()`

def calcMultiplicityData(objectsBkg, objectsSigList):
  """
  Inputs:
    objectsBkg:       pT of all objects for each background event; ndarray of shape
                      (Nb, 19) where Nb is the number of background events
    objectsSigList:   List of arrays of pT of all objects for each signal event;
                      element of list is ndarray of shape (Ns, 19), where Ns, the
                      number of signal events, varies depending on the signal type;
                      signal types in list is 'sig_A', 'sig_h0', 'sig_hch', 'sig_LQ'
  Outputs:
    multBkgList:      List of arrays corresponding to electron, muon, jet, and total multiplicities
    multSigList:      List of list of arrays corresponding to electron, muon, jet, and total multiplicities for each signal type
                      E.g. multSigList[0] is the list of electron multiplicities for all signal types
                      Total multiplicity is defined as multiplicity of all particle-type objects (i.e. excluding MET which is always present)
  """

  #-- Get multiplicities for background data --#
  multElectrons_Bkg = np.count_nonzero(objectsBkg[:, 1:5], axis=1)
  multMuons_Bkg     = np.count_nonzero(objectsBkg[:, 5:9], axis=1)
  multJets_Bkg      = np.count_nonzero(objectsBkg[:, 9:19], axis=1)
  multTotal_Bkg     = np.count_nonzero(objectsBkg[:, 1:19], axis=1) # Exclude MET since it is always present

  multBkgList = [multElectrons_Bkg, multMuons_Bkg, multJets_Bkg, multTotal_Bkg]

  #-- Get multiplicities for signal data --#
  listMultElectrons_Sig, listMultMuons_Sig, listMultJets_Sig, listMultTotal_Sig = [],[],[],[]
  nSignalCategories = len(objectsSigList)
  for i in range(nSignalCategories):
    objectsSig = objectsSigList[i]

    listMultElectrons_Sig.append(np.count_nonzero(objectsSig[:, 1:5], axis=1))
    listMultMuons_Sig.append(np.count_nonzero(objectsSig[:, 5:9], axis=1))
    listMultJets_Sig.append(np.count_nonzero(objectsSig[:, 9:19], axis=1))
    listMultTotal_Sig.append(np.count_nonzero(objectsSig[:, 1:19], axis=1))

  multSigList = [listMultElectrons_Sig, listMultMuons_Sig, listMultJets_Sig, listMultTotal_Sig]

  return multBkgList, multSigList

# ##### `calcMaxAvSI()`

def calcMaxAvSI(scoreDict, sigAliasList=['sig_A', 'sig_h0', 'sig_hch', 'sig_LQ'], minTPR=0.05, NSIGFIGS=4, SIMPLE=False):

  print("Max SI (TPR):")
  # Loop over signal types
  for alias in sigAliasList:

    #-- Get SI curve components --#
    av_si  = scoreDict['avStdQuantities'][alias]['SI']['mean']
    std_si = scoreDict['avStdQuantities'][alias]['SI']['std']

    #-- Get max av SI, std, and corresponding TPR in range [minTPR, 1] --#
    av_maxSI, std_maxSI, corr_tpr = calcMaxAvSI_helper(av_si, std_si, minTPR=minTPR)

    # Report max in trimmed TPR range
    if SIMPLE:
      print("    %s $\pm$ %s (%s)"%(roundToSigFig(av_maxSI, NSIGFIGS), roundToSigFig(std_maxSI, NSIGFIGS), roundToSigFig(corr_tpr, NSIGFIGS)))
    else:
      print("    %s is %s $\pm$ %s (TPR = %s)"%(alias, roundToSigFig(av_maxSI, NSIGFIGS), roundToSigFig(std_maxSI, NSIGFIGS), roundToSigFig(corr_tpr, NSIGFIGS)))


# ##### `calcMaxAvSI_helper()`

def calcMaxAvSI_helper(av_si, std_si, minTPR=0.05):

  #-- Set base TPR --#
  base_tpr = np.linspace(0, 1, 101)

  #-- Trim SI curve components to account for minTPR --#
  assert (minTPR >=0 and minTPR <=1)  # minTPR must be in range [0,1]
  minMask = minTPR <= base_tpr        # Sets TPR values less than minTPR to False

  av_si_trimmed   = av_si[minMask]
  std_si_trimmed  = std_si[minMask]
  tpr_trimmed     = base_tpr[minMask]

  #-- Get max in trimmed TPR range --#
  aMaxSI = np.argmax(av_si_trimmed)

  return av_si_trimmed[aMaxSI], std_si_trimmed[aMaxSI], tpr_trimmed[aMaxSI]

# ##### `calcMaxAvF1()`

def calcMaxAvF1(scoreDict, sigAliasList=['sig_A', 'sig_h0', 'sig_hch', 'sig_LQ'], minTPR=0.05, NSIGFIGS=4, SIMPLE=False):

  print("Max F1 (TPR):")
  # Loop over signal types
  for alias in sigAliasList:

    #-- Get F1 curve components --#
    av_f1  = scoreDict['avStdQuantities'][alias]['F1']['mean']
    std_f1 = scoreDict['avStdQuantities'][alias]['F1']['std']

    #-- Get max av F1, std, and corresponding TPR in range [minTPR, 1] --#
    av_maxF1, std_maxF1, corr_tpr = calcMaxAvSI_helper(av_f1, std_f1, minTPR=minTPR)

    # Report max in trimmed TPR range
    if SIMPLE:
      print("    %s $\pm$ %s (%s)"%(roundToSigFig(av_maxF1, NSIGFIGS), roundToSigFig(std_maxF1, NSIGFIGS), roundToSigFig(corr_tpr, NSIGFIGS)))
    else:
      print("    %s is %s $\pm$ %s (TPR = %s)"%(alias, roundToSigFig(av_maxF1, NSIGFIGS), roundToSigFig(std_maxF1, NSIGFIGS), roundToSigFig(corr_tpr, NSIGFIGS)))


# ##### `calcMaxAvF1_helper()`

def calcMaxAvF1_helper(av_f1, std_f1, minTPR=0.05):

  #-- Set base TPR --#
  base_tpr = np.linspace(0, 1, 101)

  #-- Trim SI curve components to account for minTPR --#
  assert (minTPR >=0 and minTPR <=1)  # minTPR must be in range [0,1]
  minMask = minTPR <= base_tpr        # Sets TPR values less than minTPR to False

  av_f1_trimmed   = av_f1[minMask]
  std_f1_trimmed  = std_f1[minMask]
  tpr_trimmed     = base_tpr[minMask]

  #-- Get max in trimmed TPR range --#
  aMaxF1 = np.argmax(av_f1_trimmed)

  return av_f1_trimmed[aMaxF1], std_f1_trimmed[aMaxF1], tpr_trimmed[aMaxF1]

# # Background data augmentation functions

# Modified from the [original code](https://github.com/bmdillon/AnomalyCLR/blob/main/EventLevelAnomalyAugmentations.py).
# Implemented bug fixes (noted with `#!` comments) and put in comment descriptions of what is happening.

# ### Validation functions to ensure selection requirements are satisfied

# ##### `check_data()`

def check_data(data):
  """
  data: shape (nEvents, 19, 4)
  """

  #-- Check phi --#
  # All phi angles should be in the range [-pi, pi)
  phiCheck = np.all(data[:,:,2] < np.pi) and np.all(data[:,:,2] >= -np.pi)
  print("             phiCheck pass? ",phiCheck)

  #-- Check eta --#
  # Electrons must have |eta|<3.
  # Muons must have     |eta|<2.1
  # Jets must have      |eta|<4.
  eta_eleCheck = np.all(np.abs(data[:,1:5,1]) < 3.)
  eta_muCheck = np.all(np.abs(data[:,5:9,1]) < 2.1)
  eta_jetCheck = np.all(np.abs(data[:,9:,1]) < 4.)
  print("         eta_eleCheck pass? ",eta_eleCheck)
  print("          eta_muCheck pass? ",eta_muCheck)
  print("         eta_jetCheck pass? ",eta_jetCheck)

  #-- Lepton pT check--#
  # All leptons must have pT >= 3.
  # The leading lepton must have pT >= 23.

  mask = data[:,1:9,0] != 0. # Get rid of zero-padded objects
  pT_eleMuCheck         = np.all(data[:,1:9,0][mask] > 3. )
  pT_leadingLeptonCheck = np.all((data[:,1:9,0][:,0] > 23.) | (data[:,1:9,0][:,4] > 23.))
  print("        pT_eleMuCheck pass? ",pT_eleMuCheck)
  print("pT_leadingLeptonCheck pass? ",pT_leadingLeptonCheck)

  #-- Jet pT check--#
  # All jets must have pT >= 15.

  mask = data[:,9:,0] != 0. # Get rid of zero-padded objects
  pT_jetCheck = np.all(data[:,9:,0][mask] > 3. )
  print("          pT_jetCheck pass? ",pT_jetCheck)

  return [phiCheck, eta_eleCheck, eta_muCheck, eta_jetCheck, pT_eleMuCheck, pT_leadingLeptonCheck, pT_jetCheck]

# ### Reformatting data to pass to code and back again

# The code assumes a different structure for the events.  
# 
# By default the events are in an `[nEvents, 19, 4]` array, where the last 4 columns are pT, eta, phi, identity. This means `[:,0,:]` is the information of MET from each event. Similarly `[:,1:5,:]` are electrons, `[:,5:9,:]` are muons, `[:,9:,:]` are jets.
# 
# In the anomaly augmentation code, the last two axes are swapped and a one-hot encoding is added to indicate whether a particle of a certain type is there. So the shape is `[nEvents, 7, 19]` where `[nEvents, 0:4, 19]` are pT, eta, phi, identity and
# ```
# [nEvents, 4, i] is non-zero if the ith particle is an electron with a non-zero entry
# [nEvents, 5, i] is non-zero if the ith particle is a muon with a non-zero entry
# [nEvents, 6, i] is non-zero if the ith particle is a jet with a non-zero entry
# ```
# 
# 
# 

# #### Pass to the code

# Above we reformatted a single event, but now we need to reformat a block of events. We'll start with 2 for easy visualization, but the code itself should generalize

# ##### `add_one_hot_columns()`

def add_one_hot_columns(batch):
  """
  batch:  shape=(nEvents, 19, 4),
          The last 4 columns are: pT, eta, phi, ID.
          Where ID = 1. for MET, 2. for electron, 3. for muon, and 4. for jet

  returns:  shape=(nEvents, 19, 7)
  """
  oneHotCols = np.zeros(shape=(batch.shape[0],batch.shape[1],3))
  dummyOnes  = np.ones(shape=(batch.shape[0],batch.shape[1],3))

  # Select locations of existing electrons, muons, jets
  mask_els  = (batch[:, :, 3] == 2.)
  mask_mus  = (batch[:, :, 3] == 3.)
  mask_jets = (batch[:, :, 3] == 4.)

  # Switch the corresponding one-hot column entry from 0 to 1
  oneHotCols[mask_els, 0]  = dummyOnes[mask_els, 0]
  oneHotCols[mask_mus, 1]  = dummyOnes[mask_mus, 1]
  oneHotCols[mask_jets, 2] = dummyOnes[mask_jets, 2]

  # Concatenate the one-hot columns with the original data
  return np.concatenate((batch, oneHotCols), axis=-1)

def format_for_anomalyAugmentations(bkgData):
  """
  bkgData:  shape=(nEvents, 19, 4)

  returns:  shape=(nEvents, 7, 19)
  """
  batch = bkgData.copy() # so that original isn't modified

  # Add one hot columns for each particle type (i.e. electrons, muons, jets)
  new_batch = add_one_hot_columns(batch)

  # Swap last two axes of the data
  newBkgData = np.swapaxes(new_batch, 1, 2)

  return newBkgData

# ##### Test that the above code behaves as expected

# # TEST
# bkg = dataDict['bkg']['Particles'][11:13,:,:].reshape(-1,19,4)
# print("Original event")
# print(bkg)
# print("(nEvents, 19, 4) = ",bkg.shape)
# print("")

# newBkgData = format_for_anomalyAugmentations(bkg)
# print("Reformatted event")
# print("(nEvents, 7, 19) = ", newBkgData.shape)
# print("")

# print("There is no one-hot encoding for MET")
# print("pT, eta, phi, ID = ")
# print(newBkgData[:, 0:4, 0])
# print("one-hot = ")
# print(newBkgData[:, 4:, 0])
# print("")

# print("The one-hot encoding for existing electrons is the first row")
# print("pT, eta, phi, ID = ")
# print(newBkgData[:, 0:4, 1:5])
# print("one-hot = ")
# print(newBkgData[:, 4:, 1:5])
# print("")

# print("The one-hot encoding for existing muons is the second row")
# print("pT, eta, phi, ID = ")
# print(newBkgData[:, 0:4, 5:9])
# print("one-hot = ")
# print(newBkgData[:, 4:, 5:9])
# print("")

# print("The one-hot encoding for existing jets is the third row")
# print("pT, eta, phi, ID = ")
# print(newBkgData[:, 0:4, 9:])
# print("one-hot = ")
# print(newBkgData[:, 4:, 9:])
# print("")

# #### Get from the code

# We need to undo the formatting done previously, but also account for the fact that the augmentation code causes particles within categories to be ordered a bit weirdly (i.e. some entries and lots of zeros in between).

# augBkg = neg_augs( newBkgData, scaler_pt=1.0, scale_angle=False, etaphi_smear_strength=1.0, addobj=True, addobj_wcpm=True, shpt=True, shmet=True, shporm=True, seed=0)

# print(augBkg.shape)
# print(augBkg[:,0:4,1:5])
# print(augBkg[:,0:4,1:5].shape)
# print(augBkg[:,0:4,5:9].shape)
# print(augBkg[:,0:4,9:].shape)

# We essentially want to
# 
# 1.   Make sure all objects are sorted (within their type) in ascending pT order
# 2.   Remove one-hot encoding
# 3.   Reshape to `(nEvents, 4, 19)`
# 
# 

# ##### `format_for_analysis()`

def format_for_analysis(augBkgData):
  """
  augBkgData:  shape = (nEvents, 7, 19)
  bkgData:     shape = (nEvents, 19, 4)
  """
  # Copy so original isn't modified
  augBkg = augBkgData.copy()

  # Make sure each category (electrons, muons, jets) are sorted high pT to low pT
  indEle = np.flip(np.argsort(augBkg[:,0,1:5], axis=1), axis=1)
  indMu  = np.flip(np.argsort(augBkg[:,0,5:9], axis=1), axis=1)
  indJet = np.flip(np.argsort(augBkg[:,0,9:], axis=1), axis=1)

  augBkg[:,:,1:5] = np.take_along_axis(augBkg[:,:,1:5], indEle[:, np.newaxis, :], axis=-1)
  augBkg[:,:,5:9] = np.take_along_axis(augBkg[:,:,5:9], indMu[:, np.newaxis, :], axis=-1)
  augBkg[:,:,9:]  = np.take_along_axis(augBkg[:,:,9:],  indJet[:, np.newaxis, :], axis=-1)

  # Return augBkg with last two axes swapped and one-hot encodings removed
  return np.swapaxes(augBkg[:,0:4,:], 1, 2)

# newBkgData = format_for_analysis(augBkg)

# print(augBkg.shape)
# print(newBkgData.shape)
# print("")
# print("Electrons")
# print(augBkg[:,0:4,1:5])
# print(newBkgData[:,1:5,:])
# print("")
# print("Muons")
# print(augBkg[:,0:4,5:9])
# print(newBkgData[:,5:9,:])
# print("")
# print("Jets")
# print(augBkg[:,0:4,9:])
# print(newBkgData[:,9:,:])

# ### Augmentation functions

# ##### `augmentAndSaveData()`

def augmentAndSaveData(dataDict, saveFilePath='Data/anomalyAugmented_background_for_training.h5', batchSize=5000, nEvents=None, DEBUG=False):

  #-- Preliminaries --#
  if DEBUG:
    nEvents   = batchSize
    SEED      = 0
  elif nEvents is not None:
    SEED      = None
  else:
    SEED      = None
    nEvents   = dataDict['bkg']['Particles'].shape[0]
  print("Processing %d events"%nEvents)

  if (nEvents % batchSize) == 0:
    nBatches = (nEvents // batchSize)
    LEFTOVERBATCH = False
    print("%d batches with size %d with no events remaining"%(nBatches, batchSize))
  else:
    nBatches = (nEvents // batchSize) + 1
    LEFTOVERBATCH = True
    print("%d batches with size %d with one more batch of size %d"%(nBatches-1, batchSize, nEvents % batchSize))

  #-- Create/open file to store data --#
  print("Saving data to ", saveFilePath)
  f = h5py.File(saveFilePath, 'a')
  f.create_dataset('augBkg', (nEvents,19,4))

  #-- Loop over events --#
  #arr = np.arange(nEvents*19*4).reshape(nEvents,19,4)
  end=0
  for i in tqdm(range(nBatches)):

    # Set batch range
    start = end
    if i==nBatches-1 and LEFTOVERBATCH:
      end   = start+(nEvents % batchSize)
    else:
      end  = start+batchSize

    #-- Augment data --#
    # Transform into expected batch structure
    # Augment
    # Transform back into regular structure
    bkgData = dataDict['bkg']['Particles'][start:end,:,:]
    newBkgData = format_for_anomalyAugmentations(bkgData)
    augBkgData = neg_augs( newBkgData, scaler_pt=1.0, scale_angle=False, etaphi_smear_strength=1.0, addobj=True, addobj_wcpm=True, shpt=True, shmet=True, shporm=True, seed=SEED)
    newAugBkgData = format_for_analysis(augBkgData)

    #-- Save to file--#
    f['augBkg'][start:end,:,:] = newAugBkgData

  # Close file
  f.close()

  if DEBUG:
    return newAugBkgData

# ##### `neg_augs()`

def neg_augs( batch, scaler_pt, scale_angle, etaphi_smear_strength, addobj=True, addobj_wcpm=True, shpt=True, shmet=True, shporm=False, seed=None, DEBUG=False):

  # Set seed for reproducibility
  if seed != None:
    np.random.seed(seed)

  # Copy so that original isn't modified
  batch_aug = batch.copy()

  # Minimum pT requirements
  minLeptonBase, minLeadingpT, minJetBase  = 3., 23., 15. # GeV

  # Split batch_aug into (2) is okay and (2) is not okay groups
  #   For some events, augmentation (2) will result in the transformed event
  #   being the same as the original (which is not what we want)
  #   To avoid this, we want to make sure that (2) is only applied to events
  #   which will transform under (2)
  aug2isOkay = aug_2_conditions(batch_aug, minJetBase=minJetBase, minLeptonBase=minLeptonBase, minLeadingpT=minLeadingpT) # mask; shape=(nEvents,)


  #-- Transform (2) is NOT okay batch --#

  batch_aug2isNotOkay = batch_aug[~aug2isOkay,:,:]

  # Check that either (1) or (3) is specified (i.e. if only (2) is specified don't transform)
  if addobj or shporm:
    # Decide which and how many augmentations to do
    #   (1)      addobj=True <-> "ao"
    #   (3)      shporm=True <-> "spm"
    n_augs = 0
    aug_list = []
    if addobj: n_augs+=1; aug_list.append("ao")
    if shporm: n_augs+=1; aug_list.append("spm")

    # Randomly generate an integer (0,...,n_augs-1)for each event
    #    Each event only recieves one augmentation type
    rands = np.random.randint( low=0, high=n_augs, size=batch_aug2isNotOkay.shape[0] )

    # Select events for each augmentation type and augment those events with the corresponding function
    rand_opts = range( n_augs )
    for j in range( n_augs ):
        aug = aug_list[j]
        n = rand_opts[j]
        if aug=="ao":       batch_aug2isNotOkay[ np.where(rands==n) ] = add_objects( batch_aug2isNotOkay[ np.where(rands==n) ], minJetBase=minJetBase, minLeptonBase=minLeptonBase, DEBUG=DEBUG )
        if aug=="spm":      batch_aug2isNotOkay[ np.where(rands==n) ] = shift_met_or_pt( batch_aug2isNotOkay[ np.where(rands==n) ], DEBUG=DEBUG )


  #-- Transform (2) is okay batch --#

  batch_aug2isOkay    = batch_aug[aug2isOkay,:,:]

  # Decide which and how many augmentations to do
  #   (1)      addobj=True <-> "ao"
  #   (2) addobj_wcpm=True <-> "aowcpm"
  #   (3)      shporm=True <-> "spm"
  n_augs = 0
  aug_list = []
  if addobj: n_augs+=1; aug_list.append("ao")
  if addobj_wcpm: n_augs+=1; aug_list.append("aowcpm")
  if shporm: n_augs+=1; aug_list.append("spm")

  # Randomly generate an integer (0,...,n_augs-1)for each event
  #    Each event only recieves one augmentation type
  rands = np.random.randint( low=0, high=n_augs, size=batch_aug2isOkay.shape[0] )

  # Select events for each augmentation type and augment those events with the corresponding function
  rand_opts = range( n_augs )
  for j in range( n_augs ):
      aug = aug_list[j]
      n = rand_opts[j]
      if aug=="ao":       batch_aug2isOkay[ np.where(rands==n) ] = add_objects( batch_aug2isOkay[ np.where(rands==n) ], minJetBase=minJetBase, minLeptonBase=minLeptonBase, DEBUG=DEBUG )
      if aug=="aowcpm":   batch_aug2isOkay[ np.where(rands==n) ] = add_objects_constptmet( batch_aug2isOkay[ np.where(rands==n) ], scaler_pt, scale_angle, etaphi_smear_strength,
                                                                                          minJetBase=minJetBase, minLeptonBase=minLeptonBase, minLeadingpT=minLeadingpT, DEBUG=DEBUG )
      if aug=="spm":      batch_aug2isOkay[ np.where(rands==n) ] = shift_met_or_pt( batch_aug2isOkay[ np.where(rands==n) ], DEBUG=DEBUG )

  #-- Reform events into the original shape --# #! Done automatically? Need to copy?
  batch_aug[~aug2isOkay,:,:] = batch_aug2isNotOkay
  batch_aug[aug2isOkay,:,:]  = batch_aug2isOkay

  return batch_aug

# ##### augmentation (1): `add_objects()`

# (1)
def add_objects(batch, minJetBase=15., minLeptonBase=3., DEBUG=False):
  # Make repeatable
  if DEBUG: np.random.seed(0)

  # Copy so that original isn't modified
  batch_filled = batch.copy()
  if DEBUG:
    print("Original batch_filled")
    print(batch_filled)
    print("")

  # ! These aren't used... hardcoded instead below
  n_els = 4
  n_mus = 4
  n_jets = 10

  # Calculate number of nonzero electrons, muons, and jets for each event
  # Shape = (nEvents, 1)
  n_nonzero_els  = np.count_nonzero(batch_filled[:, 4, 1:5], axis=1)
  n_nonzero_mus  = np.count_nonzero(batch_filled[:, 5, 5:9], axis=1)
  n_nonzero_jets = np.count_nonzero(batch_filled[:, 6, 9:], axis=1)

  # Randomly generate how many new objects of each type should be added (without exceeding maximum)
  # Shape = (nEvents, 1)
  n_new_els  = np.random.randint( 0, high=4-n_nonzero_els+1 )
  n_new_mus  = np.random.randint( 0, high=4-n_nonzero_mus+1 )
  n_new_jets = np.random.randint( 0, high=10-n_nonzero_jets+1 )
  if DEBUG:
    for (s,x) in zip(["n_new_els","n_new_mus","n_new_jets"],[n_new_els, n_new_mus, n_new_jets]):
      print("%s = "%s,x)
    print("")

  # Get maximum pT of all objects for each event
  # Shape = (nEvents, 1)
  maxpts = np.max( batch[:,0,:], axis=-1 )
  if DEBUG:
    print("maxpts")
    print(maxpts)
    print("")

  # Loop over events
  for n in range( batch_filled.shape[0] ):

    # Generate and add new electrons satisfying: pT > 3, -pi <= phi pi, -3 < eta < 3
    # New pT is random fraction of max pT in event
    el_pts = np.expand_dims( minLeptonBase + (maxpts[n]-minLeptonBase) * np.random.rand( n_new_els[n] ), axis=1 )
    el_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_els[n]) - 0.5 ), axis=1 )
    el_etas = np.expand_dims( 2*3 * ( np.random.rand(n_new_els[n]) - 0.5 ), axis=1 )
    el_one_hot = np.concatenate( [np.zeros(shape=(n_new_els[n],1)), np.ones(shape=(n_new_els[n],1)), np.zeros(shape=(n_new_els[n],1)), np.zeros(shape=(n_new_els[n],1))], axis=1 )
    #els = np.concatenate( [el_pts, el_phis, el_etas, el_one_hot], axis=1 ) #! ERROR here: should be pT, eta, phi ....
    els = np.concatenate( [el_pts, el_etas, el_phis, el_one_hot], axis=1 )
    el_start = 1 + n_nonzero_els[n]
    el_end   = 1 + n_nonzero_els[n] + n_new_els[n]
    batch_filled[n,:,el_start:el_end] = np.transpose( els )

    # Generate and add new muons satisfying: pT > 3, -pi <= phi pi, -2.1 < eta < 2.1
    # New pT is random fraction of max pT in event
    mu_pts = np.expand_dims( minLeptonBase + (maxpts[n]-minLeptonBase) * np.random.rand( n_new_mus[n] ), axis=1 )
    mu_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_mus[n]) - 0.5 ), axis=1 )
    mu_etas = np.expand_dims( 2*2.1 * ( np.random.rand(n_new_mus[n]) - 0.5 ), axis=1 )
    mu_one_hot = np.concatenate( [np.zeros(shape=(n_new_mus[n],1)), np.zeros(shape=(n_new_mus[n],1)), np.ones(shape=(n_new_mus[n],1)), np.zeros(shape=(n_new_mus[n],1))], axis=1 )
    #mus = np.concatenate( [mu_pts, mu_phis, mu_etas, mu_one_hot], axis=1 ) #! ERROR: should be pT, eta, phi ....
    mus = np.concatenate( [mu_pts, mu_etas, mu_phis, mu_one_hot], axis=1 )
    mu_start = 5 + n_nonzero_mus[n]
    mu_end = 5 + n_nonzero_mus[n] + n_new_mus[n]
    batch_filled[n,:,mu_start:mu_end] = np.transpose( mus )

    # Generate and add new jets satisfying: pT > 15, -pi <= phi pi, -4 < eta < 4
    # New pT is random fraction of max pT in event
    jet_pts = np.expand_dims( minJetBase + (maxpts[n]-minJetBase) * np.random.rand( n_new_jets[n] ), axis=1 )
    jet_phis = np.expand_dims( 2*np.pi * ( np.random.rand(n_new_jets[n]) - 0.5 ), axis=1 )
    jet_etas = np.expand_dims( 2*4 * ( np.random.rand(n_new_jets[n]) - 0.5 ), axis=1 )
    jet_one_hot = np.concatenate( [np.zeros(shape=(n_new_jets[n],1)), np.zeros(shape=(n_new_jets[n],1)), np.zeros(shape=(n_new_jets[n],1)), np.ones(shape=(n_new_jets[n],1))], axis=1 )
    #jets = np.concatenate( [jet_pts, jet_phis, jet_etas, jet_one_hot], axis=1 ) #! ERROR: should be pT, eta, phi ....
    jets = np.concatenate( [jet_pts, jet_etas, jet_phis, jet_one_hot], axis=1 )
    jet_start = 9 + n_nonzero_jets[n]
    jet_end   = 9 + n_nonzero_jets[n] + n_new_jets[n]
    batch_filled[n,:,jet_start:jet_end] = np.transpose( jets )

    # Recalculate MET
    # NOTE: There is a sign error in this code
    old_met_pt = batch_filled[n,0,0]
    old_met_phi = batch_filled[n,2,0]
    old_met = np.array( [ old_met_pt * np.sin(old_met_phi), old_met_pt * np.cos(old_met_phi) ] )
    if DEBUG:
      print("old_met")
      print(old_met)
      print("")

    new_obj = np.concatenate( [ els[:,0:3], mus[:,0:3], jets[:,0:3] ], axis=0 )
    if DEBUG:
      print("Total (px,py) of new objects")
      print(np.array( [ new_obj[:,0] * np.sin(new_obj[:,2]), new_obj[:,0] * np.cos(new_obj[:,2]) ] ).sum(axis=-1))
      print("")

    new_met = old_met - np.array( [ new_obj[:,0] * np.sin(new_obj[:,2]), new_obj[:,0] * np.cos(new_obj[:,2]) ] ).sum(axis=-1) #! Sign ERROR, should be old_met - ... not old_met + ...
    if DEBUG:
      print("MET change (note the sign difference)")
      print(new_met - old_met)
      print("")

    new_met_pt = np.sqrt( new_met[0]**2 + new_met[1]**2 )
    #new_met_phi = np.arcsin( new_met[0]/new_met_pt ) #! This doesn't adequately reconstruct the full range -pi to pi
    if new_met[1]<0. and new_met[0]>0.:
      new_met_phi =  np.pi - np.arcsin( new_met[0]/new_met_pt )
    elif new_met[1]<0. and new_met[0]<0.:
      new_met_phi = -np.pi - np.arcsin( new_met[0]/new_met_pt )
    else:
      new_met_phi = np.arcsin( new_met[0]/new_met_pt )

    batch_filled[n,0,0] = new_met_pt
    batch_filled[n,2,0] = new_met_phi
    if DEBUG:
      print("Modified batch_filled")
      print(batch_filled)
      print("")
  return batch_filled

# ##### augmentation (2): `add_objects_constptmet()` + helper functions: `get_std_rivet()`, `etaphi_smear_events()`, `collinear_fill_e_mu()`, `collinear_fill_jets()`

def get_std_rivet(pTs, scaler_pt, A=0.028, B=25, C=0.1):
    #  standard deviation for the Rivet detector simulation

    # Only select pT > 0 entries for all events
    # (nEvents,1)
    mask = (pTs > 0)
    np_sett_dict = np.seterr(over = 'ignore')

    if scaler_pt != None:
        std_rivet  = A/(1+np.exp( ( (pTs *scaler_pt) -B)/C) )
    else:
        std_rivet  = A/(1+np.exp( ( pTs -B)/C) )
    std_rivet[~mask] = 0
    np.seterr(over = np_sett_dict['over'])
    return std_rivet

def etaphi_smear_events(batch, scaler_pt, scale_angle, strength=1.0, DEBUG=False):
  # Make repeatable
  if DEBUG: np.random.seed(0)

  # Copy so that original isn't modified
  batch_distorted = batch.copy()
  if DEBUG:
    print("Original batch_distorted")
    print(batch_distorted)
    print("")

  # Get standard deviation for pT smear amount
  # The std is determined by a step function so that pT in the range (0, B - window) will all be scaled by A
  # After B, the std is much less or zero
  # => Larger pT will be scaled less or not at all
  std = get_std_rivet( batch_distorted[:,0, 1:], scaler_pt )
  noise_eta = np.random.normal( loc=0.0, scale=strength*std )
  noise_phi = np.random.normal( loc=0.0, scale=strength*std )
  noise     = np.stack( [noise_eta, noise_phi], axis=1 )

  # Smear eta, phi of all particle objects (i.e. not MET)
  # Note that this is effectively doing a resampling of eta, phi from a Gaussian with mean eta, phi and variance strength*std, strength*std
  batch_distorted[:,1:3,1:] += noise

  # scale_angle seems to be whether phi's are assumed to be scaled to a different range in the data, so default should be False
  if scale_angle:
    # Ensure all smeared phis are in the range (-1,1)
    batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]>1, batch_distorted[:, 2, 1:]-2, batch_distorted[:, 2, 1:])
    batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]< -1, batch_distorted[:, 2, 1:]+2, batch_distorted[:, 2, 1:])

    # Check that all smeared electron etas are in the range (-3,3)/4
    crosses_upper_bound_e = batch_distorted[:, 1, 1:5] > (3./4)
    crosses_lower_bound_e = batch_distorted[:, 1, 1:5] < (-3./4.)
    crosses_e = crosses_lower_bound_e | crosses_upper_bound_e # Logical OR
    # Check that all smeared electron etas are in the range (-2.1,2.1)/4
    crosses_upper_bound_mu = batch_distorted[:, 1, 5:9] > (2.1/4.)
    crosses_lower_bound_mu = batch_distorted[:, 1, 5:9] < (-2.1/4.)
    crosses_mu = crosses_lower_bound_mu | crosses_upper_bound_mu
    # Check that all smeared jet etas are in the range (-1,1)
    crosses_upper_bound_jet = batch_distorted[:, 1, 9:] > 1.
    crosses_lower_bound_jet = batch_distorted[:, 1, 9:] < -1.
    crosses_jet = crosses_lower_bound_jet | crosses_upper_bound_jet
    # If objects violate these conditions, set their pT, eta, phi, etc. all to zero
    for i in range( np.shape(batch_distorted)[1] ):
        batch_distorted[:, i, 1:5][crosses_e]  = 0.
        batch_distorted[:, i, 5:9][crosses_mu] = 0.
        batch_distorted[:, i, 9:][crosses_jet] = 0.
  else:
    # Ensure all smeared phis are in the range [-pi,pi]
    #! Really should be [-pi,pi)
    #! Doesn't correctly handle the case if smeared phi is outside the range [-2pi,2pi].  This is very unlikely, but technically possible...
    batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]>np.pi, batch_distorted[:, 2, 1:]-np.pi, batch_distorted[:, 2, 1:])
    batch_distorted[:, 2, 1:] = np.where(batch_distorted[:, 2, 1:]< -np.pi, batch_distorted[:, 2, 1:]+np.pi, batch_distorted[:, 2, 1:])

    # Check that all smeared electron etas are in the range (-3,3)
    crosses_upper_bound_e = batch_distorted[:, 1, 1:5] > 3.
    crosses_lower_bound_e = batch_distorted[:, 1, 1:5] < -3.
    crosses_e = crosses_lower_bound_e | crosses_upper_bound_e
    if DEBUG:
      print("crosses_e")
      print(crosses_e)
      print("")
    # Check that all smeared muon etas are in the range (-2.1,2.1)
    crosses_upper_bound_mu = batch_distorted[:, 1, 5:9] > 2.1
    crosses_lower_bound_mu = batch_distorted[:, 1, 5:9] < -2.1
    crosses_mu = crosses_lower_bound_mu | crosses_upper_bound_mu
    if DEBUG:
      print("crosses_mu")
      print(crosses_mu)
      print("")
    # Check that all smeared jet etas are in the range (-4,4)
    crosses_upper_bound_jet = batch_distorted[:, 1, 9:] > 4.
    crosses_lower_bound_jet = batch_distorted[:, 1, 9:] < -4.
    crosses_jet = crosses_lower_bound_jet | crosses_upper_bound_jet
    if DEBUG:
      print("crosses_jet")
      print(crosses_jet)
      print("")
    # Fix objects that violate these conditions to be at but not over the edge
    #   Note that the former fix was to eliminate the violating particles
    #   but that strategy is problematic if the leading lepton got eliminated
    epsilon = 1e-6 #np.finfo(np.float64).eps
    batch_distorted[:, 1, 1:5][crosses_lower_bound_e]  = -3.  + epsilon
    batch_distorted[:, 1, 1:5][crosses_upper_bound_e]  =  3.  - epsilon
    batch_distorted[:, 1, 5:9][crosses_lower_bound_mu] = -2.1 + epsilon
    batch_distorted[:, 1, 5:9][crosses_upper_bound_mu] =  2.1 - epsilon
    batch_distorted[:, 1, 9:][crosses_lower_bound_jet] = -4.  + epsilon
    batch_distorted[:, 1, 9:][crosses_upper_bound_jet] =  4.  - epsilon

  return batch_distorted

def collinear_fill_e_mu(batch, minBase=3., minLeading=23., DEBUG=False):

  # Set minimum pT allowed (minBase + epsilon)
  # https://stackoverflow.com/questions/48382823/generating-random-numbers-in-numpy-with-strict-lower-bounds
  epsilon = np.finfo(np.float64).eps
  min_pT  = minBase + epsilon

  # Make repeatable
  if DEBUG: np.random.seed(0)

  batch_filled = batch.copy()

  # ELECTRONS
  n_constit = 4
  n_nonzero = np.count_nonzero(batch_filled[:, 4, 1:5], axis=1)
  n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
  idx_flip = np.where(n_nonzero != n_split)
  mask_split = batch_filled[:, 4, 1:5] != 0
  mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
  mask_split[idx_flip]  = np.invert(mask_split[idx_flip])

  #! Start bug fix to maintain pT selection criteria after splitting

  # Get theoretical minima based on selection criteria
  #    pT for leptons must be > min_pT=3GeV and for the leading lepton pT must be >23GeV
  minA_pT = min_pT*np.ones(shape=mask_split.shape)              # Shape=(nEvents,4)
  minA_pT[:,0][(batch_filled[:,0,1] >= batch_filled[:,0,5])] = minLeading+epsilon    # batch[:,0,1] >= batch[:,0,5] selects leading electrons
  if DEBUG:
    print("minA_pT for Electrons")
    print(minA_pT)
    print("")
  mask_pT_largeEnoughToSplit = batch_filled[:, 0, 1:5] > minA_pT + min_pT
  new_mask_split = mask_split & mask_pT_largeEnoughToSplit
  extra = batch_filled[:, 0, 1:5] - (minA_pT + min_pT)
  if DEBUG:
    print("extra")
    print(extra)
    print("")
  r_split = np.random.uniform(size=new_mask_split.shape)
  if DEBUG:
    print("r_split")
    print(r_split)
    print("")
  a = minA_pT * new_mask_split +       r_split * new_mask_split*extra
  b = min_pT * new_mask_split + (1.0-r_split) * new_mask_split*extra
  c =                ~new_mask_split*batch_filled[:, 0, 1:5]
  batch_filled[:, 0, 1:5] = a + c + np.flip(b, axis=1)
  batch_filled[:, 1, 1:5] += np.flip(new_mask_split*batch_filled[:, 1, 1:5], axis=1)
  batch_filled[:, 2, 1:5] += np.flip(new_mask_split*batch_filled[:, 2, 1:5], axis=1)
  batch_filled[:, 3, 1:5] += np.flip(new_mask_split*batch_filled[:, 3, 1:5], axis=1) #! Bug fix to also fill identity
  batch_filled[:, 4, 1:5] += np.flip(new_mask_split*batch_filled[:, 4, 1:5], axis=1)

  #! End bug fix to maintain pT selection criteria after splitting

  # MUONS
  n_constit = 4
  n_nonzero = np.count_nonzero(batch_filled[:, 5, 5:9], axis=1)
  n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
  idx_flip = np.where(n_nonzero != n_split)
  mask_split = batch_filled[:, 5, 5:9] != 0
  mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
  mask_split[idx_flip] = np.invert(mask_split[idx_flip])

  #! Start bug fix to maintain pT selection criteria after splitting

  # Get theoretical minima based on selection criteria
  #    pT for leptons must be > min_pT=3GeV and for the leading lepton pT must be >23GeV
  minA_pT = min_pT*np.ones(shape=mask_split.shape)              # Shape=(nEvents,4)
  minA_pT[:,0][(batch_filled[:,0,1] < batch_filled[:,0,5])] = minLeading+epsilon    # batch[:,0,1] < batch[:,0,5] selects leading muons
  if DEBUG:
    print("minA_pT for Muons")
    print(minA_pT)
    print("")
  mask_pT_largeEnoughToSplit = batch_filled[:, 0, 5:9] > minA_pT + min_pT
  new_mask_split = mask_split & mask_pT_largeEnoughToSplit
  extra = batch_filled[:, 0, 5:9] - (minA_pT + min_pT)
  r_split = np.random.uniform(size=new_mask_split.shape)
  a = minA_pT * new_mask_split +       r_split * new_mask_split*extra
  b = min_pT * new_mask_split + (1.0-r_split) * new_mask_split*extra
  c =                ~new_mask_split*batch_filled[:, 0, 5:9]
  batch_filled[:, 0, 5:9] = a + c + np.flip(b, axis=1)
  batch_filled[:, 1, 5:9] += np.flip(new_mask_split*batch_filled[:, 1, 5:9], axis=1)
  batch_filled[:, 2, 5:9] += np.flip(new_mask_split*batch_filled[:, 2, 5:9], axis=1)
  batch_filled[:, 3, 5:9] += np.flip(new_mask_split*batch_filled[:, 3, 5:9], axis=1) #! Bug fix to also fill identity
  batch_filled[:, 5, 5:9] += np.flip(new_mask_split*batch_filled[:, 5, 5:9], axis=1)

  #! End bug fix to maintain pT selection criteria after splitting

  return batch_filled

def collinear_fill_jets(batch, minBase=15., DEBUG=False):

  # Set minimum pT allowed (minBase + epsilon)
  # https://stackoverflow.com/questions/48382823/generating-random-numbers-in-numpy-with-strict-lower-bounds
  epsilon = np.finfo(np.float64).eps
  min_pT  = minBase + epsilon

  # Make repeatable
  if DEBUG: np.random.seed(0)

  # Copy so that original isn't modified
  batch_filled = batch.copy()
  if DEBUG:
    print("Original batch_filled")
    print(batch_filled)
    print("")

  # Get number of zero and non-zero objects
  # (nEvents,1)
  n_constit = 10
  n_nonzero = np.count_nonzero(batch_filled[:, 6, 9:], axis=1)
  if DEBUG:
    print("n_nonzero, n_constit - n_nonzero = ",n_nonzero, n_constit-n_nonzero)
    print("")

  # Get the number of objects that will be split.
  # Note we can't split more objects than are there so n_split <= n_nonzero
  # And we also can't split more than there are spaces to add n_split <= n_constit-n_nonzero
  # So we choose n_split to be the lesser of the two
  # (nEvents,1)
  n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
  if DEBUG:
    print("n_split (min of the two) = ",n_split)
    print("")

  # Empty array if n_nonzero == n_split for all events
  # Otherwise returns the event index where there are more non-zero jets than zero jets
  # (nEvents,1)
  idx_flip = np.where(n_nonzero != n_split)
  if DEBUG:
    print("idx_flip", idx_flip)
    print("")

  # Mask jet entries that are not zero for all events
  # (nEvents, n_constit=10)
  mask_split = batch_filled[:, 6, 9:] != 0
  if DEBUG:
    print("mask_split.shape", mask_split.shape)
    for i in range(batch.shape[0]):
      print("mask_split[%d,:]"%i)
      print(mask_split[i,:])
    print("")

  # Select the events where there are more non-zero jets than zero jets (idx_flip)
  if DEBUG:
    print("mask_split[idx_flip]")
    print(mask_split[idx_flip])
    print("np.flip(...)")   # Reverse the order of elements in an array along the given axis
  mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
  if DEBUG:
    print(mask_split[idx_flip])
    print("np.invert(...)") # Change True <-> False
  mask_split[idx_flip] = np.invert(mask_split[idx_flip])
  if DEBUG:
    print(mask_split[idx_flip])
    print("")

  #! Bug fix
  # Ensure that entries to be split also have pT > 2*min, set the mask of any that fail to false
  # pT = batch_filled[:, 0, 9:]
  # mask_split.shape = (nEvents, n_constit=10)
  mask_pT_largeEnoughToSplit = batch_filled[:, 0, 9:] > 2*min_pT
  new_mask_split = mask_split & mask_pT_largeEnoughToSplit
  if DEBUG:
    print("mask_split")
    print(mask_split)
    print("mask_pT_largeEnoughToSplit")
    print(mask_pT_largeEnoughToSplit)
    print("new_mask_split")
    print(new_mask_split)
    print("")

  # Get random split fractions
  r_split = np.random.uniform(size=new_mask_split.shape)
  if DEBUG:
    print("r_split.shape",r_split.shape) # (nEvents, n_constit=10)
    print("r_split")
    print(r_split)
    print("")

  # Get the extra amount to be split
  #   Note that for some of these, extra will be negative
  #   but those entries will be set to False and thus not contribute by new_mask_split
  extra = batch_filled[:, 0, 9:] - 2*min_pT

  # Split the pT for valid objects (i.e. with at least twice the minimum)
  # For each event split the first n_split pT=min+extra into min + r*extra and min + (1-r)*extra
  a = min_pT * new_mask_split +       r_split * new_mask_split*extra
  b = min_pT * new_mask_split + (1.0-r_split) * new_mask_split*extra
  if DEBUG:
    print("a+b = pT for the first n_split entries")
    print("a   = ",a)
    print("b   = ",b)
    print("c captures the remaining pT entries")
  c =                ~new_mask_split*batch_filled[:, 0, 9:]
  if DEBUG:
    print("c   = ",c)
    print("a+b+c = ",a+b+c)
    print("pT    = ",batch_filled[:, 0, 9:])
    print("")

  # Now take one part (i.e. b) and assign it to the last n_split zero entries and recombine
  if DEBUG:
    print("np.flip(b, axis=1)")
    print(np.flip(b, axis=1))
    print("")
  batch_filled[:, 0, 9:] = a + c + np.flip(b, axis=1)

  # Set all other quantities (i.e. eta, phi) to be the same
  batch_filled[:, 1, 9:] += np.flip(new_mask_split*batch_filled[:, 1, 9:], axis=1)
  batch_filled[:, 2, 9:] += np.flip(new_mask_split*batch_filled[:, 2, 9:], axis=1)
  batch_filled[:, 3, 9:] += np.flip(new_mask_split*batch_filled[:, 3, 9:], axis=1) #! Bug fix to also fill identity
  batch_filled[:, 6, 9:] += np.flip(new_mask_split*batch_filled[:, 6, 9:], axis=1)
  if DEBUG:
    print("New batch_filled")
    print(batch_filled)
    print("")

  return batch_filled

def add_objects_constptmet( batch, scaler_pt, scale_angle, etaphi_smear_strength, minJetBase=15., minLeptonBase=3., minLeadingpT=23., DEBUG=False):
  batch_filled = batch.copy()
  batch_filled = collinear_fill_jets( batch_filled, minBase=minJetBase, DEBUG=DEBUG ) # Minimum pT for jets is 15 GeV
  batch_filled = collinear_fill_e_mu( batch_filled, minBase=minLeptonBase, minLeading=minLeadingpT, DEBUG=DEBUG ) # Minimum pT for leptons is 3 GeV, min for leading lepton is 23 GeV
  batch_filled = etaphi_smear_events( batch_filled, scaler_pt, scale_angle, strength=etaphi_smear_strength, DEBUG=DEBUG )
  return batch_filled

# ##### augmentation (3): `shift_met_or_pt()`

# (3)
def shift_met_or_pt( batch, DEBUG=False):
  # Make repeatable
  if DEBUG: np.random.seed(0)

  # Copy so that original isn't modified
  batch_shifted = batch.copy()
  if DEBUG:
    print("Original batch_shifted")
    print(batch_shifted)
    print("")

  # Random numbers for each event to decide whether to shift the MET (0), pT (1), or both (2)
  # Shape = (nEvents, 1)
  rands = np.random.randint( low=0, high=3, size=batch_shifted.shape[0] )
  if DEBUG:
    print("If rands==0, shift MET; ==1, shift pT; ==2, shift both")
    print("rands = ",rands)
    print("")

  # Randomly generate a constant multiplicative shift factor for each event for pT and MET
  shifts     =  1.0 + np.random.rand( batch_shifted.shape[0] ) * 4.0  # Uniform range [  1, 5)
  shifts_met =  0.5 + np.random.rand( batch_shifted.shape[0] ) * 4.5  # Uniform range [0.5, 5)
  if DEBUG:
    print("    shifts = ", shifts)
    print("shifts_met = ",shifts_met)
    print("")

  # Shift MET, pT, or both for each event depending on rands
  batch_shifted[np.where(rands==0),0,0 ] *= shifts_met[np.where(rands==0)]
  batch_shifted[np.where(rands==1),0,1:] *= np.expand_dims( shifts[np.where(rands==1)], axis=-1 )
  batch_shifted[np.where(rands==2),0,: ] *= np.expand_dims( shifts[np.where(rands==2)], axis=-1 )
  if DEBUG:
    print("New batch_shifted")
    print(batch_shifted)
    print("")
  return batch_shifted

# ### Functions to check whether augmentation (2) is okay for that event (i.e. okay => the event will transform under (2) )

# ##### `aug_2_jetConditions()`

def aug_2_jetConditions(batch_aug, minBase=15.):
  """
  Checks whether any jet in an event will transform under augmentation (2).
  If the jets of an event will transform, then the mask is True for that event.
  """
  epsilon = np.finfo(np.float64).eps
  min_pT  = minBase + epsilon

  # Figure out which jets might be split based on filled and available slots
  #   mask_split is True for jets that might be split False otherwise; shape=(nEvents, n_constit)
  n_constit = 10
  n_nonzero = np.count_nonzero(batch_aug[:, 6, 9:], axis=1)
  n_split   = np.minimum(n_nonzero, n_constit-n_nonzero)
  idx_flip  = np.where(n_nonzero != n_split)
  mask_split = batch_aug[:, 6, 9:] != 0
  mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
  mask_split[idx_flip] = np.invert(mask_split[idx_flip])
  mask_pT_largeEnoughToSplit = batch_aug[:, 0, 9:] > 2*min_pT
  new_mask_split = mask_split & mask_pT_largeEnoughToSplit

  # Decide whether any jets in the event transform
  #    passJet shape is (nEvents,)
  passJet = np.any(new_mask_split, axis=1)

  return passJet

## Example
# EVENTNUMBER = 5
# testEvent   = dataDict['bkg']['Particles'][0:EVENTNUMBER,:,:].reshape(EVENTNUMBER,-1,4)
# batch = format_for_anomalyAugmentations(testEvent)
# print(aug_2_jetConditions(batch, minBase=15.)) # array([False, False, False,  True,  True])
# print(batch[:,:,9:13])# 13 is just to make print-out more readable since the rest are zero

# ##### `aug_2_electronConditions()`

def aug_2_electronConditions(batch_aug, minBase=3., minLeading=23.):

  epsilon = np.finfo(np.float64).eps
  min_pT  = minBase + epsilon

  n_constit = 4
  n_nonzero = np.count_nonzero(batch_aug[:, 4, 1:5], axis=1)
  n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
  idx_flip = np.where(n_nonzero != n_split)
  mask_split = batch_aug[:, 4, 1:5] != 0
  mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
  mask_split[idx_flip]  = np.invert(mask_split[idx_flip])


  # Get theoretical minima based on selection criteria
  #    pT for leptons must be > min_pT=3GeV and for the leading lepton pT must be >23GeV
  minA_pT = min_pT*np.ones(shape=mask_split.shape)   # Shape=(nEvents,4)
  minA_pT[:,0][(batch_aug[:,0,1] >= batch_aug[:,0,5])] = minLeading+epsilon    # batch[:,0,1] >= batch[:,0,5] selects leading electrons
  mask_pT_largeEnoughToSplit = batch_aug[:, 0, 1:5] > minA_pT + min_pT
  new_mask_split = mask_split & mask_pT_largeEnoughToSplit

  # Decide whether any electrons in the event transform
  #    passElectron shape is (nEvents,)
  passElectron = np.any(new_mask_split, axis=1)

  return passElectron

## Example
# EVENTNUMBER = 5
# testEvent   = dataDict['bkg']['Particles'][0:EVENTNUMBER,:,:].reshape(EVENTNUMBER,-1,4)
# batch = format_for_anomalyAugmentations(testEvent)
# print(aug_2_electronConditions(batch, minBase=3.,minLeading=23.)) # array([False False  True False False])
# print(batch[:,:,1:5])

# ##### `aug_2_muonsConditions()`

def aug_2_muonsConditions(batch_aug, minBase=3., minLeading=23.):

  epsilon = np.finfo(np.float64).eps
  min_pT  = minBase + epsilon

  n_constit = 4
  n_nonzero = np.count_nonzero(batch_aug[:, 5, 5:9], axis=1)
  n_split = np.minimum(n_nonzero, n_constit-n_nonzero)
  idx_flip = np.where(n_nonzero != n_split)
  mask_split = batch_aug[:, 5, 5:9] != 0
  mask_split [idx_flip] = np.flip(mask_split[idx_flip], axis=1)
  mask_split[idx_flip] = np.invert(mask_split[idx_flip])

  # Get theoretical minima based on selection criteria
  #    pT for leptons must be > min_pT=3GeV and for the leading lepton pT must be >23GeV
  minA_pT = min_pT*np.ones(shape=mask_split.shape)  # Shape=(nEvents,4)
  minA_pT[:,0][(batch_aug[:,0,1] < batch_aug[:,0,5])] = minLeading+epsilon    # batch[:,0,1] < batch[:,0,5] selects leading muons
  mask_pT_largeEnoughToSplit = batch_aug[:, 0, 5:9] > minA_pT + min_pT
  new_mask_split = mask_split & mask_pT_largeEnoughToSplit

  # Decide whether any muons in the event transform
  #    passMuon shape is (nEvents,)
  passMuon = np.any(new_mask_split, axis=1)

  return passMuon

## Example
# EVENTNUMBER = 5
# testEvent   = dataDict['bkg']['Particles'][0:EVENTNUMBER,:,:].reshape(EVENTNUMBER,-1,4)
# batch = format_for_anomalyAugmentations(testEvent)
# print(aug_2_muonsConditions(batch, minBase=3., minLeading=23.)) # array([False False False  True False])
# print(batch[:,:,5:9])

# ##### `aug_2_conditions()`

def aug_2_conditions(batch_aug, minJetBase=15., minLeptonBase=3., minLeadingpT=23.):
  """
  Function to determine which events in batch_aug will fail to transform under augmentation (2).
  Since these events won't transform, augmentation (2) should not be performed.

  Inputs:
    batch_aug:  Batch of events, shape = (nEvents, 7, 19)
  Outputs:
    Mask with shape=(nEvents,1). True indicates that augmentation (2) can be performed. False indicates that it cannot.
  """

  # Will jets transform
  passJet = aug_2_jetConditions(batch_aug, minBase=minJetBase) # shape=(nEvents,)

  # Will electrons transform
  passElectron = aug_2_electronConditions(batch_aug, minBase=minLeptonBase, minLeading=minLeadingpT) # shape=(nEvents,)

  # Will muons transform
  passMuon = aug_2_muonsConditions(batch_aug, minBase=minLeptonBase, minLeading=minLeadingpT) # shape=(nEvents,)

  # If any of the above conditions are True for an event then the overall mask should be True for that event
  return passJet | passElectron | passMuon

## Example
# EVENTNUMBER = 5
# testEvent   = dataDict['bkg']['Particles'][0:EVENTNUMBER,:,:].reshape(EVENTNUMBER,-1,4)
# batch = format_for_anomalyAugmentations(testEvent)
# print(aug_2_conditions(batch)) # array([False False  True  True  True])
# print(batch)
#
## To get events that are able to be transformed by (2)
# aug2condPass = aug_2_conditions(batch_aug)
# print(batch[aug2condPass,:,:])
## To get events that are unable to be transformed by (2)
# print(batch[~aug2condPass,:,:])

# # Plotting functions

# ## Misc. helper functions

# ##### `RGBAtoRGBAtuple()`

def RGBAtoRGBAtuple(color, TYPE='tuple'):
  """
  Quick function to convert human RGBA to python RGBA tuple format. Example:
     Human RGBA  = (120,15,116,1)
     Python RGBA = RGBAtoRGBAtuple((120,15,116,1))

  Inputs:
    color:   Human RGBA tuple
    TYPE:    Flag to indicate whether you want function to return a list or tuple; default is tuple
  Outputs:
    Python RGBA tuple (or list)
  """
  r = color[0]/255
  g = color[1]/255
  b = color[2]/255
  a = color[3]

  if TYPE=='list':
    return [r, g, b, a]
  else:
    return (r, g, b, a)

# ## Data plotting functions

# ##### `plotDataHists() `

def plotDataHists(objectBkg, objectSigList, plotArgDict):
  """
  Plot histograms of an object's pT, eta, phi for Signal and Signal+Background
  for each signal case.

  Inputs:
    objectBkg:      A background event's object's (e.g. MET) pT, eta, phi;
                    shape = (Nb, 3); Nb = number of background events
    objectSigList:  A list of signal event object's (e.g. MET) pT, eta, phi for
                    nCases number of signal cases; List of tuples with shapes
                    (Ns, 3); Ns = number of signal events
    plotArgDict:    Dictionary of plotting arguments (see example below)

  Outputs:
    Nothing returned; Show plot


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
      plotArgDict = {}
      plotArgDict['pltDim']             = (1,3) # i.e. 1 row, 3 columns
      plotArgDict['xAxisLimsList']      = [(0, 1500), (-5, 5), (-np.pi, np.pi)]
      plotArgDict['title']              = r'MET'
      plotArgDict['nBins']              = 50
      plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
      plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (7*pltDim[1], 7*pltDim[0])
  fig = plt.figure(constrained_layout=True, figsize=fig_size)
  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  axes = []
  for i in range(pltDim[1]):
    axes.append(fig.add_subplot(gs[:, i]))

  xLabelList = [r'$p_{\rm T}$', r'$\eta$', r'$\phi$']

  #-- Define color map --#
  # Reference: https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(objectSigList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))

  #-- Combine data for S+B plots for all signal types --#
  objectBkgSigList = []
  for objectSig in objectSigList:
    objectBkgSigList.append(np.concatenate((objectBkg, objectSig), axis=0))


  #-- Loop over axes to make plots --#
  for i in range(pltDim[1]):

    ax = axes[i]

    # Set axis and title information
    xmin, xmax = plotArgDict['xAxisLimsList'][i]
    ax.set_xlim(xmin, xmax)
    ax.set(xlabel=xLabelList[i])
    ax.xaxis.label.set_size(16)

    if i==0:
      ax.set(ylabel=r'Simulated events')
      ax.yaxis.label.set_size(16)

    if i==1:
      ax.set_title(plotArgDict['title'], fontsize=20)

    # Set bin information
    bins = np.linspace(xmin, xmax, plotArgDict['nBins'])

    #-- Plot background case --#
    _, _, _,   = ax.hist(objectBkg[:,i], bins=bins, histtype = 'step', edgecolor='black', linestyle='-', linewidth=2, fill=False, log=True, label='Background')

    #-- Loop over signal cases --#
    for j in range(nCases):
      objectBkgSig = objectBkgSigList[j]
      objectSig    = objectSigList[j]

      # Get data
      SB = objectBkgSig[:,i]
      S  = objectSig[:,i]

      # Make plot
      Slabel = plotArgDict['sigObjectNameList'][j]+' Signal'
      SBlabel = Slabel+' + Background'
      _, _, _,   = ax.hist(SB, bins=bins, histtype = 'step', edgecolor=colorList[j], linestyle='--', linewidth=2, fill=False, log=True, label=SBlabel)

    if i==0:
      ax.legend(loc='upper right')

  #-- Show the plot --#
  plt.show()

# ##### `plotMultiplicityData() `

def plotMultiplicityData(multBkgList, multSigList, plotArgDict):
  """
  Plot histograms of the multiplicity of electrons, muons, and jets for Background and each Signal case.

  Inputs:
    multBkgList:      List of arrays corresponding to electron, muon, and jet multiplicities
                      E.g. multBkgList[0] is an array of shape (Nb,) where  Nb = number of background events
                      multBkgList[0][i] is the electron multiplicity for the ith event
    multSigList:      List of list of arrays corresponding to electron, muon, and jet multiplicities for each signal type
                      E.g. multSigList[0] is the list of electron multiplicities for all signal types
                      multSigList[0][j] is an array of shape (Ns,) where Ns = number of signal events in the jth signal type
                      multSigList[0][j][i] is the electron multiplicity for the ith event of signal type j
    plotArgDict:    Dictionary of plotting arguments (see example below)

  Outputs:
    Nothing returned; Show plot


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
      plotArgDict = {}
      plotArgDict['pltDim']             = (1,3) # i.e. 1 row, 3 columns
      plotArgDict['xAxisLimsList']      = [(-0.5, 5.5), (-0.5, 5.5), (-0.5, 10.5)]
      plotArgDict['title']              = r'Multiplicity'
      plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
      plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (7*pltDim[1], 7*pltDim[0])
  fig = plt.figure(constrained_layout=True, figsize=fig_size)
  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  axes = []
  for i in range(pltDim[1]):
    axes.append(fig.add_subplot(gs[:, i]))

  xLabelList = [r'$e$ multiplicity', r'$\mu$ multiplicity', r'${\rm jet}$ multiplicity']


  #-- Define color map --#
  # Reference: https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(multSigList[0])  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))


  #-- Loop over axes to make plots --#
  # i=0 => Electrons
  # i=1 => Muons
  # i=2 => Jets
  for i in range(pltDim[1]):
    ax = axes[i]

    # Set axis and title information
    xmin, xmax = plotArgDict['xAxisLimsList'][i]
    ax.set_xlim(xmin, xmax)
    ax.set(xlabel=xLabelList[i])
    ax.xaxis.label.set_size(16)

    if i==0:
      ax.set(ylabel=r'Percent of simulated events')
      ax.yaxis.label.set_size(16)

    if i==1:
      ax.set_title(plotArgDict['title'], fontsize=20)

    # Set bin information
    if i!=2:
      bins = np.linspace(0, 5, 6)-0.5
    else:
      bins = np.linspace(0, 11, 12)-0.5

    #-- Plot background case --#
    _, _, _,   = ax.hist(multBkgList[i], bins=bins, histtype = 'step', edgecolor='black', linestyle='-', linewidth=2, fill=False, log=True, label='Background', density=True)

    #-- Loop over signal cases --#
    for j in range(nCases):
      multSigList_ = multSigList[i]

      # Make plot
      Slabel = plotArgDict['sigObjectNameList'][j]+' Signal'
      _, _, _,   = ax.hist(multSigList_[j], bins=bins, histtype = 'step', edgecolor=colorList[j], linestyle='--', linewidth=2, fill=False, log=True, label=Slabel, density=True)

    if i==2:
      ax.legend(loc='upper right')

  #-- Show the plot --#
  plt.show()

# ##### `plotDataAugHists()`

def plotDataAugHists(objectBkg, objectAugBkg, plotArgDict):
    """
    Plot histograms of an object's pT, eta, phi for Background and Augmented Bkg.

    Inputs:
        objectBkg:      A background event's object's (e.g. MET) pT, eta, phi;
                        shape = (Nb, 3); Nb = number of background events
        objectAugBkg:   An augmented background event's object's (e.g. MET) pT, eta, phi;
                        shape = (Ns, 3); Ns = number of augmented background events (fake signal)
        plotArgDict:    Dictionary of plotting arguments (see example below)

    Outputs:
        Nothing returned; Show plot


    Example plotArgDict:
        plotArgDict = {}
        plotArgDict['pltDim']             = (1,3) # i.e. 1 row, 3 columns
        plotArgDict['xAxisLimsList']      = [(0, 1500), (-5, 5), (-np.pi, np.pi)]
        plotArgDict['title']              = r'MET'
        plotArgDict['nBins']              = 50
        plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
    """
    #-- Preliminary Figure Setup --#
    pltDim = plotArgDict['pltDim']
    fig_size = (7*pltDim[1], 7*pltDim[0])
    fig = plt.figure(constrained_layout=True, figsize=fig_size)
    gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
    axes = []
    for i in range(pltDim[1]):
        axes.append(fig.add_subplot(gs[:, i]))

    xLabelList = [r'$p_{\rm T}$', r'$\eta$', r'$\phi$']

    #-- Loop over axes to make plots --#
    for i in range(pltDim[1]):

        ax = axes[i]

        # Set axis and title information
        xmin, xmax = plotArgDict['xAxisLimsList'][i]
        ax.set_xlim(xmin, xmax)
        ax.set(xlabel=xLabelList[i])
        ax.xaxis.label.set_size(16)

        if i==0:
            ax.set(ylabel=r'Simulated events')
            ax.yaxis.label.set_size(16)

        if i==1:
            ax.set_title(plotArgDict['title'], fontsize=20)

        # Set bin information
        bins = np.linspace(xmin, xmax, plotArgDict['nBins'])

        #-- Plot background case --#
        _, _, _,   = ax.hist(objectBkg[:,i], bins=bins, histtype = 'step', edgecolor='black', linestyle='-', linewidth=2, fill=False, log=True, label='Background')
        _, _, _,   = ax.hist(objectAugBkg[:,i], bins=bins, histtype = 'step', edgecolor='purple', linestyle='--', linewidth=2, fill=False, log=True, label='Augmented Background')

        if i==0:
            ax.legend(loc='upper right')

    #-- Show the plot --#
    plt.show()

# ## OT results plotting functions

# ##### `plotScoreHists() `

def plotScoreHists(scoreBkg, scoreSigList, plotArgDict):
  """
  Plot a 1D histogram of the anomaly score for the background and each signal case

  Inputs:
    scoreBkg:      An array of anomaly scores for a set of background events;
                   shape = (Nb, ); Nb = number of background events
    scoreSigList:  A list of arrays of anomaly scores for nCases number of signal cases;
                   List of length nCases of arrays with shapes (Ns, 3);
                   Ns = number of signal events
    plotArgDict:    Dictionary of plotting arguments (see example below)

  Outputs:
    Nothing returned; Show plot

  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
  plotArgDict = {}
  plotArgDict['pltDim']             = (3,3)
  plotArgDict['xAxisLims']          = (0, 17500)
  plotArgDict['xLabel']             = r'Anomaly Score: $W_2^2(\cdot, \cdot)$'
  plotArgDict['yAxisLims']          = (1, 1e4)
  plotArgDict['yLabel']             = r'Counts'
  plotArgDict['title']              = r''
  plotArgDict['nBins']              = 100
  plotArgDict['logY']               = True
  plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
  plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  if 'density' in  list(plotArgDict.keys()):
    DENSITY = plotArgDict['density']
  else:
    DENSITY = False

  #-- Define color map --#
  # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(scoreSigList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))

  #-- Make histogram plot --#
  binsArr = np.linspace(xmin, xmax, plotArgDict['nBins'])
  _, _, _   = ax.hist(scoreBkg, bins=binsArr, histtype = 'step', edgecolor='black', linestyle='-', linewidth=2, fill=False, log=True, label='Background', density=DENSITY)

  for i in range(nCases):
    label = plotArgDict['sigObjectNameList'][i]
    _, _, _,   = ax.hist(scoreSigList[i], bins=binsArr, histtype = 'step', edgecolor=colorList[i], linestyle='--', linewidth=2, fill=False, log=True, label=label, density=DENSITY)

  #-- Show the plot with legend --#
  ax.legend()
  plt.show()

# ##### `plotMaxIndividualOTScoresPerEvent() `

def plotMaxIndividualOTScoresPerEvent(indxs, maxArr, plotArgDict):
  """
  Plot the (regulated) Significance Improvement (SI) curve from values
  precalculated using the calcROCmetrics() function.

      x-axis is the Signal (Acceptance) Efficiency <=> TPR <=> \eps_S
      y-axis is the (regulated) Significance Improvement <=> SI := TPR/sqrt(FPR + SIreg) <=> SI := eps_S/sqrt(eps_B + SIreg)

  Inputs:
    tprList:      List of TPR arrays for each background to signal type pairing; Length=nCases
                  tprList[i] is an array of shape (Q,) with Q>2
    SIList:       List of SI arrays for each background to signal type pairing; Length=nCases
                  SIList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
    plotArgDict = {}
    plotArgDict['pltDim']    = (3,3)
    plotArgDict['xAxisLims'] = (0, 1.05)
    plotArgDict['xLabel']    = r'Event Pairs'
    plotArgDict['yAxisLims'] = (0, 10)
    plotArgDict['yLabel']    = r'Max OT distance'
    plotArgDict['title']     = r'wBS_sig_A'
    plotArgDict['coreColorList'] = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
  """
  nEventPairs = maxArr.shape[0]

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  #-- Get color list for plotting --#
  coreColorList = plotArgDict['coreColorList']
  colorList     = []
  for i in range(maxArr.shape[0]):
    indx = indxs[i]
    colorList.append(coreColorList[indx])

  #-- Make plot --#
  eventPairsArr = np.arange(nEventPairs)
  ax.scatter(eventPairsArr, maxArr, color=colorList)

  #-- Plot key --#
  deltaY = (ymax - ymin)
  deltaX = (xmax - xmin)
  ax.text(xmin + 0.20*deltaX, ymin + 0.95*deltaY, r'IOT$_{\rm MET}$', color=coreColorList[0], fontsize=16, fontweight='bold')
  ax.text(xmin + 0.40*deltaX, ymin + 0.95*deltaY,   r'IOT$_{\rm e}$', color=coreColorList[1], fontsize=16, fontweight='bold')
  ax.text(xmin + 0.60*deltaX, ymin + 0.95*deltaY, r'IOT$_{\rm \mu}$', color=coreColorList[2], fontsize=16, fontweight='bold')
  ax.text(xmin + 0.80*deltaX, ymin + 0.95*deltaY, r'IOT$_{\rm jet}$', color=coreColorList[3], fontsize=16, fontweight='bold')

  #-- Show plot --#
  plt.show()

# ## Performance metric plotting functions

# ### Repeat runs

# ##### `plotROCcurve_errorBand() `

def plotROCcurve_errorBand(av_fprList, std_fprList, plotArgDict, TYPE='Classic'):
  """
  Plot ROC curves with error bands from values precalculated using the calcROCmetrics(..., INTERPOLATE=True) function.

  Inputs:
    av_fprList:   List of average FPR arrays for each background to signal type pairing; Length=nCases
                  av_fprList[i] is an array of shape (Q,) with Q>2
    std_fprList:  List of std FPR arrays for each background to signal type pairing; Length=nCases
                  std_fprList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)
    TYPE:         TYPE of ROC curve to plot; choices are 'Classic' (default) or 'Modern'
                  (see below for explanation)


  Explanation of TYPE choices:
    For all TYPE choices the x-axis is defined as the Signal (Acceptance)
    Efficiency <=> TPR <=> \eps_S

    if TYPE == 'Classic':
        y-axis is the Background Rejection Efficiency <=> TNR <=> 1 - FPR <=> 1 - \eps_B
    otherwise:
        y-axis is the Background (Acceptance) Efficiency <=> FPR <=> \eps_B


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows (assuming TYPE == 'Classic')
      plotArgDict = {}
      plotArgDict['pltDim']    = (3,3)
      plotArgDict['xAxisLims'] = (0, 1.05)
      plotArgDict['xLabel']    = r'Signal Efficiency (TPR)' # OR r'$\eps_S$'
      plotArgDict['yAxisLims'] = (0, 1.05)
      plotArgDict['yLabel']    = r'Background Rejection (TNR)' # OR r'$1 - \eps_B$'
      plotArgDict['title']     = r'ROC curve, $W_2^2(\cdot, \cdot)$ anomaly score'
      plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
      plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """
  #-- Set base TPR --#
  base_tpr = np.linspace(0, 1, 101)

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  #-- Define color map --#
  # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(av_fprList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get ROC curve components --#
    av_fpr, std_fpr, tpr = av_fprList[i], std_fprList[i], base_tpr

    #-- Get appropriate shading color --#
    colorAlpha = list(colorList[i])
    colorAlpha[3] = 0.5
    colorAlpha = tuple(colorAlpha)

    #-- Plot ROC curve --#
    legendLabel = plotArgDict['sigObjectNameList'][i]
    if TYPE == 'Classic':
      av_tnr = 1. - av_fpr
      _   = ax.plot(tpr, av_tnr, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
      _   = ax.fill_between(tpr, av_tnr-std_fpr, av_tnr+std_fpr, color=colorAlpha) # Note: std_tnr = std_fpr
    else:
      _   = ax.plot(tpr, av_fpr, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
      _   = ax.fill_between(tpr, av_fpr-std_fpr, av_fpr+std_fpr, color=colorAlpha)


  #-- Show the plot with legend --#
  if TYPE == 'Classic':
    ax.legend(loc='lower left')
  else:
    ax.legend(loc='upper left')
  plt.show()

# ##### `plotROCcurve_errorBand_specificMethod() `

def plotROCcurve_errorBand_specificMethod(av_fprList, std_fprList, plotArgDict, TYPE='Classic', NSIGFIGS=4, saveFigPath=''):
  """
  Plot ROC curves with error bands from values precalculated using the calcROCmetrics(..., INTERPOLATE=True) function.
  Instead of comparing one method on all signal types, we compare all methods on each signal type.
  Also adds ability to report AUC on plot.

  Inputs:
    av_fprList:   List of average FPR arrays for each background to signal type pairing; Length=nCases
                  av_fprList[i] is an array of shape (Q,) with Q>2
    std_fprList:  List of std FPR arrays for each background to signal type pairing; Length=nCases
                  std_fprList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)
    TYPE:         TYPE of ROC curve to plot; choices are 'Classic' (default) or 'Modern'
                  (see below for explanation)


  Explanation of TYPE choices:
    For all TYPE choices the x-axis is defined as the Signal (Acceptance)
    Efficiency <=> TPR <=> \eps_S

    if TYPE == 'Classic':
        y-axis is the Background Rejection Efficiency <=> TNR <=> 1 - FPR <=> 1 - \eps_B
    otherwise:
        y-axis is the Background (Acceptance) Efficiency <=> FPR <=> \eps_B


  Example plotArgDict:
  # Define colors to use for 4 signal types
      METHOD_COLOR_ARR = np.array([ RGBAtoRGBAtuple((1, 27, 62, 1), TYPE='list'),
                              RGBAtoRGBAtuple((81, 39, 54, 1), TYPE='list'),
                              RGBAtoRGBAtuple((48, 53, 13, 1), TYPE='list'),
                              RGBAtoRGBAtuple((141, 109, 58, 1), TYPE='list')
                              ])
  # Or use a matplotlib color map
      METHOD_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows (assuming TYPE == 'Classic')
      plotArgDict = {}
      plotArgDict['pltDim']    = (3,3)
      plotArgDict['xAxisLims'] = (0, 1.05)
      plotArgDict['xLabel']    = r'$\epsilon_S$' # OR  r'Signal Efficiency (TPR)'
      plotArgDict['yAxisLims'] = (0, 1.05)
      plotArgDict['yLabel']    = r'$1 - \epsilon_B$' # OR r'Background Rejection (TNR)'
      plotArgDict['title']     = r'Performance on signal case %s'%(sigName)
      plotArgDict['CMAP']            = METHOD_COLOR_ARR

      plotArgDict['methodNameList']  = methodNameList
      plotArgDict['avAUCList']       = av_aucList
      plotArgDict['stdAUCList']      = std_aucList
      plotArgDict['delta_yposPerc'] = 0.04
      plotArgDict['xposPerc']       = 0.05
      plotArgDict['yposPerc']       = 0.2
  """
  #-- Set base TPR --#
  base_tpr = np.linspace(0, 1, 101)

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  #-- Define color map --#
  # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(av_fprList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))


  #-- Preliminary alternate legend setting --#
  # Set initial position of text positions
  delta_ypos = plotArgDict['delta_yposPerc']*plotArgDict['yAxisLims'][1]
  if TYPE == 'Classic':
    xpos = plotArgDict['xposPerc']*plotArgDict['xAxisLims'][1]
    ypos = plotArgDict['yposPerc']*plotArgDict['yAxisLims'][1]
    ax.text(xpos, ypos, plotArgDict['title'], size=14, weight="bold")
    ypos -= delta_ypos

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get ROC curve components --#
    av_fpr, std_fpr, tpr = av_fprList[i], std_fprList[i], base_tpr

    #-- Get appropriate shading color --#
    colorAlpha = list(colorList[i])
    colorAlpha[3] = 0.5
    colorAlpha = tuple(colorAlpha)

    #-- Plot ROC curve --#
    aucMean = roundToSigFig(plotArgDict['avAUCList'][i], NSIGFIGS)
    aucStd  = roundToSigFig(plotArgDict['stdAUCList'][i], NSIGFIGS)
    legendLabel = plotArgDict['methodNameList'][i] + r', AUC: %s $\pm$ %s'%(aucMean, aucStd)

    if plotArgDict['methodNameList'][i] != '3D OT+One-class SVM':

      if TYPE == 'Classic':
        av_tnr = 1. - av_fpr
        _   = ax.plot(tpr, av_tnr, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
        _   = ax.fill_between(tpr, av_tnr-std_fpr, av_tnr+std_fpr, color=colorAlpha) # Note: std_tnr = std_fpr
      else:
        _   = ax.plot(tpr, av_fpr, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
        _   = ax.fill_between(tpr, av_fpr-std_fpr, av_fpr+std_fpr, color=colorAlpha)

    #-- Alternate legend --#
    ax.text(xpos, ypos, legendLabel, size=14, color=colorList[i])
    ypos -= delta_ypos

  #-- Save figure if desired --#
  if saveFigPath!='':
    print("Saving figure to file: ", saveFigPath)
    plt.savefig(saveFigPath)

  plt.show()


# ##### `plotInvROCcurve_errorBand() `

def plotInvROCcurve_errorBand(av_fprInvList, std_fprInvList, plotArgDict, minTPR=0.05):
  """
  Plot FPR inverse curves with error bands from values precalculated using the calcROCmetrics(..., INTERPOLATE=True) function.

    x-axis is the Signal (Acceptance) Efficiency <=> TPR <=> \eps_S
    y-axis is the inverted the Background (Acceptance) Efficiency <=> FPR <=> \eps_B

  NOTE: This is often also called a "ROC curve" however this is misleading because
  the AUC is NOT the area under this curve, so it is NOT a ROC curve.

  Inputs:
    av_fprList:   List of average FPR arrays for each background to signal type pairing; Length=nCases
                  av_fprList[i] is an array of shape (Q,) with Q>2
    std_fprList:  List of std FPR arrays for each background to signal type pairing; Length=nCases
                  std_fprList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)
    minTPR:       Minimum TPR value to avoid wonky-ness from low statistics; must be between 0. and 1.


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
      plotArgDict = {}
      plotArgDict['pltDim']    = (3,3)
      plotArgDict['xAxisLims'] = (0, 1.05)
      plotArgDict['xLabel']    = r'$\epsilon_S$ (TPR)'
      plotArgDict['yAxisLims'] = (1, 1e4)
      plotArgDict['yLabel']    = r'$\epsilon_B^{-1}$ (FPR$^{-1}$)'
      plotArgDict['title']     = r'$W_2^2(\cdot, \cdot)$ anomaly score performance'
      plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
      plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Set base TPR --#
  base_tpr = np.linspace(0, 1, 101)

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set_yscale('log')
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  #-- Define color map --#
  nCases    = len(av_fprInvList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases)) # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get inv ROC curve components --#
    av_fprInv, std_fprInv, tpr = av_fprInvList[i], std_fprInvList[i], base_tpr

    #-- Trim inv ROC curve components to account for minTPR --#
    assert (minTPR >=0 and minTPR <=1)  # minTPR must be in range [0,1]
    minMask = minTPR <= tpr             # Sets TPR values less than minTPR to False

    av_fprInv_trimmed   = av_fprInv[minMask]
    std_fprInv_trimmed  = std_fprInv[minMask]
    tpr_trimmed         = tpr[minMask]

    #-- Get appropriate shading color --#
    colorAlpha = list(colorList[i])
    colorAlpha[3] = 0.5
    colorAlpha = tuple(colorAlpha)

    #-- Plot inv ROC curve --#
    legendLabel = plotArgDict['sigObjectNameList'][i]
    _ = ax.plot(tpr_trimmed, av_fprInv_trimmed, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
    _ = ax.fill_between(tpr_trimmed, av_fprInv_trimmed-std_fprInv_trimmed, av_fprInv_trimmed+std_fprInv_trimmed, color=colorAlpha)

  #-- Show the plot with legend --#
  ax.legend(loc='upper right')
  plt.show()

# ##### `plotSIcurve_errorBand() `

def plotSIcurve_errorBand(av_SIList, std_SIList, plotArgDict, minTPR=0.05, NSIGFIGS=4):
  """
   Plot the (regulated) Significance Improvement (SI) curve with error bands from values precalculated using the calcROCmetrics(..., INTERPOLATE=True) function.

      x-axis is the Signal (Acceptance) Efficiency <=> TPR <=> \eps_S
      y-axis is the (regulated) Significance Improvement <=> SI := TPR/sqrt(FPR + SIreg) <=> SI := eps_S/sqrt(eps_B + SIreg)

  Inputs:
    av_SIList:    List of average SI arrays for each background to signal type pairing; Length=nCases
                  av_SIList[i] is an array of shape (Q,) with Q>2
    std_SIList:   List of std SI arrays for each background to signal type pairing; Length=nCases
                  std_SIList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)
    minTPR:       Minimum TPR value to avoid wonky-ness from low statistics; must be between 0. and 1.
    NSIGFIGS:     Number of significant figures to use when reporting max SI and corresponding TPR

  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
    plotArgDict = {}
    plotArgDict['pltDim']    = (3,3)
    plotArgDict['xAxisLims'] = (0, 1.05)
    plotArgDict['xLabel']    = r'$\epsilon_S$ (TPR)'
    plotArgDict['yAxisLims'] = (0, 10)
    plotArgDict['yLabel']    = r'$\epsilon_S/ \sqrt{\epsilon_B}$ (SI)'
    plotArgDict['title']     = r'SI Curve, $W_2^2(\cdot, \cdot)$ anomaly score'
    plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
    plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Set base TPR --#
  base_tpr = np.linspace(0, 1, 101)

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  #-- Define color map --#
  # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(av_SIList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get SI curve components --#
    av_si, std_si, tpr = av_SIList[i], std_SIList[i], base_tpr

    #-- Trim SI curve components to account for minTPR --#
    assert (minTPR >=0 and minTPR <=1)  # minTPR must be in range [0,1]
    minMask = minTPR <= tpr             # Sets TPR values less than minTPR to False

    av_si_trimmed   = av_si[minMask]
    std_si_trimmed  = std_si[minMask]
    tpr_trimmed     = tpr[minMask]

    #-- Get appropriate shading color --#
    colorAlpha = list(colorList[i])
    colorAlpha[3] = 0.5
    colorAlpha = tuple(colorAlpha)

    #-- Plot SI curve --#
    legendLabel = plotArgDict['sigObjectNameList'][i]
    _ = ax.plot(tpr_trimmed, av_si_trimmed, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
    _ = ax.fill_between(tpr_trimmed, av_si_trimmed-std_si_trimmed, av_si_trimmed+std_si_trimmed, color=colorAlpha)

    # Report max in trimmed TPR range
    aMaxSI = np.argmax(av_si_trimmed)
    print("Max SI for %s is %s \pm %s at TPR = %s"%(legendLabel, roundToSigFig(av_si_trimmed[aMaxSI], NSIGFIGS), roundToSigFig(std_si_trimmed[aMaxSI], NSIGFIGS), roundToSigFig(tpr_trimmed[aMaxSI], NSIGFIGS)))

  #-- Show the plot with legend --#
  if 'nLegendColumns' in plotArgDict.keys():
    ax.legend(loc='upper right', ncol=plotArgDict['nLegendColumns'])
  else:
    ax.legend(loc='upper right')
  plt.show()

# ##### `plotSIcurve_errorBand_specificMethod() `

def plotSIcurve_errorBand_specificMethod(av_SIList, std_SIList, plotArgDict, minTPR=0.05, NSIGFIGS=4, saveFigPath=''):
  """
  Plot the (regulated) Significance Improvement (SI) curve with error bands from values precalculated using the calcROCmetrics(..., INTERPOLATE=True)
  function. Instead of comparing one method on all signal types, we compare all methods on each signal type.
  Also adds ability to report max SI (and corresponding TPR) on plot.

      x-axis is the Signal (Acceptance) Efficiency <=> TPR <=> \eps_S
      y-axis is the (regulated) Significance Improvement <=> SI := TPR/sqrt(FPR + SIreg) <=> SI := eps_S/sqrt(eps_B + SIreg)


  Inputs:
    av_SIList:    List of average SI arrays for each background to signal type pairing; Length=nCases
                  av_SIList[i] is an array of shape (Q,) with Q>2
    std_SIList:   List of std SI arrays for each background to signal type pairing; Length=nCases
                  std_SIList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)
    minTPR:       Minimum TPR value to avoid wonky-ness from low statistics; must be between 0. and 1.
    NSIGFIGS:     Number of significant figures to use when reporting max SI and corresponding TPR
    saveFigPath:  Path to save figure; default is to not save figure

  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
    plotArgDict = {}
    plotArgDict['pltDim']    = (3,3)
    plotArgDict['xAxisLims'] = (0, 1.05)
    plotArgDict['xLabel']    = r'$\epsilon_S$ (TPR)'
    plotArgDict['yAxisLims'] = (0, 10)
    plotArgDict['yLabel']    = r'$\epsilon_S/ \sqrt{\epsilon_B}$ (SI)'
    plotArgDict['title']     = r'SI Curve, $W_2^2(\cdot, \cdot)$ anomaly score'
    plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
    plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']

    plotArgDict['methodNameList']  = methodNameList
    plotArgDict['avMaxSIList']     = av_maxSIList
    plotArgDict['stdMaxSIList']    = std_maxSIList
    plotArgDict['corrTPRList']     = corr_tprList
    plotArgDict['delta_yposPerc'] = 0.04
    plotArgDict['xposPerc']       = 0.05
    plotArgDict['yposPerc']       = 0.2
  """

  #-- Set base TPR --#
  base_tpr = np.linspace(0, 1, 101)

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  #-- Define color map --#
  # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(av_SIList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))

  #-- Preliminary alternate legend setting --#
  # Set initial position of text positions
  delta_ypos = plotArgDict['delta_yposPerc']*plotArgDict['yAxisLims'][1]
  xpos = plotArgDict['xposPerc']*plotArgDict['xAxisLims'][1]
  ypos = plotArgDict['yposPerc']*plotArgDict['yAxisLims'][1]
  ax.text(xpos, ypos, plotArgDict['title'], size=14, weight="bold")
  ypos -= delta_ypos

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get SI curve components --#
    av_si, std_si, tpr = av_SIList[i], std_SIList[i], base_tpr

    #-- Trim SI curve components to account for minTPR --#
    assert (minTPR >=0 and minTPR <=1)  # minTPR must be in range [0,1]
    minMask = minTPR <= tpr             # Sets TPR values less than minTPR to False

    av_si_trimmed   = av_si[minMask]
    std_si_trimmed  = std_si[minMask]
    tpr_trimmed     = tpr[minMask]

    #-- Get appropriate shading color --#
    colorAlpha = list(colorList[i])
    colorAlpha[3] = 0.5
    colorAlpha = tuple(colorAlpha)

    #-- Plot SI curve --#
    maxSImean = roundToSigFig(plotArgDict['avMaxSIList'][i], NSIGFIGS)
    maxSIstd  = roundToSigFig(plotArgDict['stdMaxSIList'][i], NSIGFIGS)
    corrTPR   = roundToSigFig(plotArgDict['corrTPRList'][i], 2)
    legendLabel = plotArgDict['methodNameList'][i] + r', max SI: %s $\pm$ %s ($\epsilon_s$=%s)'%(maxSImean, maxSIstd, corrTPR)

    if plotArgDict['methodNameList'][i] != '3D OT+One-class SVM': #!

      _ = ax.plot(tpr_trimmed, av_si_trimmed, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
      _ = ax.fill_between(tpr_trimmed, av_si_trimmed-std_si_trimmed, av_si_trimmed+std_si_trimmed, color=colorAlpha)

    #-- Alternate legend --#
    ax.text(xpos, ypos, legendLabel, size=14, color=colorList[i])
    ypos -= delta_ypos

  #-- Save figure if desired --#
  if saveFigPath!='':
    print("Saving figure to file: ", saveFigPath)
    plt.savefig(saveFigPath)

  plt.show()

# ### Single test run

# ##### `plotROCcurve() `

def plotROCcurve(aucList, fprList, tprList, plotArgDict, TYPE='Classic'):
  """
  Plot ROC curves from values precalculated using the calcROCmetrics() function.

  Inputs:
    aucList:      List of AUC scores for each background to signal type pairing; Length=nCases
    fprList:      List of FPR arrays for each background to signal type pairing; Length=nCases
                  fprList[i] is an array of shape (Q,) with Q>2
    tprList:      List of TPR arrays for each background to signal type pairing; Length=nCases
                  tprList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)
    TYPE:         TYPE of ROC curve to plot; choices are 'Classic' (default) or 'Modern'
                  (see below for explanation)


  Explanation of TYPE choices:
    For all TYPE choices the x-axis is defined as the Signal (Acceptance)
    Efficiency <=> TPR <=> \eps_S

    if TYPE == 'Classic':
        y-axis is the Background Rejection Efficiency <=> TNR <=> 1 - FPR <=> 1 - \eps_B
    otherwise:
        y-axis is the Background (Acceptance) Efficiency <=> FPR <=> \eps_B


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows (assuming TYPE == 'Classic')
      plotArgDict = {}
      plotArgDict['pltDim']    = (3,3)
      plotArgDict['xAxisLims'] = (0, 1.05)
      plotArgDict['xLabel']    = r'Signal Efficiency (TPR)' # OR r'$\eps_S$'
      plotArgDict['yAxisLims'] = (0, 1.05)
      plotArgDict['yLabel']    = r'Background Rejection (TNR)' # OR r'$1 - \eps_B$'
      plotArgDict['title']     = r'ROC curve, $W_2^2(\cdot, \cdot)$ anomaly score'
      plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
      plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  #-- Define color map --#
  # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(aucList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get ROC curve components --#
    auc, fpr, tpr = aucList[i], fprList[i], tprList[i]

    #-- Plot ROC curve --#
    legendLabel = plotArgDict['sigObjectNameList'][i]+': AUC='+str(auc)
    if TYPE == 'Classic':
      tnr = 1. - fpr
      _   = ax.plot(tpr, tnr, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)
    else:
      _   = ax.plot(tpr, fpr, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)

  #-- Show the plot with legend --#
  if TYPE == 'Classic':
    ax.legend(loc='lower left')
  else:
    ax.legend(loc='upper left')
  plt.show()

# ##### `plotInvROCcurve() `

def plotInvROCcurve(aucList, fprInvList, tprList, plotArgDict):
  """
  Plot FPR inverse curves from values precalculated using the calcROCmetrics() function.

    x-axis is the Signal (Acceptance) Efficiency <=> TPR <=> \eps_S
    y-axis is the inverted the Background (Acceptance) Efficiency <=> FPR <=> \eps_B

  NOTE: This is often also called a "ROC curve" however this is misleading because
  the AUC is NOT the area under this curve, so it is NOT a ROC curve.

  Inputs:
    aucList:      List of AUC scores for each background to signal type pairing; Length=nCases
    fprInvList:   List of inverse FPR arrays for each background to signal type pairing; Length=nCases
                  fprInvList[i] is a masked array of shape (Q,) with Q>2 with division by zero cases masked
    tprList:      List of TPR arrays for each background to signal type pairing; Length=nCases
                  tprList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
      plotArgDict = {}
      plotArgDict['pltDim']    = (3,3)
      plotArgDict['xAxisLims'] = (0, 1.05)
      plotArgDict['xLabel']    = r'$\epsilon_S$ (TPR)'
      plotArgDict['yAxisLims'] = (1, 1e4)
      plotArgDict['yLabel']    = r'$\epsilon_B^{-1}$ (FPR$^{-1}$)'
      plotArgDict['title']     = r'$W_2^2(\cdot, \cdot)$ anomaly score performance'
      plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
      plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set_yscale('log')
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  #-- Define color map --#
  nCases    = len(aucList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases)) # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get ROC curve components --#
    auc, fprInv, tpr = aucList[i], fprInvList[i], tprList[i]

    #-- Plot ROC curve --#
    legendLabel = plotArgDict['sigObjectNameList'][i]+': AUC='+str(auc)
    _ = ax.plot(tpr, fprInv, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)

  #-- Show the plot with legend --#
  ax.legend(loc='upper right')
  plt.show()

# ##### `plotSIcurve() `

def plotSIcurve(tprList, SIList, plotArgDict):
  """
   Plot the (regulated) Significance Improvement (SI) curve from values
   precalculated using the calcROCmetrics() function.

      x-axis is the Signal (Acceptance) Efficiency <=> TPR <=> \eps_S
      y-axis is the (regulated) Significance Improvement <=> SI := TPR/sqrt(FPR + SIreg) <=> SI := eps_S/sqrt(eps_B + SIreg)

  Inputs:
    tprList:      List of TPR arrays for each background to signal type pairing; Length=nCases
                  tprList[i] is an array of shape (Q,) with Q>2
    SIList:       List of SI arrays for each background to signal type pairing; Length=nCases
                  SIList[i] is an array of shape (Q,) with Q>2
    plotArgDict:  Dictionary of plotting arguments (see example below)


  Example plotArgDict:
  # Define colors to use for 4 signal types
      SIGNAL_COLOR_ARR = np.array([ RGBAtoRGBAtuple((231, 186, 81, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((140, 162, 82, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((165, 81, 148, 1), TYPE='list'),
                                    RGBAtoRGBAtuple((214, 97, 107, 1), TYPE='list')
                                  ])
  # Or use a matplotlib color map
      SIGNAL_COLOR_ARR = plt.colormaps['rainbow'].reversed()

  # Then define dictionary as follows
    plotArgDict = {}
    plotArgDict['pltDim']    = (3,3)
    plotArgDict['xAxisLims'] = (0, 1.05)
    plotArgDict['xLabel']    = r'$\epsilon_S$ (TPR)'
    plotArgDict['yAxisLims'] = (0, 10)
    plotArgDict['yLabel']    = r'$\epsilon_S/ \sqrt{\epsilon_B}$ (SI)'
    plotArgDict['title']     = r'SI Curve, $W_2^2(\cdot, \cdot)$ anomaly score'
    plotArgDict['CMAP']               = SIGNAL_COLOR_ARR
    plotArgDict['sigObjectNameList']  = [r'$A$', r'$h^0$', r'$h^\pm$', r'$LQ$']
  """

  #-- Preliminary Figure Setup --#
  pltDim = plotArgDict['pltDim']
  fig_size = (3*pltDim[1], 3*pltDim[0])

  fig = plt.figure(constrained_layout=False, figsize=fig_size)

  gs = GridSpec(pltDim[0], pltDim[1], figure=fig, hspace=0.1)
  ax = fig.add_subplot(gs[:, :])

  xmin, xmax = plotArgDict['xAxisLims']
  ax.set_xlim(xmin, xmax)
  ax.set(xlabel=plotArgDict['xLabel'])
  ax.xaxis.label.set_size(16)

  ymin, ymax = plotArgDict['yAxisLims']
  ax.set_ylim(ymin, ymax)
  ax.set(ylabel=plotArgDict['yLabel'])
  ax.yaxis.label.set_size(16)

  ax.set_title(plotArgDict['title'], fontsize=20)

  #-- Define color map --#
  # https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
  nCases    = len(tprList)  # Number of cases to plot
  if isinstance(plotArgDict['CMAP'], np.ndarray) and (plotArgDict['CMAP'].shape[0]==nCases):
    colorList = plotArgDict['CMAP']
  else:
    CMAP      = plotArgDict['CMAP']
    colorList = CMAP(np.linspace(0,1,nCases))

  #-- Loop over signal cases --#
  for i in range(nCases):

    #-- Get ROC curve components --#
    si, tpr = SIList[i], tprList[i]

    #-- Plot ROC curve --#
    legendLabel = plotArgDict['sigObjectNameList'][i]
    _ = ax.plot(tpr, si, color=colorList[i], linestyle='-', linewidth=2, label=legendLabel)

  #-- Show the plot with legend --#
  ax.legend(loc='upper right')
  plt.show()

# # Machine Learning Functions

# ## SVM Classification

# ##### `SVM_ROC_Metrics()`

def SVM_ROC_Metrics(bkg_scores, sig_scores, C = 1.0, gamma = 'scale'):
  np.random.shuffle(bkg_scores)
  np.random.shuffle(sig_scores)

  bkg_indicator = np.zeros(len(bkg_scores))
  sig_indicator = np.ones(len(sig_scores))

  bkg_x_train, bkg_x_test, bkg_y_train, bkg_y_test = train_test_split(bkg_scores, bkg_indicator, test_size=0.25, random_state=123)
  bkg_x_train, bkg_x_test = np.array(bkg_x_train).reshape(-1, 1), np.array(bkg_x_test).reshape(-1, 1)

  sig_x_train, sig_x_test, sig_y_train, sig_y_test = train_test_split(sig_scores, sig_indicator, test_size=0.25, random_state=123)
  sig_x_train, sig_x_test = np.array(sig_x_train).reshape(-1, 1), np.array(sig_x_test).reshape(-1, 1)

  x_train = np.concatenate((bkg_x_train, sig_x_train), axis = 0)
  y_train = np.concatenate((bkg_y_train, sig_y_train))

  x_test = np.concatenate((bkg_x_test, sig_x_test), axis = 0)
  y_test = np.concatenate((bkg_y_test, sig_y_test))

  # x = np.concatenate((bkg_scores, sig_scores))

  # y = [0] * len(bkg_scores) + [1] * len(sig_scores)

  # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
  # x_train, x_test = np.array(x_train).reshape(-1, 1), np.array(x_test).reshape(-1, 1)

  # Create the SVM with RBF kernel
  clf = SVC(kernel='rbf', C = C, gamma = gamma)

  # Train the SVM
  clf.fit(x_train, y_train)

  # Calibrate the SVC model
  calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
  calibrated_clf.fit(x_train, y_train)

  # Get prediction probabilities for the test set
  y_prob = calibrated_clf.predict_proba(x_test)[:,1]

  # # Get predicted probabilities for the positive class (1 - Signal)
  # y_prob = clf.predict_proba(x_test)[:, 1]

  # Compute ROC curve and AUC
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  roc_auc = roc_auc_score(y_test, y_prob)
  tnr = 1-fpr

  return tnr,tpr,roc_auc

# ##### `SVM_Classification_With_Best_Hyperparameters()`

def SVM_Classification_With_Best_Hyperparameters(bkg_scores, sig_scores):
    C_range = [1,10]
    gamma_range = [1,10]
    auc_list = []
    tnr_list = []
    tpr_list = []

    for C in C_range:
      for gamma in gamma_range:
        tnr,tpr,roc_auc = SVM_ROC_Metrics(bkg_scores, sig_scores, C, gamma)
        tnr_list.append(tnr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)

    max_index = auc_list.index(max(auc_list))

    return tnr_list[max_index], tpr_list[max_index], auc_list[max_index]

# ## kNN Classification

# ##### `kNN_with_score_list()`

def kNN_with_score_list(bkg_score, sig_score, train_number, val_number, test_number, sig_bkg_ratio, neighbor_list):
  np.random.seed(123)
  np.random.shuffle(bkg_score)
  np.random.shuffle(sig_score)

  bkg_indicator = np.zeros(len(bkg_score))
  sig_indicator = np.ones(len(sig_score))

  x_train = np.concatenate((bkg_score[:train_number],sig_score[:train_number*sig_bkg_ratio]))
  x_train = x_train.reshape(-1,1)
  y_train = np.concatenate((bkg_indicator[:train_number],sig_indicator[:train_number*sig_bkg_ratio]))

  x_val = np.concatenate((bkg_score[train_number:train_number+val_number],sig_score[train_number*sig_bkg_ratio:(train_number+val_number)*sig_bkg_ratio]))
  x_val = x_val.reshape(-1,1)
  y_val = np.concatenate((bkg_indicator[train_number:train_number+val_number],sig_indicator[train_number*sig_bkg_ratio:(train_number+val_number)*sig_bkg_ratio]))

  x_trainval = np.concatenate((bkg_score[:train_number+val_number],sig_score[:(train_number+val_number)*sig_bkg_ratio]))
  x_trainval = x_trainval.reshape(-1,1)
  y_trainval = np.concatenate((bkg_indicator[:train_number+val_number],sig_indicator[:(train_number+val_number)*sig_bkg_ratio]))

  x_test = np.concatenate((bkg_score[train_number+val_number:train_number+val_number+test_number],sig_score[(train_number+val_number)*sig_bkg_ratio:(train_number+val_number+test_number)*sig_bkg_ratio]))
  x_test = x_test.reshape(-1,1)
  y_test = np.concatenate((bkg_indicator[train_number+val_number:train_number+val_number+test_number],sig_indicator[(train_number+val_number)*sig_bkg_ratio:(train_number+val_number+test_number)*sig_bkg_ratio]))

  # print(len(bkg_score),len(sig_score))
  # print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_trainval.shape, y_trainval.shape, x_test.shape, y_test.shape)

  models, pred_vals = [], []
  for neighbor in neighbor_list:
    # print("Processing neighbor k=", neighbor)
    model = KNeighborsClassifier(n_neighbors=neighbor)
    model.fit(x_train,y_train)
    pred_val = model.predict_proba(x_val)[:,1]
    models.append(model)
    pred_vals.append(pred_val)

  auc_list = []
  for pred in pred_vals:
    auc = roc_auc_score(y_val, pred)
    auc_list.append(auc)

  max_index = auc_list.index(max(auc_list))
  best_k = neighbor_list[max_index]
  # print(max(auc_list))
  # print(best_k)
  # print(auc_list)

  best_model = KNeighborsClassifier(n_neighbors=best_k)
  best_model.fit(x_trainval, y_trainval)
  best_pred = best_model.predict_proba(x_test)[:,1]
  best_auc = roc_auc_score(y_test, best_pred)

  return best_auc, best_k

# ##### `kNN_with_distance_matrix()`

def kNN_with_distance_matrix(l_matrix, labels, train_number, val_number, test_number, neighbor_list, AUC_list = False):
    '''
    Inputs:

        l_matrix:          The distance matrix(2D np array) of the two types of jets. Shape:(njets, njets)
                          (NOTE: This matrix must be symmetric and tracless for it to be a meaningful OT distance matrix)

        labels:           An array of labels for which type of particle each row/column belongs to. Shape(njets,)

        train_number:     The number of jets used for training the kNN model, set to be the first "train_number" of rows/cols

        val_number:       The number of jets used for the validation process(determining the best number of neighbors), set to be the "val_number" of rows/cols right after the training rows/cols

        test_number:      The number of jets used for the testing process, set to be the "test_number" of rows/cols right after the validation rows/cols

        neighbor_list:   The list of neighbors wanted to be used for the kNN classification

        AUC_list:         Default is False. When set to True, return the whole list of AUCs during the validation process

    Outputs:

        best_auc:         The AUC computed using the best-k-neighbor returned by the validation process.

        auc_list:         The entire list of AUC computed during the validation process. Only returned if AUC_list is set to be True

    '''

    # Split the distance matrix and labels into training, validation and testing sets
    l_matrix_train         = l_matrix[:train_number, :train_number]
    l_matrix_val_train     = l_matrix[train_number:train_number+val_number, :train_number]
    l_matrix_trainval      = l_matrix[:train_number+val_number, :train_number+val_number]
    l_matrix_test_trainval = l_matrix[train_number+val_number:test_number + train_number + val_number, :train_number+val_number]

    # Split the labels into training, validation and testing sets
    train_labels           = labels[:train_number]
    val_labels             = labels[train_number:train_number+val_number]
    trainval_labels        = labels[:train_number+val_number]
    test_labels            = labels[train_number+val_number:test_number+train_number+val_number]

    models, pred_vals = [], []

    for neighbor in tqdm(neighbor_list, desc='Fitting Models'):
        model = KNeighborsClassifier(n_neighbors=neighbor, metric='precomputed')
        model.fit(l_matrix_train, train_labels)
        pred_val = model.predict_proba(l_matrix_val_train)[:,1]
        models.append(model)
        pred_vals.append(pred_val)

    auc_list = []

    # Compute the AUC for each k-neighbor model
    for pred in pred_vals:
        auc = roc_auc_score(val_labels, pred)
        auc_list.append(auc)

    max_index = auc_list.index(max(auc_list))

    # Get the best k-neighbor model
    best_k_neighbor = neighbor_list[max_index]

    best_model = KNeighborsClassifier(n_neighbors=best_k_neighbor, metric='precomputed')

    # Train the best k-neighbor model
    best_model.fit(l_matrix_trainval, trainval_labels)

    # Get the prediction probabilities for the testing set
    best_pred = best_model.predict_proba(l_matrix_test_trainval)[:,1]

    kNN_metrics = kNN_ROC_metrics(test_labels, best_pred, Interpolate = True)

    best_auc = roc_auc_score(test_labels, best_pred)

    if AUC_list == True:
        return best_auc, best_k_neighbor, best_model, auc_list

    else:
        return best_auc, best_k_neighbor, best_model, kNN_metrics

# ##### `kNN_cross_validation()`

def kNN_cross_validation(l_matrix, labels, neighbor_list, k_fold):
    '''
    This function is basically a wrapper for the kNN_with_distance_matrix function, but with the addition of k-fold cross validation.

    Inputs:

        l_matrix:          The distance matrix(2D np array) of the bkg and sig events. Shape:(n, n)

        labels:            An array of labels for which type of event(bkg/sig) each row/column belongs to. Shape(n,)

        neighbor_list:     The list of neighbors wanted to be used for the kNN classification

        k_fold:            The number of folds wanted to be used for the k-fold cross validation

    Outputs:

        np.mean(auc_list): The mean of the AUCs computed during the k-fold cross validation

        np.std(auc_list):  The standard deviation of the AUCs computed during the k-fold cross validation

        np.mean(best_k_list): The mean of the best-k-neighbors computed during the k-fold cross validation

        np.std(best_k_list): The standard deviation of the best-k-neighbors computed during the k-fold cross validation

        metrics_dict:      A dictionary containing all the metrics computed during the k-fold cross validation.
    '''
    length = l_matrix.shape[0]
    folded_length = length//k_fold
    index_list = []

    for i in range(0, k_fold):
        index_list.append(list(range(i*folded_length, (i+1)*folded_length)))

    auc_list = []
    best_k_list = []
    metrics_dict = {}

    # perform k-fold cross validation
    for i in range(0, k_fold):
        new_index = index_list[i]
        for j in range(1,k_fold):
            new_index.extend(index_list[(i+j)%k_fold])
        l_matrix_new = l_matrix[new_index,:][:,new_index]
        labels_new = labels[new_index]
        best_auc, best_k, _, kNN_metrics = kNN_with_distance_matrix(l_matrix_new, labels_new, folded_length*(k_fold-2), folded_length, folded_length, neighbor_list, AUC_list=False)
        auc_list.append(best_auc)
        best_k_list.append(best_k)
        metrics_dict['repeat'+str(i)] = kNN_metrics

    return np.mean(auc_list), np.std(auc_list), np.mean(best_k_list), np.std(best_k_list), metrics_dict

# ##### `rNN_with_distance_matrix()`

def rNN_with_distance_matrix(l_matrix, labels, train_number, val_number, test_number, radius_list, AUC_list = False):
    '''
    Inputs:

        l_matrix:          The distance matrix(2D np array) of the two types of jets. Shape:(njets, njets)
                          (NOTE: This matrix must be symmetric and tracless for it to be a meaningful OT distance matrix)

        labels:           An array of labels for which type of particle each row/column belongs to. Shape(njets,)

        train_number:     The number of jets used for training the kNN model, set to be the first "train_number" of rows/cols

        val_number:       The number of jets used for the validation process(determining the best number of neighbors), set to be the "val_number" of rows/cols right after the training rows/cols

        test_number:      The number of jets used for the testing process, set to be the "test_number" of rows/cols right after the validation rows/cols

        radius_list:      The list of radii wanted to be used for the kNN classification

        AUC_list:         Default is False. When set to True, return the whole list of AUCs during the validation process

    Outputs:

        best_auc:         The AUC computed using the best-k-neighbor returned by the validation process.

        auc_list:         The entire list of AUC computed during the validation process. Only returned if AUC_list is set to be True

    '''

    l_matrix_train         = l_matrix[:train_number, :train_number]
    l_matrix_val_train     = l_matrix[train_number:train_number+val_number, :train_number]
    l_matrix_trainval      = l_matrix[:train_number+val_number, :train_number+val_number]
    l_matrix_test_trainval = l_matrix[train_number+val_number:test_number + train_number + val_number, :train_number+val_number]

    train_labels           = labels[:train_number]
    val_labels             = labels[train_number:train_number+val_number]
    trainval_labels        = labels[:train_number+val_number]
    test_labels            = labels[train_number+val_number:test_number+train_number+val_number]

    models, pred_vals = [], []

    for radius in tqdm(radius_list, desc='Fitting Models'):
        model = RadiusNeighborsClassifier(radius=radius, metric='precomputed')
        model.fit(l_matrix_train, train_labels)
        pred_val = model.predict_proba(l_matrix_val_train)[:,1]
        models.append(model)
        pred_vals.append(pred_val)

    auc_list = []

    for pred in pred_vals:
        auc = roc_auc_score(val_labels, pred)
        auc_list.append(auc)

    max_index = auc_list.index(max(auc_list))

    best_radius = radius_list[max_index]

    best_model = RadiusNeighborsClassifier(radius=best_radius, metric='precomputed')

    best_model.fit(l_matrix_trainval, trainval_labels)

    best_pred = best_model.predict_proba(l_matrix_test_trainval)[:,1]

    best_auc = roc_auc_score(test_labels, best_pred)

    if AUC_list == True:
        return best_auc, best_radius, best_model, auc_list

    else:
        return best_auc, best_radius, best_model

# ##### `rNN_cross_validation()`

def rNN_cross_validation(l_matrix, labels, radius_list, k_fold = 5):
    '''
    This function is basically a wrapper function around the rNN_with_distance_matrix function. It performs a k-fold cross validation
    for the rNN model and returns the mean and standard deviation of the AUC and the best radius.

    Inputs:

        l_matrix:          The distance matrix(2D np array) of bkg and sig events. Shape:(n, n)

        labels:            An array of labels for which type of event(bkg/sig) each row/column belongs to. Shape(n,)

        radius_list:       The list of radii wanted to be used for the rNN classification

        k_fold:            The number of folds wanted to be used for the cross validation process. Default is 5.

    Outputs:

        mean_auc:          The mean of the AUCs computed during the cross validation process

        std_auc:           The standard deviation of the AUCs computed during the cross validation process

        mean_best_radius:  The mean of the best radii computed during the cross validation process

        std_best_radius:   The standard deviation of the best radii computed during the cross validation process
    '''
    length = l_matrix.shape[0]
    folded_length = length//k_fold
    index_list = []

    for i in range(0, k_fold):
        index_list.append(list(range(i*folded_length, (i+1)*folded_length)))

    auc_list = []
    best_radius_list = []

    for i in range(0, k_fold):
        new_index = index_list[i]
        for j in range(1,k_fold):
            new_index.extend(index_list[(i+j)%k_fold])
        l_matrix_new = l_matrix[new_index,:][:,new_index]
        labels_new = labels[new_index]
        best_auc, best_radius, best_model = rNN_with_distance_matrix(l_matrix_new, labels_new, folded_length*(k_fold-2), folded_length, folded_length, radius_list, AUC_list=False)
        auc_list.append(best_auc)
        best_radius_list.append(best_radius)

    return np.mean(auc_list), np.std(auc_list), np.mean(best_radius_list), np.std(best_radius_list)

# ##### `SVM_with_distance_matrix()`

def SVM_with_distance_matrix(l_matrix, labels, train_number, val_number, test_number, gamma_list, C_list, kernel = 'rbf', AUC_list = False):
    '''
    Inputs:

        l_matrix:          The distance matrix(2D np array) of bkg and sig of events. Shape:(n, n)

        labels:            An array of labels for which type of event(bkg/sig) each row/column belongs to. Shape(n,)

        train_number:      The number of events used for training the SVM model, set to be the first "train_number" of rows/cols

        val_number:        The number of events used for the validation process(determining the best hyperparameters), set to be the "val_number" of rows/cols right after the training rows/cols

        test_number:       The number of events used for the testing process, set to be the "test_number" of rows/cols right after the validation rows/cols

        gamma_list:        The list of gamma values wanted to be used for the SVM classification

        C_list:            The list of C values wanted to be used for the SVM classification

        kernel:            The kernel used for the SVM classification. Default is 'rbf'

        AUC_list:          Default is False. When set to True, return the whole list of AUCs during the validation process

    Outputs:

        best_auc:          The AUC computed using the best hyperparameters returned by the validation process.

        best_gamma:        The best gamma hyperparameter returned by the validation process.

        best_C:            The best C hyperparameter returned by the validation process.

        best_model:        The best SVM model returned by the validation process.

        auc_list:          The entire list of AUC computed during the validation process. Only returned if AUC_list is set to be True
    '''
    models, pred_vals = [], []

    i = 0
    train_labels           = labels[:train_number]
    val_labels             = labels[train_number:train_number+val_number]
    trainval_labels        = labels[:train_number+val_number]
    test_labels            = labels[train_number+val_number:test_number+train_number+val_number]

    if kernel == 'rbf':
        for gamma in gamma_list:
        # for gamma in tqdm(gamma_list, desc='Gamma Loop'):
            l_matrix_gamma = np.exp(-gamma*l_matrix)

            l_matrix_train         = l_matrix_gamma[:train_number, :train_number]
            l_matrix_val_train     = l_matrix_gamma[train_number:train_number+val_number, :train_number]
            l_matrix_trainval      = l_matrix_gamma[:train_number+val_number, :train_number+val_number]
            l_matrix_test_trainval = l_matrix_gamma[train_number+val_number:test_number + train_number + val_number, :train_number+val_number]
            models.append([])
            pred_vals.append([])

            for C in C_list:
                model = SVC(gamma=gamma, C=C, kernel='precomputed')
                model.fit(l_matrix_train, train_labels)
                pred_val = model.predict(l_matrix_val_train)
                models[i].append(model)
                pred_vals[i].append(pred_val)

            i += 1

    else:
        l_matrix_train         = l_matrix[:train_number, :train_number]
        l_matrix_val_train     = l_matrix[train_number:train_number+val_number, :train_number]
        l_matrix_trainval      = l_matrix[:train_number+val_number, :train_number+val_number]
        l_matrix_test_trainval = l_matrix[train_number+val_number:test_number + train_number + val_number, :train_number+val_number]

        train_labels           = labels[:train_number]
        val_labels             = labels[train_number:train_number+val_number]
        trainval_labels        = labels[:train_number+val_number]
        test_labels            = labels[train_number+val_number:test_number+train_number+val_number]

    auc_list = []

    i = 0

    for pred_row in pred_vals:
        auc_list.append([])
        for pred in pred_row:
            auc = roc_auc_score(val_labels, pred)
            auc_list[i].append(auc)
        i += 1

    auc_array = np.array(auc_list)
    max_index = np.unravel_index(auc_array.argmax(), auc_array.shape)

    best_gamma = gamma_list[max_index[0]]
    best_C = C_list[max_index[1]]

    l_matrix_gamma = np.exp(-best_gamma*l_matrix)

    l_matrix_train         = l_matrix_gamma[:train_number, :train_number]
    l_matrix_val_train     = l_matrix_gamma[train_number:train_number+val_number, :train_number]
    l_matrix_trainval      = l_matrix_gamma[:train_number+val_number, :train_number+val_number]
    l_matrix_test_trainval = l_matrix_gamma[train_number+val_number:test_number + train_number + val_number, :train_number+val_number]

    best_model = SVC(gamma=best_gamma, C = best_C, kernel='precomputed')

    best_model.fit(l_matrix_trainval, trainval_labels)

    best_pred = best_model.predict(l_matrix_test_trainval)

    best_auc = roc_auc_score(test_labels, best_pred)

    if AUC_list == True:
        return best_auc, best_gamma, best_C, best_model, auc_list

    else:
        return best_auc, best_gamma, best_C, best_model

# ##### `SVM_cross_validation()`

def SVM_cross_validation(l_matrix, labels, gamma_list, C_list, kernel = 'rbf', k_fold = 5):
    '''
    This function is basically a wrapper function around the SVM_with_distance_matrix function. It performs a k-fold cross validation
    for the SVM model and returns the mean and standard deviation of the AUC and the best hyperparameters.

    Inputs:
        l_matrix:          The distance matrix(2D np array) of the signal and bkg samples. Shape:(n, n)

        labels:            An array of labels for which type of event(bkg/sig) each row/column belongs to. Shape(n,)

        gamma_list:        The list of gamma values wanted to be used for the SVM classification

        C_list:            The list of C values wanted to be used for the SVM classification

        kernel:            The kernel wanted to be used for the SVM classification. Default is 'rbf'

        k_fold:            The number of folds wanted to be used for the cross validation. Default is 5

    Outputs:
        auc_list:          The list of AUCs computed during the cross validation process

        best_gamma_list:   The list of best gamma values computed during the cross validation process

        best_C_list:       The list of best C values computed during the cross validation process
    '''
    length = l_matrix.shape[0]
    folded_length = length//k_fold
    index_list = []

    for i in range(0, k_fold):
        index_list.append(list(range(i*folded_length, (i+1)*folded_length)))

    auc_list = []
    best_gamma_list = []
    best_C_list = []

    for i in range(0, k_fold):
        new_index = index_list[i]
        for j in range(1,k_fold):
            new_index.extend(index_list[(i+j)%k_fold])
        l_matrix_new = l_matrix[new_index,:][:,new_index]
        labels_new = labels[new_index]

        best_auc, best_gamma, best_C, best_model = SVM_with_distance_matrix(l_matrix_new, labels_new, folded_length*(k_fold-2), folded_length, folded_length, gamma_list, C_list, AUC_list=False)

        auc_list.append(best_auc)
        best_gamma_list.append(best_gamma)
        best_C_list.append(best_C)

    return auc_list, best_gamma_list, best_C_list

# ##### `OneClassSVM_with_distance_matrix()`

def OneClassSVM_with_distance_matrix(train_matrix, test_matrix, test_labels, gamma, nu):
    '''
    Inputs:

        train_matrix:     The train distance matrix(2D np array). Shape:(n, n)
                          (NOTE: This matrix must be symmetric and tracless for it to be a meaningful OT distance matrix)

        test_matrix:      The distance matrix(2D np array) of the two types of jets. Shape:(m, n)
                          (NOTE: This matrix is usually not symmetric and tracelsss)

        test_labels:      An array of binary labels for which type of particle each row/column belongs to. Shape(m,)

        gamma:            The gamma parameter for the RBF kernel

        nu:               The nu parameter for the OneClassSVM model

    Outputs:

        auc, f1_score, ROC_metrics: The performance metrics of the OneClassSVM model
    '''
    train_matrix_gamma = np.exp(-gamma*train_matrix)
    test_matrix_gamma = np.exp(-gamma*test_matrix)

    model = svm.OneClassSVM(nu = nu, kernel='precomputed')
    model.fit(train_matrix_gamma)

    pred = model.predict(test_matrix_gamma)

    ROC_metrics = kNN_ROC_metrics(test_labels, pred, Interpolate = True)

    auc = roc_auc_score(test_labels, pred)
    f1_score = metrics.f1_score(test_labels, pred)

    return auc, f1_score, ROC_metrics

# ##### `kNN_ROC_metrics()`

def kNN_ROC_metrics(labels, prediction_proba, SIreg=0.0001, Interpolate=False):
    '''
    Calculate ROC metrics for kNN classifier
    Inputs:
        labels:             An array of binary labels for background and signal.
        prediction_proba:   An array of probabilities for each row/column being a signal jet.
        SIreg:              The regularization parameter for the Significance Improvement (SI) curve. Default is 0.0001
        Interpolate:        Default is False. When set to True, interpolate the ROC curve to have 101 points.
    Outputs:
        auc, fpr, tpr, si, fprInv, f1_score: The AUC, FPR, TPR, SI, FPR^{-1}, and F1 score of the ROC curve.
    '''
    fpr_raw, tpr_raw, _ = roc_curve(labels, prediction_proba)

    if Interpolate:
        base_tpr = np.linspace(0, 1, 101) # 0.00, 0.01, ..., 1.0
        tpr = base_tpr
        fpr = np.interp(base_tpr, tpr_raw, fpr_raw)
    else:
        tpr = tpr_raw
        fpr = fpr_raw

    auc = roc_auc_score(labels, prediction_proba)

    fpr_sqrt = np.sqrt(fpr + SIreg)
    si = tpr/fpr_sqrt

    fpr_masked = ma.masked_where(fpr==0., fpr)
    fprInv = 1./fpr_masked
    
    convention_1 = np.isin(labels, [0,1]).all() # 0 for background, 1 for signal
    convention_2 = np.isin(labels, [-1,1]).all() # 1 for background, -1 for signal
    
    if convention_1:
          P   = np.count_nonzero(labels == 1) # Number of signal (labels==1)
          N   = np.count_nonzero(labels == 0) # Number of background (labels==0)
    elif convention_2:
          P   = np.count_nonzero(labels == -1) # Number of signal (labels==-1)
          N   = np.count_nonzero(labels == 1)  # Number of background (labels==1)
    else:
          print('Invalid label convention. Please use 0 for background, 1 for signal or 1 for background, -1 for signal.')
          return None
          
    tp  = P*tpr
    fp  = N*fpr
    fn  = P*(1-tpr) # P*fnr
    f1_score = (2*tp)/(2*tp + fp + fn)

    return auc, fpr, tpr, si, fprInv, f1_score


