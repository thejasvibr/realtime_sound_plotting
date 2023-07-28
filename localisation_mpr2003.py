# -*- coding: utf-8 -*-
"""
Mellen, Pachter, Raquet 2003: IEEE Trans. Aerospace and Electronic Systems
==========================================================================
This is also known as the spherical-intersection method (SX) - as first
proposed by Schau & Robinson 1987.

Also see Malanowski & Kulpa 2012 for a repeat of the formulation in Mellen, 
Pachter & Raquet 2003. 


Reference
---------
* Mellen, G., Pachter, M., & Raquet, J. (2003). Closed-form solution for deter-
  mining emitter location using time difference of arrival measurements. IEEE 
  Transactions on Aerospace and Electronic Systems, 39(3), 1056-1058.
* Malanowski, M., & Kulpa, K. (2012). Two methods for target localization
  in multistatic passive radar. IEEE transactions on Aerospace and Electroni-
  c Systems, 48(1), 572-580.
* Schau, H. C., & Robinson, A. Z. (1987). Passive source localization employing
 intersecting spherical surfaces from time-of-arrival differences. IEEE Transa-
 ctions on Acoustics, Speech, and Signal Processing, 35(8), 1223-1225.
"""
import numpy as np 
import scipy.spatial as spatial

def tristar_mellen_pachter(*args):
    '''Wrapper around mellen_pachter_raquet_2003
    which only outputs positive y axis sources.
    '''
    sources = mellen_pachter_raquet_2003(*args)
    #print('sources:', sources)
    infront_of_array = []
    if sources.size>3:
        for each in sources:
            x,y,z = each
            if y>= 0:
                infront_of_array.append(each)
    elif sources.size==3:
        x,y,z = sources
        if y>= 0:
            infront_of_array.append(sources)
    return infront_of_array

def mellen_pachter_raquet_2003(mic_array, di):
    '''
    A re-formulation of the spherical-intersection method originally proposed by 
    Schau & Robinson 1987. 
    
    Parameters
    ----------
    mic_array : (m,n) np.array
        M mic coordinates with N dimensions.
    di : (m-1,) np.array
        Range differences w.r.t. 0th sensor. 
    
    Returns 
    -------
    xs : (1,n) or (2,n) np.array
        If only one valid solution - then returns (1,n) np.array
        else returns both valid solutions as a (2,n) np.array. 
        
    Reference
    ---------
    * Mellen, G., Pachter, M., & Raquet, J. (2003). Closed-form solution for deter-
      mining emitter location using time difference of arrival measurements. IEEE 
      Transactions on Aerospace and Electronic Systems, 39(3), 1056-1058.
    '''
    m_mics, n_dim = mic_array.shape
    if not n_dim == 3:
        raise NotImplementedError(f'{n_dim} dimensions detected. This implementation only supports 3D arrays.')
    if m_mics <= n_dim:
        raise ValueError(f'Not enough mics. {m_mics} mics detected, but >= {n_dim+1} mics required.')
    
    # 0th sensor is assumed to be at origin
    S = mic_array[1:,:] - mic_array[0,:] # eqn. 11
    # eqn. 12
    z = np.sum(S**2,1) - di**2
    z *= 0.5 
    # eqn. 17 - without the weighting matrix R
    inv_StS = np.linalg.inv(np.dot(S.T,S))
    inv_StS_St = np.dot(inv_StS, S.T)
    a = np.dot(inv_StS_St, z)
    # eqn. 18
    b = np.dot(inv_StS_St, di)    
    # eqn. 22
    Rs_12= solve_eqn_22(a, b)
    # substitute Rs into eqn. 19
    xs = choose_correct_mpr_solutions(mic_array, Rs_12, (a,b), di)
    return xs

def choose_correct_mpr_solutions(mic_array, Rs_12, a_b, obs_di):
    '''
    The logic behind choice of source solutions is described in 
    Section C of Malanowski & Kulpa (2012). 

    Parameters
    ----------
    mic_array : (m,n) np.array
        M mics with N dimensions.
    Rs_12 : (2,) np.array
        The two estimated range values
    a_b : tuple with two (3,) np.arrays 
        (a, b) -  see mpr_2003
    obs_di : (m-1,) np.array
        M-1 np.array with range differences w.r.t 0th sensor. 
    
    Returns 
    -------
    xs : (0,), (3,) or (2,3) np.array
        Produces a single source estimate when there's only one positive Rs.
        In case of two positive Rs, then outputs both valid solutions.  
    
    References
    ----------
    * Malanowski, M., & Kulpa, K. (2012). Two methods for target localization
      in multistatic passive radar. IEEE transactions on Aerospace and Electroni-
      c Systems, 48(1), 572-580.
    '''
    num_Rs_postive = np.sum(Rs_12>0)
    a, b = a_b

    if num_Rs_postive==2:
        Rs1, Rs2  = Rs_12
        
        x_candidates = np.array([[a - b*Rs1],
                                 [a - b*Rs2]]).reshape(2,-1)
        xs = x_candidates + mic_array[0,:]
    elif num_Rs_postive==1:
        # find which one is positive
        positive_ind = int(np.argmax(Rs_12))
        xs = a-b*Rs_12[positive_ind]
        xs += mic_array[0,:]
    else:
        xs = np.array([])

    return xs

def solve_eqn_22(a,b):
    '''
    Implements Equation 22 in MPR 2003. 

    Parameters
    ----------
    a,b : (3,) np.arrays
    
    Returns
    -------
    Rs12 : (2,) np.array
        Array holding Rs1 and Rs2 - the two range estimates arising from 
        the quadratic equation.
    '''
    a1, a2, a3 = a
    b1, b2, b3 = b
    term1 = a1*b1 + a2*b2 + a3*b3
    # split the numerator
    term2_i = term1**2
    bsquare_term = (b1**2+ b2**2 + b3**2) - 1 

    term2_ii = bsquare_term*(a1**2 + a2**2 + a3**2)
    term2 = np.sqrt(term2_i - term2_ii)
    denominator = bsquare_term
    numerator1 = term1 + term2
    numerator2 = term1 - term2
    Rs12 = np.array([numerator1/denominator, numerator2/denominator])
    return Rs12

if __name__ == "__main__":
    np.random.seed(82319)
    source = np.array([-0.5,3,4])
    ndim = source.size
    num_mics = 4
    mic_array = np.array([[0,0,1],
                          [0,1,0],
                          [1,0,0],
                          [1,1.2,0]])
    mic_source_dist = lambda X,Y : spatial.distance.euclidean(X,Y)
    mic_2_source = np.apply_along_axis(mic_source_dist, 1, mic_array, source)
    xs = source - mic_array[0,:]
    di = mic_2_source[1:] - mic_2_source[0] #+ np.random.normal(0,1e-1,num_mics-1)
    print(di)
    print('what')
    out = mellen_pachter_raquet_2003(mic_array, di)
    print(out)
