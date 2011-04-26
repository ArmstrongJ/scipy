# Discrete Lyapunov Equation Solver(s)
#
# Author:Jeffrey Armstrong <jeff@rainbow-100.com>
#
# Based on the pydare package by the same author
# http://code.google.com/p/pydare/
# and appropriately relicensed.  Contains no GPL code.
#

import numpy
import numpy.linalg
import scipy.linalg
import warnings
import math

from scipy.signal.lyap import sylvester_bartels_stewart

ITER_LIMIT = 10000
LYAPUNOV_EPSILON = 1.0E-6

def dlyap_iterative(a,q,eps=LYAPUNOV_EPSILON,iter_limit=ITER_LIMIT):
    """Solves the Discrete Lyapunov Equation (X = A X A' + Q) via an iterative
    method.  The routine returns an estimate of X based on the A and Q input
    matrices.  
    
    This iterative solver requires that the eigenvalues of the square matrix
    A be within the unit circle for convergence reasons.
    
    Iterative Discrete-Time Lyapunov Solver based on:
     
    Davinson and Man, "The Numerical Solution of A'Q+QA=-C." 
    IEEE Transactions on Automatic Control, Volume 13, Issue 4, August, 1968.  p. 448.
    
    Parameters
    -----------
    a,q : ndarray
        Arrays representing matrices of the discrete Lyapunov equation
    
    eps : float
        Value to use for convergence conditions.  Defaults to 
        scipy.signal.dlyap.LYAPUNOV_EPSILON
    
    iteration_limit : integer
        Specifies the maximum number of iterations to use when employing the
        iterative solver.  Defaults to scipy.signal.dlyap.ITER_LIMIT
    
    Returns
    -------
    x : ndarray
        The solution to the discrete Lyapunov equation
    """
    error = 1E+6
        
    x = q
    ap = a
    apt = a.transpose()
    last_change = 1.0E+10
    count = 1

    (m,n) = a.shape
    if m != n:
        raise ValueError("input 'a' must be square") 
    
    det_a = numpy.linalg.det(a)
    if det_a > 1.0:
        raise ValueError("input 'a' must have eigenvalues within the unit circle") 
    
    while error > LYAPUNOV_EPSILON and count < iter_limit:
        change = ap*q*apt
            
        x = x + change
        
        #if numpy.linalg.norm(change) > last_change:
        #    raise ValueError('A is not convergent')
        #last_change = abs(change.max())#numpy.linalg.norm(change)
        ap = ap*a
        apt = apt*(a.transpose())
        error = abs(change.max())
        count = count + 1

    if count >= iter_limit:
        warnings.warn('lyap_solve: iteration limit reached - no convergence',RuntimeWarning)
        #print 'warning: lyap_solve: iteration limit reached - no convergence'
        
    return x
    
def dlyap_via_sylvester(a,q):
    """Computes the solution to the discrete Lyapunov equation (AXA' + X = Q) by
    transforming the equation into a form that can utilize a Sylvester equation
    solver.  Specifically, the equation is converted to AX + X(A')^-1 = Q(A')^-1
    for solving.  The A matrix must be invertible for this method to work.
    
    Parameters
    -----------
    a,q : ndarray
        Arrays representing matrices of the discrete Lyapunov equation
        
    Returns
    -------
    x : ndarray
        The solution to the discrete Lyapunov equation
    """

    # Compute the inverse of the transpose of a
    a_t_i = numpy.linalg.inv(a.transpose())
    
    # Create a new solution
    f = q*a_t_i

    # Using the inverse of a, solve the generalized Sylvester
    # equation to retrieve a solution to the discrete
    # Lyapunov equation
    return sylvester_bartels_stewart(a,-1.0*a_t_i,f)

def dlyap(a,q,iterative=False,iteration_limit=ITER_LIMIT):
    """Solves the discrete Lyapunov equation (X = A X A' + Q) given the values
    of A and Q.  This function provides a generalized interface to three
    available solvers.  If the iterative flag is not set, the routine will fall
    back to a direct solver.  
    
    Parameters
    -----------
    a,q : ndarray
        Arrays representing matrices of the discrete Lyapunov equation
        
    iterative : boolean
        True to use an iterative solver, False to solve directly.  Defaults to
        False
    
    iteration_limit : integer
        Specifies the maximum number of iterations to use when employing the
        iterative solver.  Defaults to scipy.signal.dlyap.ITER_LIMIT
    
    Returns
    -------
    x : ndarray
        The solution to the discrete Lyapunov equation
    """
    
    if iterative:
        return dlyap_iterative(a,q,iter_limit=iteration_limit)
    else:
        return dlyap_via_sylvester(a,q)
        