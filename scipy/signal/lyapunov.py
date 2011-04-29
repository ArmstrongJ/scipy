"""
lyapunov - Routines to support the solution of continuous and discrete
Lyapunov and Sylvester equations
"""

# Continuous Lyapunov Equation Solver(s)
#
# Author:Jeffrey Armstrong <jeff@rainbow-100.com>

import numpy
import scipy.linalg
import numpy.linalg
import warnings
import math

from scipy.linalg.lapack import get_lapack_funcs

def sylvester_bartels_stewart(a,b,q):
    """Computes a solution to the Sylvester equation (AX + XB = Q) using the
    Bartels-Stewart algorithm.  The A and B matrices first undergo Schur
    decompostions.  The resulting matrices are used to construct an alternative
    Sylvester equation (RY + YS^T = F) where the R and S matrices are in quasi-
    triangular form.  The simplified equation is then solved using DTRSYL from
    LAPACK directly."""
    
    # Compute the Schur decomp form of a
    r,u = scipy.linalg.schur(a,output='real')
    
    # Compute the Schur decomp of b (unnecessary)
    s,v = scipy.linalg.schur(b.transpose(),output='real')
    
    # Construct f
    f = u.transpose()*numpy.asmatrix(q)*v
    
    # Call the Sylvester equation solver
    trsyl, = get_lapack_funcs(('trsyl',), (r,s,f))
    if trsyl == None:
        raise RuntimeError('LAPACK implementation does not contain a Sylvester equation solver (TRSYL)')    
    y,info = trsyl(1,r,s,f,1.0,trana=0,tranb=1)
    y = numpy.asmatrix(y)
    
    return -u*y*v.transpose()

def lyapunov_bartels_stewart(a,q):
    try:
        at = a.getH()
    except AttributeError:
        at = numpy.asmatrix(a).getH()
    return sylvester_bartels_stewart(a,at,q)

def lyapunov(a,q):
    """Solves the continuous Lyapunov equation (AX + XA^H = Q) given the values
    of A and Q.  This function provides a generalized interface to two
    available solvers. A Python implementation of the Bartels-Stewart algorithm
    is used.
    
    Parameters
    ----------
    a : array_like
        A square matrix
        
    q : array_like
    
    Returns
    -------
    x : array_like
        Solution to the continuous Lyapunov equation
    """
    
    return lyapunov_bartels_stewart(a,q)
        
def sylvester(a,b,q):
    """Solves a general Sylvester equation (AX + XB = Q) via the Bartels-Stewart
    algorithm.  While the sizes of a, b, and q must be compatible, there are no 
    restrictions on the mathematical form of a and b.
    
    Parameters
    ----------
    a : array_like
    
    b : array_like
        
    q : array_like
    
    Returns
    -------
    x : array_like
        Solution to the Sylvester equation
    """
    
    return sylvester_bartels_stewart(a,b,q)
    
    
# Discrete Lyapunov Equation Solver(s)
#
# Author:Jeffrey Armstrong <jeff@rainbow-100.com>
#
# Based on the pydare package by the same author
# http://code.google.com/p/pydare/
# and appropriately relicensed.  Contains no GPL code.
#

ITER_LIMIT = 10000
LYAPUNOV_EPSILON = 1.0E-6

def dlyapunov_iterative(a,q,eps=LYAPUNOV_EPSILON,iter_limit=ITER_LIMIT):
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
    
def dlyapunov_via_sylvester(a,q):
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

def dlyapunov(a,q,iterative=False,iteration_limit=ITER_LIMIT):
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
        return dlyapunov_iterative(a,q,iter_limit=iteration_limit)
    else:
        return dlyapunov_via_sylvester(a,q)
        