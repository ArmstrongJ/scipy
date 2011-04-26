"""
lyap - Routines to support the solution of continuous Lyapunov and Sylvester
equations
"""

# Continuous Lyapunov Equation Solver(s)
#
# Author:Jeffrey Armstrong <jeff@rainbow-100.com>

import numpy
import scipy.linalg

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

def lyap_bartels_stewart(a,q):
    try:
        at = a.getH()
    except AttributeError:
        at = numpy.asmatrix(a).getH()
    return sylvester_bartels_stewart(a,at,q)

def lyap(a,q):
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
    
    return lyap_bartels_stewart(a,q)
        
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
    
    