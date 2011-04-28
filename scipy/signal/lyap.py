"""
lyap - Routines to support the solution of continuous Lyapunov and Sylvester
equations
"""

# Continuous Lyapunov Equation Solver(s)
#
# Author:Jeffrey Armstrong <jeff@rainbow-100.com>

import numpy
import scipy.linalg

try:
    import scipy.linalg.flapack
except ImportError:
    pass

try:
    import slycot
except ImportError:
    pass

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
    y,info = scipy.linalg.flapack.dtrsyl(1,r,s,f,1.0,trana=0,tranb=1)
    y = numpy.asmatrix(y)
    
    return -u*y*v.transpose()

def lyap_bartels_stewart(a,q):
    
    return sylvester_bartels_stewart(a,a.getH(),q)

def lyap_slycot(a,q): 
    """Solves the continous Lyapunov using the SLICOT library's implementation
    if available.  The routine attempts to call SB03MD to solve the discrete
    equation.  If a NameError is thrown, meaning SLICOT is not available,
    an appropriate RuntimeError is raised.
    
    More on SLICOT: http://www.slicot.org/
    
    Python Interface (Slycot): https://github.com/avventi/Slycot
    """

    x = None
    
    (m,n) = a.shape
    if m != n:
        raise ValueError("input 'a' must be square") 
    
    try:
        x,scale,sep,ferr,w = slycot.sb03md(n, -q, a, numpy.eye(n), 'C', trana='T')
    except NameError:
        raise RuntimeError('SLICOT not available')
    
    return x
    
def lyap(a,q):
    """Solves the continuous Lyapunov equation (AX + XA^H = Q) given the values
    of A and Q.  This function provides a generalized interface to two
    available solvers. If the Python interface to SLICOT is installed, 
    the routine will preferentially call the SLICOT solver; otherwise, a Python 
    implementation of the Bartels-Stewart algorithm is used.
    
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
    
    try:
        return lyap_slycot(a,q)
    except RuntimeError:
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
    
    