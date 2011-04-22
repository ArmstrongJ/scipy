# Continuous Algebraic Riccati Equation Solver(s)
#
# Author:Jeffrey Armstrong <jeff@rainbow-100.com>
#

import numpy
import numpy.linalg
import scipy.linalg

class CareSolver:
    """Class providing techniques for solving the continuous-time
    algebraic Riccati equation (CARE) (A'X + XA - XBR^-1B'X+Q=0).  
	One technique utilizes a direct solution, but it is prone to the 
    numerical condition of the R input.  
    
    Direct solution algorithm taken from:
    Laub, "A Schur Method for Solving Algebraic Riccati Equations."
    U.S. Energy Research and Development Agency under contract 
    ERDA-E(49-18)-2087.

    Author:
    Jeffrey Armstrong <jeff@rainbow-100.com>
    """

    """F'X + XF - XGX + H = 0 or """

    def __init__(self,a=None,b=None,q=None,r=None):
        """Initializes the CARE (A'X + XA - XBR^-1B'X+Q=0) solver using the
        specified inputs of the A, B, Q, and R matrices.  
		
		All inputs must be numpy matrix objects.  Any other input types may 
        lead to cryptic error messages.""" 
        
        self.a = a
        self.b = b
        self.q = q
        self.r = r
        
    def solve_direct(self):
        """Solves the DARE equation directly using a Schur decomposition method.
        This routine is prone to numerical instabilities mostly associated with
        the inversion of the R matrix.  However, in some well-defined cases, the
        algortihm may work properly and provide a considerable computational
        speed advantages over the iterative techniques.
        """
        
        g = self.b*numpy.linalg.inv(self.r)*self.b.transpose()
        
        z11 = self.a
        z12 = -1.0*g
        z21 = -1.0*self.q
        z22 = -1.0*self.a.transpose()
        
        z = numpy.vstack((numpy.hstack((z11,z12)),numpy.hstack((z21,z22))))
        
        [s,u] = scipy.linalg.schur(z,sort='lhp')

        (m,n) = u.shape
        
        u11 = u[0:m/2,0:n/2]
        u12 = u[0:m/2,n/2:n]
        u21 = u[m/2:m,0:n/2]
        u22 = u[m/2:m,n/2:n]
        u11i = numpy.linalg.inv(u11)

        self.solution = numpy.asmatrix(u21)*numpy.asmatrix(u11i)
        return self.solution
        
def care(a,b,q,r,iterative=False):
    """Solves the continuous-time Riccati equation:
    
       A'X + XA - XBR^-1B'X+Q=0
    
    returning a value or estimate of the X matrix.
    
    This function is provided for convenience; it provides a simple wrapper
    around the CareSolver class.
    
    Parameters
    -----------
    a,b,q,r : array-like
        Arrays representing matrices of the continuous algebraic Riccati 
        equation
        
    Returns
    -------
    x : ndarray
        The solution to the continuous algebraic Riccati equation
    
    See Also
    --------
    scipy.signal.CareSolver
    scipy.signal.dare
    
    """
    
    cs = CareSolver(a,b,q,r)
    
    return cs.solve()