# Continuous Algebraic Riccati Equation Solver(s)
#
# Author:Jeffrey Armstrong <jeff@rainbow-100.com>
#
import numpy
import numpy.linalg
import scipy.linalg
from scipy.signal.lyap import lyap_bartels_stewart
from scipy.optimize import fminbound

EPSILON = 1.0E-5
ITER_LIMIT = 10000

def continous_riccati_equation(*args):
    """Computes the value of the Ricatti equation:
    
        R(X) = A'X + XA - XBR^-1B'X+Q
        
        or
        
        R(X) = A'X + XA - XGX+Q
        
    Depending on the number of inputs, the proper equation will be solved.

    Syntax is either:
    
        continous_riccati_equation(a,b,q,r,x)
        
        or
        
        continous_riccati_equation(a,q,g,x)
        
    """
    
    if len(args) == 4:
        a = args[0]
        q = args[1]
        g = args[2]
        x = args[3]
    elif len(args) == 5:
        a = args[0]
        b = args[1]
        q = args[2]
        r = args[3]
        x = args[4]
        
        g = b*numpy.linalg.inv(r)*b.transpose()
    else:
        raise ArgumentError("Must pass in either (a,q,g,x) or (a,b,q,r,x)")
        
    return a.transpose()*x + x*a - x*g*x + q

class CareSolver:
    """Class providing techniques for solving the continuous-time
    algebraic Riccati equation (CARE) (A'X + XA - XBR^-1B'X+Q=0).  
	One technique utilizes a direct solution, while the second technique
    uses a Newton iterative technique.
    
    Direct solution algorithm taken from:
    Laub, "A Schur Method for Solving Algebraic Riccati Equations."
    U.S. Energy Research and Development Agency under contract 
    ERDA-E(49-18)-2087.
    
    Iterative solution algorithm taken from:
    Guo and Laub, "On A Newton-like Method for Solving Algebraic Riccati
    Equations," Siam Journal on Matrix Analysis and Applications, Vol. 21
    Number 2, 2000.

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
        
        self.g = None
        self.x = None
        
        self.nk = None
        
        self.compute_minimizer = False
        self.iterative = False
        self.iterations = 0

    def compute_g(self):
        """Computes the value b*inv(r)*b', which is needed throughout the 
        solvers."""
        
        try:
            self.g = self.b*numpy.linalg.inv(self.r)*self.b.transpose()
        except numpy.linalg.LinAlgError:
            raise ValueError('R matrix in the Riccati equation solver is ill-conditioned')
        
        return self.g
        
    def solve_direct(self):
        """Solves the DARE equation directly using a Schur decomposition method.
        This routine is prone to numerical instabilities mostly associated with
        the inversion of the R matrix.  However, in some well-defined cases, the
        algortihm may work properly and provide a considerable computational
        speed advantages over the iterative techniques.
        """
        
        g = self.compute_g()
        
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
        self.x = self.solution
        return self.solution
        
    def newton_iterative_init(self):
        """Initializes the Newton iterative solver.  The intialization procedure
        computes a matrix X0 such that A-DX0 is stable using the algorithm from:
        
            V. Sima, "An efficient Schur method to solve the stabilizing 
            problem," IEEE Trans. Automatic Control, 26 (1981), pp. 724–725.
        """
        
        # Compute the norm of the A matrix, and choose a beta that is half that
        beta = numpy.linalg.norm(self.a,'fro')/2.0
        
        # Perform a Schur decomp of A
        [anew,u] = scipy.linalg.schur(self.a,sort='lhp')
        
        # Perform a Schur decomp of "D" (actually self.g in this implementation)
        d = self.compute_g()
        dnew = u.transpose()*d
        c = dnew*dnew.transpose()
        
        # Solve the Lyapunov equation (anew+beta*i)z + z(anew+beta*i)' = 2c
        la = anew+beta*numpy.eye(anew.shape[0])
        z = lyap_bartels_stewart(la,2.0*c)
        
        # Solve for a intermediate "x"
        try:
            x_int = dnew*numpy.linalg.inv(z)
            
            # Initial guess is:
            self.x = x_int.transpose()*u.transpose()
            
        except numpy.linalg.LinAlgError:
        
            # In case of an error due to the inversion of z, 
            # initialize to the identity matrix
            self.x = numpy.eye(self.a.shape[1])
        
    def evaluate_ricatti_equation(self):
        """Evaluates the Riccati equation A'X + XA - XBR^-1B'X+Q= Ric(X) using
        the internal value self.x"""
        if self.g is None:
            self.compute_g()
        return continous_riccati_equation(self.a,self.q,self.g,self.x)

    def _tminimizer_cost(self,tmin):
        """Cost function for computing an optimal Newton iteration step
        relaxation factor"""
        
        xpass = self.x + tmin*self.nk
        ric = continous_riccati_equation(self.a,self.q,self.g,xpass)
        return (numpy.linalg.norm(ric,'fro'))**2.0

    def newton_iterative_step(self):
        """Performs a single Newton iterative step"""
    
        ric = self.evaluate_ricatti_equation()
        
        la = self.a-self.g*self.x
        self.nk = lyap_bartels_stewart(la.transpose(),ric)
        
        tminimizer = 1.0
        
        # If the relaxation factor (minimizer) computation has been explicitly
        # allowed, perform a bounded optimization to find the value.
        if self.compute_minimizer:
            tminimizer = fminbound(self._tminimizer_cost,0.0,2.0,disp=0)
        
        self.x = self.x + tminimizer*self.nk
        
    def solve_newton_iterative(self,eps,iter_limit,initial=None):
        """Solves the CARE using a Newton iterative technique."""
        if initial==None:
            self.newton_iterative_init()
        else:
            self.x = initial
            
        error = 1.0E+6
        self.iterations = 0
        while error > eps and self.iterations < iter_limit:
            xlast = self.x
            self.newton_iterative_step()
            #error = abs((self.x - xlast).max()) #numpy.linalg.norm(self.dx)
            error = numpy.linalg.norm(self.evaluate_ricatti_equation())
            self.iterations = self.iterations + 1
        
        self.solution = self.x
        return self.x
        
    def solve(self,eps=EPSILON,iter_limit=ITER_LIMIT,initial=None):
        """Solves the continuous-time Riccati equation:
        
           (A'X + XA - XBR^-1B'X+Q=0)
        
        returning a value or estimate of the X matrix.
        
        Sovler object flags:
        
        iterative   Solver Method
        False       Direct (solve_direct)
        True        Newton iterative (solve_newton_iterative)
        """
        
        if self.iterative:
            self.solution = self.solve_newton_iterative(eps,iter_limit,initial)
            
        else:
            self.solution = self.solve_direct()
            self.iterations = None
        
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
    cs.iterative = iterative
    
    return cs.solve()