"""
dltisys - Code related to discrete linear time-invariant systems
"""

# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# April 4, 2011

import math
import numpy
from scipy.interpolate import interp1d
from scipy.signal import tf2ss

def dlsim(system, u, t=None, x0=None):
    """
    Simulate output of a discrete-time linear system.

    Parameters
    ----------
    system : an instance of the LTI class or a tuple describing the system.
        The following gives the number of elements in the tuple and
        the interpretation:

        * 3: (num, den, dt)
        * 5: (A, B, C, D, dt)

    U : array_like
        An input array describing the input at each time `T`
        (interpolation is assumed between given times).  If there are
        multiple inputs, then each column of the rank-2 array
        represents an input.
    T : array_like
        The time steps at which the input is defined and at which the
        output is desired.
    X0 :
        The initial conditions on the state vector (zero by default).

    Returns
    -------
    tout : 1D ndarray
        Time values for the output.
    yout : 1D ndarray
        System response.
    xout : ndarray
        Time-evolution of the state-vector.  Only generated if the 
        input is a state-space systems.

    """

    if len(system) == 3:
        a,b,c,d = tf2ss(system[0],system[1])
        dt = system[2]
    elif len(system) == 5:
        a,b,c,d,dt = system
    else:
        raise ValueError("System argument should be a discrete transfer function or state-space system")
    
    if t is None:
        out_samples = max(u.shape)
        stoptime = (out_samples-1)*dt
    else:
        stoptime = t[-1]
        out_samples = int(math.floor(stoptime/dt))+1
    
    # Pre-build output arrays
    xout = numpy.zeros((out_samples,a.shape[0]))
    yout = numpy.zeros((out_samples,c.shape[0]))
    tout = numpy.linspace(0.0,stoptime,num=out_samples)
    
    # Check initial condition
    if x0 is None:
        xout[0,:] = numpy.zeros((a.shape[1],))
    else:
        xout[0,:] = x0
    
    # Pre-interpolate inputs into the desired time steps
    if t is None:
        u_dt = u
    else:
        u_dt_interp = interp1d(t,u.transpose(),copy=False,bounds_error=True)
        u_dt = u_dt_interp(tout)
        u_dt = u_dt.transpose()
    
    # Simulate the system
    for i in range(0,out_samples-1):
        
        xout[i+1,:] = numpy.dot(a,xout[i,:]) + numpy.dot(b,u_dt[i,:])
        yout[i,:] = numpy.dot(c,xout[i,:]) + numpy.dot(d,u_dt[i,:])
        
    # Last point
    yout[out_samples-1,:] = numpy.dot(c,xout[out_samples-1,:]) + numpy.dot(d,u_dt[out_samples-1,:])
    
    if len(system) == 3:
        return tout,yout
    elif len(system) == 5:
        return tout,yout,xout
    