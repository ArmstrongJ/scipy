
# Author: Jeffrey Armstrong <jeff@approximatrix.com>
# April 4, 2011

import numpy
import numpy.testing 
from scipy.signal import dlsim

DLSIM_A = numpy.asarray([[0.9,0.1],[-0.2,0.9]])
DLSIM_B = numpy.asarray([[0.4,0.1,-0.1],[0.0,0.05,0.0]])
DLSIM_C = numpy.asarray([[0.1,0.3],])
DLSIM_D = numpy.asarray([[0.0,-0.1,0.0],])
DLSIM_DT = 0.5

DLSIM_U = numpy.hstack( ( numpy.asmatrix(numpy.linspace(0,4.0,num=5)).transpose(),0.01*numpy.ones((5,1)),-0.002*numpy.ones((5,1)) ) )

DLSIM_YOUT = numpy.asmatrix([-0.001,-0.00073,0.039446,0.0915387,0.13195948]).transpose()
DLSIM_XOUT = numpy.asarray([[0,0],[0.0012,0.0005],[0.40233,0.00071],[1.163368,-0.079327,],[2.2402985,-0.3035679]])

DLSIM_NUM = numpy.asarray([1.0,-0.1])
DLSIM_DEN = numpy.asarray([0.3,1.0,0.2])
DLSIM_TFYOUT = numpy.asmatrix([0.0,0.0,3.33333333333333,-4.77777777777778,23.0370370370370]).transpose()

class TestDLTI(numpy.testing.TestCase):
    
    def test_dlsim(self):
    
        t_in = numpy.linspace(0,2.0,num=5)
        tout,yout,xout = dlsim((DLSIM_A,DLSIM_B,DLSIM_C,DLSIM_D,DLSIM_DT), DLSIM_U, t_in) 

        numpy.testing.assert_array_almost_equal(DLSIM_YOUT,yout)
        numpy.testing.assert_array_almost_equal(DLSIM_XOUT,xout)
        numpy.testing.assert_array_almost_equal(t_in,tout)
        
        # Interpolated control
        u_sparse = DLSIM_U[[0,4],:]
        t_sparse = numpy.asarray([0.0,2.0])
        
        tout,yout,xout = dlsim((DLSIM_A,DLSIM_B,DLSIM_C,DLSIM_D,DLSIM_DT), DLSIM_U, t_in) 

        numpy.testing.assert_array_almost_equal(DLSIM_YOUT,yout)
        numpy.testing.assert_array_almost_equal(DLSIM_XOUT,xout)
        numpy.testing.assert_array_almost_equal(t_in,tout)
        
        # Transfer functions
        tout,yout = dlsim((DLSIM_NUM,DLSIM_DEN,DLSIM_DT), DLSIM_U[:,0], t_in)
        numpy.testing.assert_array_almost_equal(DLSIM_TFYOUT,yout)
        numpy.testing.assert_array_almost_equal(t_in,tout)
