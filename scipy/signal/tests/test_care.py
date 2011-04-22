from scipy.signal.care import CareSolver
import numpy
import unittest

class CareTestCase(numpy.testing.TestCase):

    def test_direct(self):
        """Tests the direct CARE solver using example problems from:
        
            Laub, "A Schur Method for Solving Algebraic Riccati Equations."
            U.S. Energy Research and Development Agency under contract 
            ERDA-E(49-18)-2087.
        """
            
        a = numpy.matrix([[0.0,1.0],[0.0,0.0]])
        b = numpy.matrix([[0.0,],[1.0,]])
        r = numpy.matrix([[1.0,],])
        q = numpy.matrix([[1.0,0.0],[0.0,2.0]])
        
        cs = CareSolver(a,b,q,r)
        x = cs.solve_direct()
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[2.0,1.0], \
                          [1.0,2.0]]), \
            3)
    
        # This test example requires careful Schur decomposition
        a = numpy.matrix([[4.0,3.0],[-9.0/2.0, -7.0/2.0]])
        b = numpy.matrix([[1.0,],[-1.0,]])
        q = numpy.matrix([[9.0,6.0],[6.0,4.0]])
        r = numpy.matrix([[1.0,],])
        
        cs = CareSolver(a,b,q,r)
        x = cs.solve_direct()
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[21.7279,14.4853], \
                          [14.4853, 9.6569]]), \
            3)
