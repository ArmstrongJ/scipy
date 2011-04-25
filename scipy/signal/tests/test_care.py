from scipy.signal.care import CareSolver, continous_riccati_equation, care
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
            
    def test_iterative(self):
        a = numpy.matrix([[0.0,1.0],[0.0,0.0]])
        b = numpy.matrix([[0.0,],[1.0,]])
        r = numpy.matrix([[1.0,],])
        q = numpy.matrix([[1.0,0.0],[0.0,2.0]])
        
        cs = CareSolver(a,b,q,r)
        cs.iterative = True
        cs.compute_minimizer = True
        x = cs.solve()
        
        numpy.testing.assert_array_almost_equal(abs(x), \
            numpy.matrix([[2.0,1.0], \
                          [1.0,2.0]]), \
            3)
            
        # This test example requires careful Schur decomposition
        a = numpy.matrix([[4.0,3.0],[-9.0/2.0, -7.0/2.0]])
        b = numpy.matrix([[1.0,],[-1.0,]])
        q = numpy.matrix([[9.0,6.0],[6.0,4.0]])
        r = numpy.matrix([[1.0,],])
        
        cs = CareSolver(a,b,q,r)
        cs.iterative = True
        cs.compute_minimizer = True
        x = cs.solve()
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[21.7279,14.4853], \
                          [14.4853, 9.6569]]), \
            3)
            
        # Make sure the convenience function works as expected
        x = care(a,b,q,r)
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[21.7279,14.4853], \
                          [14.4853, 9.6569]]), \
            3)

    def test_continous_riccati_equation(self):
    
        a = numpy.matrix([[4.0,3.0],[-9.0/2.0, -7.0/2.0]])
        b = numpy.matrix([[1.0,],[-1.0,]])
        q = numpy.matrix([[9.0,6.0],[6.0,4.0]])
        r = numpy.matrix([[1.0,],])
        x =  numpy.matrix([[21.7279,14.4853], [14.4853, 9.6569]])
        
        ric = continous_riccati_equation(a,b,q,r,x)
        
        numpy.testing.assert_array_almost_equal(numpy.zeros((2,2)),ric,3)
        
        g = b*numpy.linalg.inv(r)*b.transpose()
        
        ric = continous_riccati_equation(a,q,g,x)
        
        numpy.testing.assert_array_almost_equal(numpy.zeros((2,2)),ric,3)
