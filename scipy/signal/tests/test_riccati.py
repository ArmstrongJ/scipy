from scipy.signal.riccati import CareSolver, continous_riccati_equation, care, DareSolver, dare
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

class DareTestCase(numpy.testing.TestCase):
    
    def test_iterative(self):
        a = numpy.matrix([[0.0,0.1,0.0],\
                          [0.0,0.0,0.1],\
                          [0.0,0.0,0.0]])
                           
        b = numpy.matrix([[1.0,0.0], \
                          [0.0,0.0], \
                          [0.0,1.0]])
                           
        r = numpy.matrix([[0.0,0.0], \
                          [0.0,1.0]])
                           
        q = numpy.matrix([[10.0**5.0, 0.0,0.0], \
                          [0.0,10.0**3.0,0.0], \
                          [0.0,0.0,-10.0]])

        ds = DareSolver(a,b,q,r)
        
        ds.iterative = True

        x = ds.solve()

        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[10.0**5.0,0.0,0.0], \
                          [0.0,10.0**3.0,0.0], \
                          [0.0,0.0,0.0]]), \
            3)
                    
    def test_direct(self):
        a = numpy.matrix([[0.8147, 0.1270],[0.9058, 0.9134]])
        b = numpy.matrix([[0.6324, 0.2785],[0.0975, 0.5469]])
        q = numpy.eye(2)
        r = numpy.matrix([[1.0,0.0],[0.0,1.0]])
        
        ds = DareSolver(a,b,q,r)
        x = ds.solve_direct()
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[2.6018,0.9969], \
                          [0.9969,1.8853]]), \
            3)
        
        # Make sure the convenience function works the same
        x = dare(a,b,q,r)
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[2.6018,0.9969], \
                        [0.9969,1.8853]]), \
            3)
        
    def test_cyclic(self):
        a = numpy.eye(2)
        b = -1.0*numpy.eye(2)
        r = numpy.eye(2)
        q = numpy.matrix([[1.0,0.0],[0.0,0.5]])
        
        ds = DareSolver(a,b,q,r)
        
        ds.use_cyclic = True
        
        x = ds.solve()
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[1.6180,0.0], \
                          [0.0,1.0]]), \
            3)
