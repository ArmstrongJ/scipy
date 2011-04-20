from scipy.signal.dare import DareSolver, dare
import numpy
import unittest

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
        ds.disable_slycot = True
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
