from scipy.signal.dlyap import dlyap_iterative, dlyap_schur, dlyap_slycot
import numpy
import numpy.testing 

from unittest.case import SkipTest

class DlyapTestCase(numpy.testing.TestCase):
    
    def setUp(self):
        self.a = numpy.matrix([[0.5,1.0],[-1.0,-1.0]])
        self.q = numpy.matrix([[2.0,0.0],[0.0,0.5]])
    
    def test_iterative(self):
        x = dlyap_iterative(self.a,self.q)
        
        numpy.testing.assert_almost_equal(4.75,x[0,0],4)
        numpy.testing.assert_almost_equal(4.1875,x[1,1],4)
        
        for i in range(0,2):
            for j in range(0,2):
                if i != j:
                    numpy.testing.assert_almost_equal(-2.625,x[i,j],4)

    def test_direct(self):
        x = dlyap_schur(self.a,self.q)
        
        numpy.testing.assert_almost_equal(4.75,x[0,0],4)
        numpy.testing.assert_almost_equal(4.1875,x[1,1],4)
        
        for i in range(0,2):
            for j in range(0,2):
                if i != j:
                    numpy.testing.assert_almost_equal(-2.625,x[i,j],4)
                    
    def test_SLICOT(self):
        try:
            import slycot
            x = dlyap_slycot(self.a,self.q)
        
            numpy.testing.assert_almost_equal(4.75,x[0,0],4)
            numpy.testing.assert_almost_equal(4.1875,x[1,1],4)
        
            for i in range(0,2):
                for j in range(0,2):
                    if i != j:
                        numpy.testing.assert_almost_equal(-2.625,x[i,j],4)
                        
        except ImportError:
            raise SkipTest('slycot is not installed.')
