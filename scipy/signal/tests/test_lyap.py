from scipy.signal.lyap import lyap_bartels_stewart, sylvester_bartels_stewart
import numpy
import numpy.testing 

class LyapTestCase(numpy.testing.TestCase):
    
    def setUp(self):
        self.a = numpy.matrix([[0.5,1.0],[-1.0,-1.0]])
        self.q = numpy.matrix([[2.0,0.0],[0.0,0.5]])
        
        self.a_big = numpy.matrix([[1.0,2.0,3.0,4.0],[2.0,0.0,0.0,1.0],[0.0,3.0,4.0,1.0],[2.0,6.0,1.0,2.0]])
        self.q_big = numpy.matrix([[3.0,2.0,6.0,4.0],[1.0,0.0,0.0,3.0],[1.0,3.0,4.0,2.0],[2.0,0.0,1.0,1.0]])

        self.asyl = numpy.matrix([[4,],])
        self.bsyl = numpy.matrix([[2.0,1.0],[2.0,1.0]])
        self.csyl = numpy.matrix([[3.0,2.0],])

    def test_lyap_bartels_stewart(self):
        x = lyap_bartels_stewart(self.a,self.q)
        
        numpy.testing.assert_array_almost_equal(x, numpy.matrix([[7.0,-4.5],[-4.5,4.75]]))
        
        x = lyap_bartels_stewart(self.a_big,self.q_big)
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[-0.624124688227253,-0.147156736900327,-0.275565365308956,-0.524644396439268], \
                          [ 0.130010540266950,-0.219206244847271,0.652181626540601,-0.545124865637686], \
                          [-0.638202727946317,-0.653068678709705,-0.513671039312065,0.966881646368826], \
                          [ 0.780605908811037,0.579417258904439,-0.854852175364995,-0.457426713836971]]) )

    def test_sylvester_bartels_stewart(self):
        x = sylvester_bartels_stewart(self.asyl, self.bsyl, self.csyl)
        
        numpy.testing.assert_array_almost_equal(x, numpy.matrix([[-0.392857142857143,-0.321428571428571],]))
        