from scipy.signal.dlyap import dlyap_iterative, dlyap_via_sylvester
import numpy
import numpy.testing 

class DlyapTestCase(numpy.testing.TestCase):
    
    def setUp(self):
        self.a = numpy.matrix([[0.5,1.0],[-1.0,-1.0]])
        self.q = numpy.matrix([[2.0,0.0],[0.0,0.5]])
        
        self.a_big = numpy.matrix([[1.0,2.0,3.0,4.0],[2.0,0.0,0.0,1.0],[0.0,3.0,4.0,1.0],[2.0,6.0,1.0,2.0]])
        self.q_big = numpy.matrix([[3.0,2.0,6.0,4.0],[1.0,0.0,0.0,3.0],[1.0,3.0,4.0,2.0],[2.0,0.0,1.0,1.0]])

    def test_iterative(self):
        x = dlyap_iterative(self.a,self.q)
        
        numpy.testing.assert_array_almost_equal(x, numpy.matrix([[4.75,	-2.625],[-2.625, 4.1875]]),decimal=3)

        # This call should fail since the matrix eigenvalues are not within the
        # unit circle
        numpy.testing.assert_raises(ValueError,dlyap_iterative,self.a_big,self.q_big)
                          
    def test_direct(self):
        x = dlyap_via_sylvester(self.a,self.q)
        
        numpy.testing.assert_array_almost_equal(x, numpy.matrix([[4.75,	-2.625],[-2.625, 4.1875]]))

        x = dlyap_via_sylvester(self.a_big,self.q_big)
        
        numpy.testing.assert_array_almost_equal(x, \
            numpy.matrix([[0.138399144365353,-0.243875962261520,0.344760384605463,-0.228201132712656] ,\
                          [0.0925434259235586,0.0884072178472512,-0.0221287761259218,-0.0696345163629717] ,\
                          [-0.579436649992910,0.0199306956069676,-0.334512307641357,0.329463495821666], \
                          [-0.0378182254549707,-0.0570756676008797,-0.189044041973412,0.0668493567210922]]))
        