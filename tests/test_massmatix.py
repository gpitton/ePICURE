from interfaces import *
from utilities import *
import numpy as np
import matplotlib.pylab as plt

def test_massmatrix():
    print ("This should take ~8s in a core i5")
    for i in range(10):
        for j in range(2,i):
            vs=IteratedVectorSpace(UniformLagrangeVectorSpace(j), np.linspace(0,1,num=i))
            v=np.ones(vs.n_dofs)
            for k in range(2,10):
                t=massmatrix(vs,k)
                assert abs(v.dot(t.dot(v))-1.0)<10**(-10)
                csc=massmatrix(vs,k,format="CSC")
                csr=massmatrix(vs,k,format="CSR")
                assert (t==csc).all() and (t==csr).all()
                

test_massmatrix()
