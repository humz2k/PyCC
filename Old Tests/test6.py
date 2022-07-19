import numpy as np
import scipy.optimize

def radiusFromPdf(Rvir, c, cumulativePdf):
    def mu(x):
        return np.log(1.0 + x) - x / (1.0 + x)

    def equ(c, target):
        return mu(c) - target

    def getX(c, p):
        target = mu(c) * p
        x = scipy.optimize.brentq(equ, 0.0, c, args = target)
        return x	

    # A simple root-finding algorithm. 
    x = getX(c, cumulativePdf)

    r = Rvir / c * x
	
    return r