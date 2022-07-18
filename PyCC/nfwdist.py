import numpy as np
from scipy import special

def pnfwunorm(q, con=5):
  if hasattr(con, '__len__'):
    y = np.outer(q,con)
  else:
    y = q*con
  return np.log(1.0 + y)-y/(1.0 + y)

def dnfw(x, con=5, log=False):
  if hasattr(con, '__len__'):
    con = np.array(con)
    d = np.outer(x,con**2)/((np.outer(x,con)+1.0)**2*(1.0/(con+1.0)+np.log(con+1.0)-1.0))
  else:
    d = (x*con**2)/(((x*con)+1.0)**2*(1.0/(con+1.0)+np.log(con+1.0)-1.0))
  if hasattr(x, '__len__'):
    d[x>1] = 0 
    d[x<=0] = 0
  else:
    if (x > 1):
      d = 0
    elif (x <= 0):
      d = 0
  if (log):
    return np.log(d)
  else:
    return d

def pnfw(q, con=5, logp=False):
  p = pnfwunorm(q, con=con)/pnfwunorm(1, con=con)
  if hasattr(q, '__len__'):
    p[q>1] = 1
    p[q<=0] = 0
  else:
    if (q > 1):
      p = 1
    elif (q <= 0):
      p = 0
  if(logp):
    return np.log(p)
  else:
    return p

def qnfw(p, con=5, logp=False):
  if (logp):
    p = np.exp(p)
  if hasattr(p, '__len__'):
    p[p>1] = 1
    p[p<=0] = 0
  else:
    if (p > 1):
      p = 1
    elif (p <= 0):
      p = 0
  if hasattr(con, '__len__'):
    p = np.outer(p,pnfwunorm(1, con=con))
  else:
    p *= pnfwunorm(1, con=con)
  return (-(1.0/np.real(special.lambertw(-np.exp(-p-1))))-1)/con

def rnfw(n, con=5):
  if hasattr(n, '__len__'):
    n=len(n)
  return qnfw(np.random.rand(int(n)), con=con)