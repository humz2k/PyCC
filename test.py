import PyCC
import numpy as np

df = PyCC.Distributions.Plummer(1000)
outdf = PyCC.evaluate(df)
print(outdf)