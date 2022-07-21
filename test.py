import PyCC
import numpy as np

df = PyCC.Distributions.Uniform(1000)
outdf,stats = PyCC.evaluate(df)
print(outdf)
print(stats)