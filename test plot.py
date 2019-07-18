import numpy as np
import pylab as pl
from matplotlib import collections  as mc

segments = []
colors = np.zeros(shape=(10,4))
x = range(10)
y = np.random.choice(10,10)
z = np.random.choice(10,10)
i = 0

for x1, x2, y1,y2, z1, z2 in zip(x, x[1:], y, y[1:], z, z[1:]):
    if z1 > 4:
        colors[i] = tuple([1,0,0,1])
    elif z1 <= 4:
        colors[i] = tuple([0,1,0,1])
    else:
        colors[i] = tuple([0,0,1,1])
    segments.append([(x1, y1), (x2, y2)])
    i += 1

lc = mc.LineCollection(segments, colors=colors, linewidths=2)
fig, ax = pl.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)
pl.show()