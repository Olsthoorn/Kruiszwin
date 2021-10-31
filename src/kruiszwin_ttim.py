#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 00:32:57 2020

# Modeling the amount of water to discharge from sewage water working trench
in suburb Kruiszwin Julianadorp using `ttim`

The program `ttim` is Mark Bakker's software to model multilayer aquifer
systems, using analytical elements. It's available in Python via pip
(see documentation by googling Mark Bakker and ttim).

@author: Theo 20201114, 20201206
"""
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from matplotlib.path import Path
from matplotlib.transforms import Bbox
import pandas as pd
import ttim
import os
import sys
import pdb

sys.path.insert(0, '/Users/Theo/GRWMODELS/python/tools/')
sys.path.insert(0, '/Users/Theo/GRWMODELS/python/Nectaerra/Kruiszwin/src/')
import shape.shapetools as sht

gis  = '/Users/Theo/GRWMODELS/python/Nectaerra/Kruiszwin/data/spatial/'
home = '/Users/Theo/GRWMODELS/python/Nectaerra/Kruiszwin/'

import kruiszwin as kzw

os.chdir(home)

#%% ========= Generalities ===================================================
attribs = lambda obj: [o for o in dir(obj) if not o.startswith('_')]

def lss():
    """Return next linestyle in sequence."""
    ls_ = ['solid', 'dashed', 'dash_dot', 'dotted']
    for i in range(100):
        yield ls_[i % len(ls_)]


def clrs():
    """Return next color in sequence."""
    clr_ = 'rbgkmc'
    for i in range(100):
        yield clr_[i % len(clr_)]


def newfig(title="title?", xlabel="xlabel?", ylabel="ylabel?",
           xlim=None, ylim=None, xscale='linear', yscale='linear',
           size_inches=(14, 6)):
    """Generate a new axes on a new figure and return the axes."""
    fig, ax = plt.subplots()
    fig.set_size_inches(size_inches)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.grid()
    return ax


def lsloc(L=100., angle=90., x0=0., y0=0., n=100):
    """Return a line of length L under angle angle with center in x0, y0."""
    ex = np.cos(angle * np.pi / 180)
    ey = np.sin(angle * np.pi / 180)
    s = np.linspace(-L/2, L/2, n + 1)
    x = x0 + ex * s
    y = y0 + ey * s
    return x, y, s


def cobj2shapedf(cobj):
    """Return Shape_df object generated from contour object.

    We will turn the contour lines of a single level into a single path. Contour lines of a single
    level are not continuous; they generally consist of everal paths. Each contourline is therefore
    held in a linecollection object with .getpaths() as method. We turn the resulting paths into a single
    path using the path-codes to indicate where a new line starts.
    We combine the list if paths with the levels in a pd.DataFrame. Other information pertaining to the
    contourlines can be added to this DataFrame, for instance, the time, color etc.

    The resulting Shape_df object, can be saved to a shapefile, which allows subsequent visualization within
    QGIS.

    Parameters
    ----------
    cobj: contour object
        return value from ax.contour(...)

    @TO 20201116
    """
    LINETO = 2
    MOVETO = 1

    paths = []
    levels = []
    for linecollection, level in zip(cobj.collections, cobj.levels):
        path = np.zeros((0, 2))
        codes = np.array([], dtype=int)
        for p in linecollection.get_paths():
            if len(p) == 0:
                continue
            path = np.vstack((path, p.vertices))
            codes = np.hstack((codes, MOVETO, LINETO * np.ones(len(p) - 1, dtype=int)))
        if len(path) > 0:
            levels.append(level)
            paths.append(Path(path, codes=codes))

    return sht.Shape_df(paths=paths, data=pd.DataFrame(levels, columns=['level']), shapeType=3)


#%% ===== Read the data =======================================================

# Parameters to generate the model. Well use this as **kwargs

# Get the data from a cross section
dirs = kzw.Dir_struct(home=home, case_folder='Kruiszwin_ttim',
    executables={'mflow':'mf2005.mac',
                 'mt3d':'mt3dms5b.mac',
                 'seawat':'swt_v4.mac'})

workbook   = os.path.join(dirs.data, "Julianadorp.xlsx")
layers_df  = pd.read_excel(workbook, sheet_name='Boringen', engine="openpyxl")
piez_df    = pd.read_excel(workbook, sheet_name='Peilbuizen',
                        skiprows=[1], index_col=0, engine="openpyxl")
soil_props = pd.read_excel(workbook, sheet_name='Grondsoort',
                           index_col=0, engine="openpyxl")
spd_df     = pd.read_excel(workbook, sheet_name='SPD', index_col=0, engine="openpyxl")


# Make a list of Boring objects and reset their index
borehole = dict()
borehole_names = np.unique(layers_df['name'])
for name in borehole_names:
    borehole[name] = kzw.Boring(name, layers_df.loc[layers_df['name'] == name].copy())
    borehole[name].layers.index = np.arange(len(borehole[name].layers.index), dtype=int)

use_name = 'DKMG110'
print(borehole[use_name].layers)


## ==== Generate TTIM aquifer properties from layers
layers = borehole[use_name].layers

## Make sure tmin > 0
tmin = 1e-3
tmax = 200
props, aquifs, atards = kzw.layers2aquifs_atards(layers,
                    phreatictop=True, topboundary='conf', tmin=tmin, tmax=tmax, tstart=tmin)

pprint(props)
model = ttim.ModelMaq(**props)

print("\nAquifers:")
print(aquifs)
print("\nAquitards:")
print(atards)


# Add a linesink adn then solve:
# Specify tsandh (ts and h, tuples of simulation time and the head in the linesink)
lsProps = {'tsandh':[(  1e-3, -1.0)
                    ],
          'wh':'2H',           # two sided flowb
          'layers':[0],        # in which layers
          }

x0, y0, L, angle =0., 0, 100, 90
xLS, yLS, s = lsloc(L=100, angle=angle, x0=x0, y0=y0, n=5)

ls = ttim.linesink.HeadLineSinkString(model, xy=np.vstack((xLS, yLS)).T, label='TestSink', **lsProps)

model.solve()


# Gernate a line
x, y, s = lsloc(L=5 * L, angle=angle + 90., x0=x0, y0=y0, n=100)

# Choose a set of times
t = np.array([1, 3, 7, 28, 90, 180])

# Compute the head along the line x, y in the first and second aquifer:
h0 = model.headalongline(x, y, t, layers=0)[0]
h1 = model.headalongline(x, y, t, layers=1)[0]

ax = newfig(f"Head along line layer {0} d", "s", "Head [m]")

for tt, h in zip(t, h0):
    ax.plot(s, h, label=f't = {tt:.1f}')
ax.legend()

ax = newfig(f"Head along line layer {1}", "s", "Head [m]")
for tt, h in zip(t, h1):
    ax.plot(s, h, label=f't = {tt:.1f} d')
ax.legend()


# Show the discharge of the HeadLinsinkString:
Q = ls.discharge(t)

ax = newfig("Discharge from linsink", "time", "Q m3/d", ylim=(0, 200))
ax.plot(t[:-1], Q[0][:-1], label="Discharge")


bbox = Bbox.null()
bbox.update_from_data_xy(np.vstack((xLS, yLS)).T)
bbox.update_from_data_xy(np.vstack((  x,   y)).T)

L = 50 # Beetje ruimte houden rond de bbox

# Bereken een grid voor ruimtelijke visualisatie van de stijghoogten
x0, x1, y0, y1 = bbox.x0 - L, bbox.x1 + L, bbox.y0 - 2 * L, bbox.y1 + 2 * L
dx, dy = 5., 5.
xg = np.linspace(x0, x1, int((x1 - x0) / dx + 1))
yg = np.linspace(y0, y1, int((y1 - y0) / dy + 1))

# Berekenen kost even tijd
hgr  =model.headgrid(xg, yg, t=t[-1], printrow=True)

hgr.shape # layers, time, ny, nx


ax = newfig("Test one line sink", "xRD", "yRD", size_inches=(6, 12))

# we gaan verder werken met het quad object dat contour afscheidt. Dat bevat alle info van de contourlijnen.
levels= np.linspace(-1, 1, 21)
cobj = ax.contour(xg, yg, hgr[0, -1], levels=levels)

# En de aspect ratio, zodat km in beide richtingen even groot zijn.
ax.set_aspect(1)


# =========== MODELING Kruiszwin with TTIM ====================================

## Aquifer properties from layers
layers = borehole[name].layers

t = np.array([1, 3, 7, 28, 90, 180, 360, 720, 1440])

## Make sure tmin > 0
tmin = 1e-3
tmax = 200
props, _, _ = kzw.layers2aquifs_atards(layers,
                    phreatictop=True, topboundary='conf', tmin=tmin, tmax=t[-1], tstart=tmin)

pprint(props)
model = ttim.ModelMaq(**props)


# Add a linesink
# Specify tsandh (ts and h, tuples of simulation time and the head in the linesink)
sleufProps = {'tsandh':[(  1e-3, -1.0)
                    ],
          'wh':'2H',           # two sided flowb
          'layers':[0],        # in which layers
          }
slootProps = {'tsandh':[(  1e-3, -0.0)
                    ],
          'wh':'2H',           # two sided flowb
          'layers':[0],        # in which layers
          }


# Read the surface water lines
oppwater = sht.Shape_df()
oppwater.get(os.path.join(gis, 'oppwater'))

# Sleuf
xSl, ySl, s = lsloc(L=100., angle=-6., x0=111629., y0=544961., n=1)
xySl = np.vstack((xSl, ySl)).T

# Add the line sinks, direct van de gedigitaliseerde straten
sleuf = ttim.linesink.HeadLineSink(model,
                                   x1=xSl[0], x2=xSl[-1],
                                   y1=ySl[0], y2=ySl[-1],
                                   label='Sleuf21_kn89', **sleufProps)

lsinks = [sleuf]
for p, idx in zip(oppwater.data['path'], oppwater.data['FID']):
    lsinks.append(ttim.linesink.HeadLineSinkString(model, xy=p.vertices,
                                    tsandh=slootProps['tsandh'],
                                    wh=slootProps['wh'],
                                    layers=slootProps['layers'],
                                    label=f'sloot{idx}'))

model.solve()

L = 50 # Margin around the the bbox
x0, x1, y0, y1 = oppwater.bbox.x0 - L, oppwater.bbox.x1 + L, oppwater.bbox.y0 - L, oppwater.bbox.y1 + L

x0, y0 = 111350, 544800
x1, y1 = x0 + 400, y0 + 400

xg = np.linspace(x0, x1, int((x1 - x0) / 2.5 + 1))
yg = np.linspace(y0, y1, int((y1 - y0) / 2.5 + 1))

print(x0, x1)
print(y0, y1)

# Computing the grid takes a few minuts
print("Patience, computing the grid takes a few minutes...")
hgr  =model.headgrid(xg, yg, t=180, printrow=True)
print("Done!")

print("hgr.shape: ", hgr.shape) # 3 lagen een tijd en een vak van 32 rijen en 23 kolommen (25 m tussenafstand).
print("Max value in grid: {:.5f} m".format(np.max(hgr[0, 0])))


ax = newfig("Bemaling Kruiszwin", "xRD", "yRD", size_inches=(6, 12))

# we gaan verder werken met het quad object dat contour afscheidt. Dat bevat alle info van de contourlijnen.
levels = np.unique(np.hstack((-0.05, np.linspace(-1, 0, 11))))
cobj = ax.contour(xg, yg, hgr[0, 0], levels=levels)

# Even de rioolbuizen erbij tekenen
for p, idx in zip(oppwater.data['path'].values, oppwater.data['FID']):
    ax.plot(*p.vertices.T, 'k', lw=2, label=f'sloots{idx}')
ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], label='bbox heads')

# En de aspect ratio, zodat km in beide richtingen even groot zijn.
ax.set_aspect(1)


"""
How is cobj interpreted?

It has the contour levels, the number is `n`. And it has `.collections`
which contains `n` `lineCollection`s the line of which can be grabbed
using its `.get_paths()` method, yielding all the paths that constitute
the lines pertaining to each contour level.. It is, therefore, easy to
generate a shapefile that has the contour levels, so that it can be exported
to a shapefile.
"""

# Generate a shapefile with the contours for use in QGIS
contours = cobj2shapedf(cobj)
contours.save(os.path.join(gis, "ttim_verlSleuf21_180d"), shapeType=3)


