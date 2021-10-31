#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:37:31 2020

@author: Theo
"""

#%% Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from matplotlib.transforms import Bbox
from matplotlib.pyplot import Line2D
import pandas as pd
import flopy
import flopy.utils.binaryfile as bf
from importlib import reload
from copy import copy
from pprint import pprint
import pdb

import os

os.environ["PYTHONUSERBASE"] = '/Users/Theo/miniconda3/envs/kruiszwin/bin'

import sys
tools   = '/Users/Theo/GRWMODELS/python/tools/'
GIS     = '/Users/Theo/GRWMODELS/python/Nectaerra/Kruiszwin/data/QGIS/'
src     = '/Users/Theo/GRWMODELS/python/Nectaerra/Kruiszwin/src/'
home    = '/Users/Theo/GRWMODELS/python/Nectaerra/Kruiszwin/'

sys.path.insert(0, tools)
sys.path.insert(0, src)

from shape import shapetools as sht
from fdm import Grid

os.chdir(home)

#%% General tools
attribs = lambda obj: [o for o in dir(obj) if not o.startswith('_')]

reload(sht)

AND = np.logical_and

MOVETO = 1
LINETO = 2
CLOSE  = 79

def clrs(clist=None):
    """Yield next color each time when called.

    Parameters
    ----------
    clists: list
        list of colors to get next from or None
    """
    if clist is None:
        clist = 'brgkmcy'
    for i in range(100):
        yield clist[i  % len(clist)]


def newfig(title=None, xlabel=None, ylabel=None, xlim=None, ylim=None,
           xscale=None, yscale=None, size_inches=(14, 6), fontsize=15):
    """Generate a standard new figure."""
    fig, ax = plt.subplots()
    fig.set_size_inches(size_inches)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    if xlim   is not None: ax.set_xlim(xlim)
    if ylim   is not None: ax.set_ylim(ylim)
    if xscale is not None: ax.set_xscale(xscale)
    if yscale is not None: ax.set_yscale(yscale)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    return ax


def newfig2h(titles=None, xlabels=None, ylabels=None, sharex=False, sharey=True,
           size_inches=(14, 6), shift_fraction=0.33):
    """Generate figure with two axes side-by-side of unequal width.

    Parameters
    ----------
    titles: tuple or list of two strings
        the titiles of the two axes
    xlabels: tuple or list of two strings
        the labels for the two x-axes
    ylabels: tuple or list of two strings
        the labels of the two y-axes
    sharex, sharey: bool
        whether or not to share the x or y axes
    size_inches: tuplle of 2 floats
        the size of the figure in inches
    shift_fraction: float between -0.5 and 0.5
        the fraction of the horizontal axes of the first axes
        that is added to the first horizotal axes and subtacted from the
        second horizontal axes.
        If negative, the first horizontal axes will be smaller than the second.
        If positive, the second hor. axes will be smaller than the first.
    """
    fig, axs = plt.subplots(1, 2, sharex=sharex, sharey=sharey)
    fig.set_size_inches(size_inches)
    for ax, title, xlabel, ylabel in zip(axs, titles, xlabels, ylabels):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid()

    # We extend the width of the left axes at the expense of the right axes: =================================
    ax0b = list(axs[0].get_position().bounds)
    ax1b = list(axs[1].get_position().bounds)

    b = ax0b[2] * shift_fraction

    ax0b[2] += b
    ax1b[2] -= b
    ax1b[0] += b

    axs[0].set_position(ax0b)
    axs[1].set_position(ax1b)

    return axs

def my_legend(ax=None, label_color_dict=None, **kw):
    """Add labels to the legend.

    Parameters
    ----------
    label_color_dict: dictionary
        keys are labels, and values are the desired colors.
    """
    legend_elements = []
    labels = []
    for label in label_color_dict:
        clr = label_color_dict[label]
        legend_elements.append(Line2D([0], [0], color=clr, label=label))
        labels.append(label)
    ax.legend(legend_elements, labels, **kw)


class Dir_struct:
    """KruisZwin directory structure.

    Expected directory structure

    Home/ # home directory
            bin/
                mfusg_X64.exe
                mfusg.mac
            src/
            data/
                spatial/
            doc/
            notebooks/
            cases/
    ../python/KNMI
    """

    def __init__(self, home='.', case_folder=None, executables=None):
        """Generate directory structure.

        Parameters
        ----------
        home: str
            path to home directory
        case_folder: str
            the basename name of the current case folderin the folder cases
        """
        self.home = os.path.abspath(os.path.expanduser(home))
        self._case_folder = case_folder
        self.executables = {k: os.path.join(self.bin, executables[k]) for k in executables}
        #verify existance of required files
        for k, exe in self.executables.items():
            assert os.path.isfile(exe), "Missing executable for {} '{}' not a file.".format(k, exe)
        if not os.path.isdir(self.case_folder): os.mkdir(self.case_folder)
        #assert os.path.isdir(self.meteo), "Missing meteodir '{}'.".format(self.meteo)
        #assert os.path.isdir(self.bofek), "Missing bofek folder '{}'.".format(self.bofek)

    # Directory structure
    @property
    def bin(self):
        """Yield bindary folder."""
        return os.path.join(self.home, 'bin')
    @property
    def src(self):
        """Yield source code folder."""
        return os.path.join(self.home, 'src')
    @property
    def data(self):
        """Yield data folder."""
        return os.path.join(self.home, 'data')
    @property
    def cases(self):
        """Yield folder where cases are stored."""
        return os.path.join(self.home, 'cases')
    @property
    def meteo(self):
        """Yield meteo data folder."""
        return os.path.join(self.data, 'meteo')
    @property
    def spatial(self):
        """Yield folder with spatial data.

        Each case corresponds with a folder with the case name.
        """
        return os.path.join(self.data, 'spatial')
    # @property
    # def bofek(self):
    #     """Return directory of bofek units."""
    #     return os.path.join(self.data, 'bofek')
    @property
    def case_folder(self):
        """Return results directory of current case."""
        return os.path.join(self.cases, self._case_folder)
    @property
    def case_results(self):
        """Return folder where case output goes."""
        return self.wd
    @property
    def wd(self):
        """Yield working directory (MODFLOW output) depending on case."""
        if not os.path.isdir(self.cases): os.mkdir(self.cases)
        wd = os.path.join(self.cases, self.case_folder)
        if not os.path.isdir(wd): os.mkdir(wd)
        return wd
    def cwd(self):
        """Change to current working directory."""
        os.chdir(self.wd)

#% Boreholes and piezometers

class Piezometers:
    """Piezometers."""

    def __init__(self, piezoms=None):
        """Initialize piezometers.

        Properties
        ----------
        piezoms: pd.DataFrame
            the piezometers, one record per piezometer.
            All piezometer data is contained in the piezometer's record.
        """
        self.data = piezoms

        # Add the vertial postion to the dataframe as column 'path
        ppath = []
        for i in self.data.index:
            piez = self.data.loc[i]
            ppath.append(Path(np.array([(0, piez['bkfNAP']), (0, piez['okfNAP'])]),
                           codes = [MOVETO, LINETO]))
        self.data['path'] = ppath


    def plot(self):
        """Plot the piezometers next to each other."""
        ax = newfig('piezometers', '', 'elevation [mNAP]')

        head, mv, xp = [], [], []
        for idx, x in zip(self.data.index, range(len(self.data))):
            piezom = self.data.loc[idx]
            xy = piezom['path'].vertices
            xy[:, 0] = x
            ax.plot(*xy.T, color='darkgray', lw=4)
            ax.text(*xy[0], ' _' + idx)
            # Plot embellishments
            mv.append(  piezom['mvNAP'])
            head.append(piezom[ 'hNAP'])
            xp.append(x)

        mv   = np.asarray(mv)
        head = np.asarray(head)
        xp   = np.asarray(xp)

        ax.plot(xp, mv,   'g', label='maaiveld')
        ax.plot(xp, head, 'b', label='waterstand')
        ax.legend()
        return ax


class Boring:
    """Boorgat with layers."""

    def __init__(self, name, layers, xlim=(-5, 5)):
        """Instantiate a new boring.

        Parameters
        ----------
        name: str
            the name of the borehole
        layers: pd.DataFrame
            the table of properties of each layer
        v =
        """
        self.name = name
        self.layers=layers # pd.DataFame
        self.mv = self.layers.iloc[0]['NAPtop'] # Ground surface

        self.mk_profile_patches()
        self.mk_layer_patches_and_paths(xlim=xlim)

    def print(self):
        """Print self."""
        print(self.layers)

    def mk_profile_patches(self):
        """Make list of patches showing the layers."""
        left  = -self.layers['profile_width'].values / 4
        right = +self.layers['profile_width'].values / 4
        bot   =  self.layers['NAPbot'].values
        top   =  self.layers['NAPtop'].values
        XY    = np.array([[left, left, right, right, left], [top, bot, bot, top, top]]).T

        patches = []
        for xy, fc in zip(XY, self.layers['color']):
            pth = Path(xy, codes=[MOVETO, LINETO, LINETO, LINETO, CLOSE])
            patches.append(PathPatch(pth, ec='black', fc=fc))
        self.layers.loc[:, 'profile_patch'] = patches

        self.profile_bbox = Bbox([(left[ 0] * 2, bot[-1] - 1),
                                  (right[0] * 2, top[ 0] + 1)])
        return


    def plot_profile(self, ax=None, xlim=None, ylim=None, title_fontsize=10, size_inches=None):
        """Plot this borehole."""
        self.mk_profile_patches()

        if ax is None:
            fig, ax = plt.subplots()
            ax.grid(which='major', axis='y')
            if size_inches is not None: fig.set_size_inches(size_inches)
        ax.set_title('{}'.format(self.name), fontsize=title_fontsize)
        ax.set_ylabel('NAP [m]')

        for i in self.layers.index:
            ax.add_patch(self.layers['profile_patch'].loc[i])

        ax.set_xlim(xlim if xlim is not None else (self.profile_bbox.x0, self.profile_bbox.x1))
        ax.set_ylim(ylim if ylim is not None else (self.profile_bbox.y0, self.profile_bbox.y1))
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        return


    def mk_layer_patches_and_paths(self, xlim=(-5, 5)):
        """Make at patches and path of layers to self.

        Parameters
        ----------
        b: 2-tuple of floats
            left and right end-coordinates of layers (i.e. the grid)
        """
        nlay = len(self.layers)
        left  = xlim[0] * np.ones(nlay)
        right = xlim[1] * np.ones(nlay)
        bot   =  self.layers['NAPbot'].values
        top   =  self.layers['NAPtop'].values
        XY    = np.array([[left, left, right, right, left], [top, bot, bot, top, top]]).T

        patches = []
        paths   = []
        for xy, fc in zip(XY, self.layers['color']):
            pth = Path(xy, codes=[MOVETO, LINETO, LINETO, LINETO, CLOSE])
            paths.append(pth)
            patches.append(PathPatch(pth, ec='black', fc=fc))
        self.layers.loc[:, 'layer_patch'] = patches
        self.layers.loc[:, 'path'] = paths

        self.layers_bbox = Bbox([(left[ 0] * 2, bot[-1] - 1),
                                 (right[0] * 2, top[ 0] + 1)])
        return


    def plot_layers(self, xlim=None, ylim=None, ax=None):
        """Plot the layers represented by borehole bore.

        Parameters
        ----------
        xlim: sequence of 2 floats
            xlim for plot
        use: matplotlib.Axes object
            axes to use for plotting
        """
        if not 'layer_patch' in self.layers.columns:
            raise ValueError("field 'layer_patch' missing in self,\n" +
                "Run method mk_layer_patches_and_paths(xlim=(xmin, xmax) first.")
        for patch in self.layers['layer_patch']:
            new_patch = copy(patch)
            ax.add_patch(new_patch)

        xlim_not_given = xlim is None
        ylim_not_given = ylim is None
        if xlim_not_given:
            xlim = ax.get_xlim()
            xlim = min(xlim[0], self.layers_bbox.x0), max(xlim[1], self.layers_bbox.x1)
        if ylim_not_given:
            ylim = ax.get_ylim()
            ylim = min(ylim[0], self.layers_bbox.y0), max(ylim[1], self.layers_bbox.y1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return


def plot_bores(bores, size_inches=(14, 6), ylim=None, title_fontsize=10):
    """Plot series of boreholes in separate axes.

    Parameters
    ----------
    bores: list of Boring objects
        the boreholes to be plotted as in the list
    """
    fig, axs = plt.subplots(1, len(bores), sharex=True, sharey=True)
    fig.set_size_inches(size_inches)

    # make ylim match all bores
    ylim_not_given = ylim is None

    if ylim_not_given:
        ylim = np.inf, -np.inf

    for name in bores:
        bore = bores[name]
        bore.mk_profile_patches()
        if ylim_not_given:
            ylim = min(ylim[0], bore.profile_bbox.y0), max(ylim[1], bore.profile_bbox.y1)

    # then plot each of them in its own axes
    for name, ax in zip(bores, axs):
        ax.grid(which='major', axis='y')
        bores[name].plot_profile(ax=ax, ylim=ylim,  title_fontsize=title_fontsize)
    return


#%% TTIM

def ttm_props(aquifs, atards, phreatictop=True, topboundary='conf',
                         tmin=0, tmax=10, tstart=0,
                         M=10,
                         verbose=False):
    """Return ttim aquifer props.

    Parameters
    ----------
    aquifs: pd.DataFrame with fields ['top', 'bot', 'Saq', 'kaq']
        aquifer properties
    atards: pd.DataFrame with fields['top', 'bot', 'Sll', 'c']
        aquitard properties
    phreatictop: bool
        if True --> Saq is treated as Sy otherwise as Ss is used
    topboundary: 'conf' or 'semi'
        whether or not the topboundary seni-confined.
    tmin, tmax, tstrt:  floats
        lowest and largest time for which results are to be computed.
        tstart is the first time.
    M: default 10
        number of coefficients for bactransformation of Laplace transform
    verbose: bool
    @TO 20201202
    """
    from pprint import pprint
    props = {
        'z': np.unique(np.hstack((aquifs['top'].values, aquifs['bot'].values,
                  atards['top'].values, atards['bot'].values)))[::-1],
        'kaq': aquifs['kaq'].values,
        'Saq': aquifs['Saq'].values,
        'c'  : atards['c'  ].values,
        'Sll': atards['Sll'].values,
        'phreatictop': phreatictop,
        'topboundary': topboundary,
        'tmin': tmin,
        'tmax':tmax,
        'tstart':tstart,
        'M':M}

    if verbose:
        pprint(props)
    return props



def layers2aquifs_atards(layers, phreatictop=True, topboundary='conf',
                         tmin=0, tmax=10, tstart=0, M=10,
                         verbose=False):
    """Return a list f aquifers and aquitards.

    Parameters
    ----------
    layers: pd.DataFrame
        layer properties. Required fields ['NAPtop', 'NAPbot', 'kh', 'kv', 'Sy', 'Ss']
    phreatictop: bool
        if True --> Saq is treated as Sy otherwise as Ss is used
    topboundary: 'conf' or 'semi'
        whether or not the topboundary seni-confined.
    tmin, tmax, tstrt:  floats
        lowest and largest time for which results are to be computed.
        tstart is the first time.
    M: default 10
        number of coefficients for bactransformation of Laplace transform
    verbose: bool
        info while running
    @TO 20201202
    """
    aquifs = []
    atards = []
    for k, lay in layers.T.items():
        #print(k, rec)
        D = lay['NAPtop'] - lay['NAPbot']
        if lay[  'stype'].startswith('zand'):
            aquifs.append({'top':lay['NAPtop'], 'bot':lay['NAPbot'], 'Saq': lay['Ss'] * D, 'kaq': lay['kh']})
        else:
            atards.append({'top':lay['NAPtop'], 'bot':lay['NAPbot'], 'Sll': lay['Ss'] * D, 'c': D / lay['kv']})

    if verbose:
        print("Aquifers:")
        pprint(pd.DataFrame(aquifs)); print()
        print("Aquitards")
        pprint(pd.DataFrame(atards)); print()

    # Merge aquifers layers if possible
    aqsnew = []
    a1 = aquifs.pop(0)
    while len(aquifs) > 0:
        a2 = aquifs.pop(0)
        if a1['bot'] != a2['top']:
            aqsnew.append(a1)
            a1 = a2.copy()
        else:
            D1 = a1['top'] - a1['bot']
            D2 = a2['top'] - a2['bot']
            kh = (a1['kaq'] * D1 + a2['kaq'] * D2) / (D1 + D2)
            a1 = {'top': a1['top'], 'bot': a2['bot'], 'Saq': a1['Saq'] + a2['Saq'], 'kaq': kh}
    aqsnew.append(a1)

    atsnew = []
    a1 = atards.pop(0)
    while len(atards) > 0:
        a2 = atards.pop(0)
        if a1['bot'] != a2['top']:
            atsnew.append(a1)
            a1 = a2.copy()
        else:
            c_new =  a1['c'] + a2['c']
            a1 = {'top': a1['top'], 'bot': a2['bot'], 'Sll': a1['Sll'] + a2['Sll'], 'c': c_new}
    atsnew.append(a1)

    if phreatictop:
        aqsnew[0]['Saq'] = layers.loc[0, 'Sy']

    aquifs = pd.DataFrame(aqsnew)
    atards = pd.DataFrame(atsnew)
    if verbose:
        print("Aquifers reduced:")
        pprint(aquifs); print()
        print("Aquitards reduced:")
        pprint(atards); print()

    props = ttm_props(aquifs, atards,
                      phreatictop=phreatictop,
                      topboundary=topboundary,
                      tmin=tmin,
                      tmax=tmax,
                      tstart=tstart,
                      M=M)

    return props, aquifs, atards


#%% Trenches

def get_clean_pipes(pipes=None):
        """Return data from pipes for this point.

        The original data are cleaned from unnecessary columns, while other
        columns are added including a path column with the Path of the pipe.

        Parameters
        ----------
        pipes: str or sht.Shape_df
            name of the shapefile (basename or full name)
            The shapefile should contain the pipes with 'BOB' and 'diameter'

        Returns
        -------
        sht.Shape_df with the pipes cleaned from uncessary data with fields
            'BOB', 'diam', 'rp', 'zp', 'length'
        """
        # Tursh this Shape_df into a DataFrame with a path column
        if isinstance(pipes, str):
            fname = pipes
            pipes = sht.Shape_df()
            pipes.get(fname)
            pipes = pipes.data

            if not isinstance(pipes, pd.DataFrame):
                raise ValueError("Unknown data type {} for pipes.".format(type(pipes)))

        DUMMYDIAMETER = 349 # Unique dummy diameter

        length = lambda pth: np.sum(np.sqrt(np.sum(np.diff(pth.vertices, axis=0) ** 2, axis=1)))

        pipes = pipes[['Z1', 'diameter', 'DWA/HWA', 'path']]
        pipes.columns = ['BOB', 'diam', 'type', 'path']

        pipes.loc[np.isnan(pipes['diam']), 'diam'] = DUMMYDIAMETER

        pipes['rp'] = pipes['diam'].copy() / 2000
        pipes['zp'] = pipes[ 'BOB'].copy() + pipes['rp'].copy()
        pipes['length'] = [length(p) for p in pipes['path']]

        Id = [idx for idx, b in zip(pipes.index, pipes['BOB']) if not np.isnan(b)]

        pipes = pipes.loc[Id]
        return pipes # returns a ordinary DataFrame with one column holing the path of each pipe


class Trench:
    """Class that defines a trench with pipes."""

    def __init__(self, point=None, pipes=None, mv=None, spaceH=0.2, spaceV=0.3):
        """Return a single cross section.

        Parameters
        ----------
        point: tuple
            coordinates of ancher point for this cross section
        pipes: sht.Shape_df
            cleaned pipes see "get_cleaned_pipes"
        profile: sht.Shape_df
            the drilling profile for this cross section
        mv: float
            ground surface elevation
        spaceH, spaceV: float, float
            horizontal space between trench wall and trench floor
        """
        reload(sht)
        self.point = point
        self.mv = mv
        self.spaceH = spaceH
        self.spaceV = spaceV

        self.get_info_of_nearest_2_pipes(pipes)
        return


    def get_info_of_nearest_2_pipes(self, pipes):
        """Return the distance from self.point to all pipes.

        We look for the two pipes in the streets near the points that were
        specified as locations of desired cross sections. (In self.points).

        The trench in each street may contain one or two pipes. If only one is
        present, the first is duplicated to allow uniform handling of streets
        with one and with two pipes.

        The second pipe is assumed absent (i.e. in another street) when
        the perpendicular distance d from the pont to the pipe is larger than 5 m.

        Parameters
        ----------
        pipes: sht.Shape_df
            the pipes in the shapefile
        mv: float
            ground surface elevation
        spaceH, spaceV: float, float
            horizontal space between trench wall and trench floor

        Attributes
        ----------
        d: 2-tuple: distance between point and each of the two pipes in the trench
        ip: 2-tuple, indices of the two pipes in the trench
        p0: 2-tuple, coordinates of intersection of perpendicular line from point to pipe 1
        p1: 2-tuple, coordinates of intersection of perpendicular line from point to pipe 2
        pm: 2-tuple, midpoint between p0 and p1, is center of trench close to points in self.points
        pipes: 2-tuple each pd.Series, the two pipes in the trench
        b: float, half the distance between the two pipes in the trench
        rp : 2-tuple, radii of the pipes in the trench
        zpipe: 2-tuple, vertical elevation of center of the pipes in the trench
        xpipe: 2-tuple, position of left and right pipe relative to center of trench
        mv [m]: ground-surface elevation
        spaceH: space between pipe and vertical trench wall
        spaceV: space between pipe and trench floor
        bb: matplotlib.transfoms.Bbox of vertical xsec of trench
        trench: matplitlib.path.Path of vertical circumf of trench
        """
        # Get the two pipes neares to the point

        self.pipe_color = lambda pipe_type: 'blue' if pipe_type=='HWA' else 'green'

        R = np.array([sht.point2line(self.point, line) for line in pipes['path']])
        IL = np.logical_and(R.T[1] >= 0., R.T[1] <= 1.)
        RI = sorted([(*r, i) for r, i in zip(R[IL], pipes.index[IL])], key=lambda x: x[0])[:2]
        self.d  = RI[0][0], RI[1][0] # Distance to two nearest pipes
        self.ip = RI[0][4], RI[1][4] # Index of the two nearest pipes
        self.p0 = RI[0][2], RI[0][3] # intersection with first nearest pipe
        self.p1 = RI[1][2], RI[1][3] # intersection with second nearest pipe
        if self.d[1] > 5:
            # In this case, the second pipe in other street, duplicate first pipe
            # To allow the procedure to continue without if statements.
            self.ip = self.ip[0], self.ip[0]
            self.d  = self.d[ 0], self.d[ 0]
            self.p1 = self.p0
        self.pipes = [pipes.loc[i] for i in self.ip] # remember the two pipes (each is a pd.Series)
        self.pm = (self.p0[0] + self.p1[0]) / 2, (self.p0[1] + self.p1[1]) / 2
        self.b  = (self.d[1] - self.d[0]) / 2
        self.rp = tuple([p['rp'] for p in self.pipes])
        self.zpipe = tuple([p['BOB'] + p['rp'] for p in self.pipes])
        self.xpipe = (-self.b, self.b)
        zb = np.min(np.array([p['BOB'] - self.spaceV for p in self.pipes]))
        ut = self.xpipe[0] -self.rp[0] - self.spaceH, self.xpipe[1] + self.rp[1] + self.spaceH

        mv = self.mv
        self.trench_path = Path(np.array([[ut[0], ut[0], ut[1], ut[1], ut[0]],
                                     [mv, zb, zb, mv, mv]]).T,
                           codes=[MOVETO, LINETO, LINETO, LINETO, CLOSE])
        self.trench_patch = PathPatch(self.trench_path, fc='gold')
        self.pipe_colors = tuple([self.pipe_color(pipe['type']) for pipe in self.pipes])
        self.pipe_patch = tuple([Circle((x, z), pipe['rp'], fc=self.pipe_color(pipe['type']), alpha=0.5)
                    for x, z, pipe in zip(self.xpipe, self.zpipe, self.pipes)])
        # TODO generalize this height for all drllings
        zt_ghb = zb + self.spaceV
        self.ghb_path = Path(np.array([[ut[0], ut[0], ut[1], ut[1], ut[0]],
                        [zt_ghb, zb, zb, zt_ghb, zt_ghb]]).T)
        self.bbox = sht.Bbox([[ut[0], zb],[ut[1], mv]])
        return

    def set_soil_props(self, soil_props, trench_name='TRENCH'):
        """Set soil_props using soil_props Series or dict and add trench_path.

        Parameters
        ----------
        soil_props: pd.Series or dict
            soil properties 'kh'', ''kv', 'c' etc
        """
        self.soil_props = soil_props
        self.soil_props['path'] = self.trench_path
        self.soil_props['name'] = 'TRENCH'


    def plot(self, ax=None, mv=None, aspect='equal', trenchcolor=None, **kw):
        """Plot a trench.

        Paramters
        ---------
        ax: plt.Axes
            axes to plot on
        mv: float
            ground-surface-elevation
        kw: keyword arguments
            passed on to patches.PathPatch
        aspect: use 'auto' or 'equal' or a num = hfac/wfac
            set to 1 to make scales equal and circles round
            None to ignore it.
        """
        fc = 'gold' if trenchcolor is None else 'white'

        if mv is not None:
            x, z = self.trench_path.vertices.T
            z[[0, -2, -1 ]] = mv
            xy = np.vstack((x, z)).T
            codes=[MOVETO, LINETO, LINETO, LINETO, CLOSE]
            path = Path(xy, codes=codes)
        else:
            path = self.trench_path
        ax.add_patch(PathPatch(path, fc=fc))
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xtr, ztr = path.vertices.T
        xlim = min(xlim[0], np.min(xtr)), max(xlim[1], np.max(xtr))
        xlim = min(xlim[0], -xlim[1]), max(-xlim[0], xlim[1])
        ylim = min(ylim[0], np.min(ztr)), max(ylim[1], np.max(ztr))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if trenchcolor is None:
            ax.add_patch(Circle((self.xpipe[0], self.zpipe[0]), self.rp[0],
                                fc=self.pipe_color(self.pipes[0]['type'])))
            if self.b > 0:
                ax.add_patch(Circle((self.xpipe[1], self.zpipe[1]), self.rp[1],
                                    fc=self.pipe_color(self.pipes[1]['type'])))
        ax.add_patch(PathPatch(self.ghb_path, fc='orange', alpha=0.5, zorder=200))
        ax.set_aspect(aspect)
        return


def plot_section(boring=None, trench=None, xlim=None, ylim=None,
                 ax=None, aspect='auto', trenchcolor=None, **kw):
    """Compute coordinates of Trench and plot it to check.

    Parameters
    ----------
    boring: of class Boringen
        the borehole to base teh section on
    trench: of class Trench
        the trench to be placed in this soil defined by borehole
    ax: matplotlib.Axis object
        plot if not None
    aspect: use 'auto' or 'equal' or a number = hfac/wfac
        the desired aspect ratio of y and x scales
    trench_color color
        Color use inside the trench if not None, then the pipes are also not
        drawn.
    """
    # coordinates of pipe
    boring.plot_layers(ax=ax, xlim=xlim, ylim=ylim)
    trench.plot(ax=ax, mv=boring.mv, aspect=aspect, trenchcolor=trenchcolor)
    #ax.set_xlim((boring.layers_bbox.x0, boring.layers_bbox.x1))
    #ax.set_ylim((boring.layers_bbox.y0, boring.layers_bbox.y1))
    return ax


class Trench_collection:
    """Class that defines a cross section to be modelled."""

    def __init__(self, shpfname_points, shpfname_pipes, mv=None):
        points = sht.Shape_df()
        points.get(shpfname_points)
        self.points = points.data

        self.pipes = get_clean_pipes(shpfname_pipes)
        pnts = [p.vertices[0] for p in self.points['path']]
        self.trenches = [Trench(point=p, pipes=self.pipes, mv=mv) for p in pnts]


    def __getitem__(self, key):
        """Return a trench.

        Parameters
        ----------
        key: int or slice
            number in list of trenches
        """
        return self.trenches[key]


    def __setitem__(self, key, value):
        """Set item in a trench.

        Parameters
        ----------
        key: int (or sice)
            index in list of trenches
        value:
            value to be set for the specified trenches
        """
        self.trenches[key] = value



    def plot_points(self, ax=None, **kw):
        """Plot the cross section points on the map in xRD yRD coordinates.

        Parameters
        ----------
        ax: plt.Axes
            the axes to plot on
        kw: dict
            keyword arguments passed on to ax.plot
        """
        for i, (tr, clr) in enumerate(zip(self.trenches, clrs())):
            ax.plot(*tr.pm, marker='o', mfc=clr) # plots exact trench center!

            ax.plot(*np.array([tr.p0, tr.p1]).T, color=clr) # The trench

            pth = tr.pipes[0]['path']
            ax.plot(*pth.vertices.T,  '-', color=clr, **kw)
            ax.plot(*pth.vertices[ 0], '.', mfc=clr, mec=clr, **kw)
            ax.plot(*pth.vertices[-1], '.', mfc=clr, mec=clr, **kw)
            ax.text(*tr.pm, '_punt/dsn {}'.format(i))
        #ax.legend()

        return


    def plot_pipes(self, ax=None, **kw):
        """Plot the pipes on a map with xRD and yRD coordinates.

        Parameters
        ----------
        ax: plt.Axes
            axes to plot on
        kw: dict
            keyword arguments passed on to ax.plot
        """
        diams = list(np.unique(self.pipes['diam']))
        pipe_clrs = [(a, a, a) for a in np.arange(len(diams) + 1) / (1 + len(diams))][::-1]
        clrs = lambda diam: pipe_clrs[diams.index(diam)]

        for i in zip(self.pipes.index):
            pipe = self.pipes.loc[i]
            clr = clrs(pipe['diam'])
            ax.plot(*pipe['path'].vertices.T, '-', color=clr, marker='.' ,mfc=clr, mec=clr, **kw)

        ax.legend([plt.Line2D([0], [0], color=clrs(diam)) for diam in diams],
                  ['diam={} mm'.format(diam) for diam in diams])
        return

    def plot_trenches(self, xlim=None, ylim=None, **kw):
        """Plot all trenches, as cross sections, each in it's own axes.

        Parameters
        ----------
        kw: dict
            keyword arguments passed on to trench.plot
        """
        bbs = []
        for i, trench in enumerate(self.trenches):
            bbs.append(trench.bbox)
        bbox = Bbox.union(bbs)

        for i, trench in enumerate(self.trenches):
            ax = newfig("Trench {}".format(i), "x m", "z [m]")
            trench.plot(ax=ax)

            ax.set_xlim((bbox.x0, bbox.x1) if xlim is None else xlim)
            ax.set_ylim((bbox.y0, bbox.y1) if ylim is None else ylim)
        return



class Salinity_profile:
    """Salinity profile class definition."""

    def __init__(self, name=None, mv=None, xRD=None, yRD=None, tds=None):
        """Return salinity profile.

        Parameters
        ----------
        name: str
            name of the borehole or sounding of this profile.
        mv: float
            ground surface elevation
        xRD: float
            x-coordinate (national Dutch coordinate system)
        yRD: float
            y-coordinate (national Dutch coordinate system)
        tds: pd.DataFrame
            the actual profile with fields ['z', 'Depth [m]', 'TDS [mg/l]']
        """
        self.name = name
        self.mv = mv
        self.xRD = xRD
        self.yRD = yRD
        self.tds = tds


    def plot(self, ax=None, **kw):
        """Plot salinity profile."""
        XZ = np.asarray(self.tds[['TDS [mg/l]', 'z']].copy())
        ax.plot(*XZ.T, label=
                'TDS({}) mf/L, (xRD,yRD,mv)=({:.2f}, {:.2f}, {:.3f})'
                .format(self.name, self.xRD, self.yRD, self.mv), **kw)
        return


    def interp(self, zdata=None):
        """Interpolate the salinity data to zdata.

        Parameters
        ----------
        name: str
            borehole/profile name
        zdata: np.ndarray
            z values in m NAP
        """
        tdsNew = np.interp(zdata[::-1], self.tds['z'].values[::-1], self.tds['TDS [mg/l]'].values[::-1])[::-1]
        return tdsNew, zdata


    def get_conc(self, gr=None):
        """Return concentration of data of sounding with given name.

        Parameters
        ----------
        name: str
            name of profile borehole or sounding
        gr: fdm.Grid object
            the MODFLOW network
        """
        conc, _ = self.interp(zdata=gr.ZM[:, 0, 0])
        return conc[:, np.newaxis, np.newaxis] * gr.const(1.)


class Salinity_profile_collection:
    """Collection of salinity profiles."""

    def __init__(self, workbook, sheet_name=None, coords_sheet='Coords'):
        """Generate a collection of salinity profiles.

        Parameters
        ----------
        workbook: fname
            Excel workbook name with the data
        sheet_name: str
            sheet_name in the workbook with the TDS data (specific to the file)
        coord_sheet: str
            sheet_name in the workbook with the coordinates and surface elevation of boreholes.
        """
        # Get the salinity profile data first and do some cleaning
        salinities = pd.read_excel(workbook, sheet_name=sheet_name, header=[15, 16], engine='openpyxl')
        columns = [col for col in salinities.columns if not col[0].startswith('Unnamed')]
        salinities = salinities[columns]
        salinities.columns = [(col[0], 'Depth [m]') if col[1].startswith('Gecor') else col for col in salinities.columns]

        # Get coordinates and ground surface elevation from coordinates sheet in workbook (plus some cleaning)
        coords = pd.read_excel(workbook, sheet_name=coords_sheet, engine='openpyxl')
        coords.columns = ['name', 'xRD', 'yRD', 'mv', 'Opmerking']
        coords.index = coords['name']
        coords = coords.drop(columns=['name'])

        bore_names = list(np.unique([col[0] for col in salinities.columns]))
        self.bbox = Bbox.null()
        self.profiles = dict()
        for name in bore_names:
            cols = [col for col in salinities.columns if col[0]==name]
            tds = salinities[cols]
            tds.columns = [col[1] for col in cols]
            tds = tds.dropna()
            tds['z'] = tds['Depth [m]'] + coords.loc[name, 'mv']
            self.profiles[name] = Salinity_profile(
                            name=name, mv=coords.loc[name, 'mv'],
                            xRD= coords.loc[name, 'xRD'],
                            yRD= coords.loc[name, 'yRD'],
                            tds=tds)


    def __getitem__(self, key):
        """Return the disisred profile from the collection."""
        return self.profiles[key]


    def __setitem__(self, key, value):
        """Set self.profiles[key] with value."""
        self.profiles[key] = value

    def plot(self, names=None, ax=None, **kw):
        """Plot salinity profile."""
        all_names = self.profiles.keys()
        if names is None:
            names = all_names
        elif not isinstance(names, (tuple, list)):
            names = [names]
            rest = set(names).difference(all_names)
            if len(rest) > 0:
                raise ValueError('names {} are unknown,\n'.format(str(rest)) +
                'use one or more of {} or None for all'.format(str(all_names)))
        for name in names:
            self.profiles[name].plot(ax=ax, **kw)
        return


#%% Modflow


def shapes_from_trench(zTop=None, zPipe=None, rPipe=None, space=0.2, soil_props=None, xPipe=0.):
    """Return a Shape_df for the trench with its pipe.

    Parameters
    ----------
    zTop: float
        ground  sruface elevation
    zPipe: float
        wastewater pipe depth below grouond surface
    rPipe: float
        pipe's radius
    spae: float
        space in trechch beside and below the pipe.
    soil_props: pd.DataFrame
        soil properties, for the properties of the trench
    """
    xleft, xright = xPipe - space, xPipe + space
    zBot = zPipe - rPipe - space

    # First shape: the actual trench
    pths = [Path(
            np.array([[xleft,  zTop], [xleft,  zBot],
                      [xright, zBot], [xright, zTop]])
            )] # list

    data = pd.DataFrame(soil_props.loc['trench']).T
    data['name']  ='trench'
    data['NAPtop'] = zTop
    data['NAPbot'] = zBot

    # Add the pipe, generate a circle
    alf = np.arange(0, 361, 15) * np.pi / 180.

    # Add circular path
    pths.append(Path(
                np.vstack((xPipe + rPipe * np.cos(alf), zPipe + rPipe * np.sin(alf))).T))
    pipe_df = pd.DataFrame(pd.Series(
        {'name':'pipe', 'NAPtop':zPipe + rPipe, 'NAPbot': zPipe - rPipe,
         'color':'brown', 'kh':0., 'kv':0., 'por':0., 'Sy':0., 'Ss':0.})).T

    # Add block path to allow setting boundary condition
    pths.append(Path(np.array([[xleft, xleft, xright, xright],
                              [zBot + space, zBot, zBot, zBot + space]]).T))

    # Add a block data
    ghb_df =  pd.DataFrame(soil_props.loc['trench']).T
    ghb_df['name']  = 'ghb'
    ghb_df['color'] = 'blue'
    ghb_df['alpha'] = 0.3
    ghb_df['NAPtop'] = zBot + space
    ghb_df['NAPbot'] = zBot

    # merge the three shapes:
    data = pd.merge(pd.merge(data, pipe_df, how='outer'), ghb_df, how='outer')

    return sht.Shape_df(paths_list=pths, data=data)



def generate_grid(boring=None, trench=None, x=None, z=None, zrefine=None, dzmult=1.5, zlog=-30):
    """Gerneate the model network returning Grid Object.

    The grid is generated using x and z and inserting coordinates of the
    trench walls and bottom and the tops and bottoms of the layers

    Parameters
    ----------
    layers: pd.DataFrame (from Boring.layers)
        the soil layer properties (from given borehole)
    b: float [m]
        half-width of the model
    d: np.ndarray of x-values
        x-values of the grid lines
        The trench walls will be inserted.
    z: np.ndarray (decreasing)
        z-values of the grid lines
        layer top and bottoms will be inserted as is the trench bottom
        Use ztop = expected groundwater table elevation.
    zrefine: float or None for no refinement
        Desired layer thickness for depth below the bottom of the top layer
        for use in MT3D and Seawat only. Use None for MODLOW (no refinement)
    dzmult: float
        factor increment of layer thickness for layer with zbot < -200 to prevent too many layers
    zlog: float
        elevation below which the refinement will be logarithmic instead of linear
    """
    z = np.asarray(z, dtype=float)
    x = np.unique(np.hstack((trench.bbox.x0, trench.bbox.x1, x)))
    y = [-0.5, 0.5]
    ztops = boring.layers['NAPtop'].values
    zbots = boring.layers['NAPbot'].values
    z = np.unique(np.hstack((ztops, zbots, trench.bbox.y0, z)))
    z = z[AND(z < ztops[0], z >= zbots[-1])] # excludes ground surface
    # Refine velow ztops[0] with the desired dz for
    if zrefine is not None:
        znew = np.array([])
        # Refining only from layer 1 (skip layer 0):
        Zt = boring.layers['NAPtop'].values[1:]
        Zb = boring.layers['NAPbot'].values[1:]
        for zt, zb in zip(Zt, Zb):
            dz = zt - zb
            if dz >= 2 * zrefine:
                if zb > zlog:
                    znew = np.hstack((znew, np.linspace(zt, zb, int((zt - zb) // zrefine + 1))[1:-1]))
                else:
                    n = int(np.ceil((np.log(zt - zb) - np.log(zrefine))/np.log(dzmult)))
                    z_ = znew[-1] - np.logspace(np.log10(zrefine), np.log10(zt-zb), n)
                    znew = np.hstack((znew, z_))[1:-1]
        z = np.unique(np.hstack((z, znew)))
    return Grid(x, y, z[::-1])


def stress_period_data_from_excel(workbook_name, sheet_name='SPD', gr=None):
    """Return dict of stress period data from excel.

    Parameters
    ----------
    spd_df: pd.DataFrame
        stress period data from Excel

    gr: fdm.Grid object holding the network
        the model network
    """
    spd = pd.read_excel(workbook_name, sheet_name, index_col='PER', engine='openpyxl')

    spd.columns = [col.lower() for col in spd.columns]

    for col in spd.columns:
        if col[0] in 'ijklmn':
            spd[col] = spd[col].astype(int)
        else:
            spd[col] = spd[col].astype(float)

    spd['surf'] = gr.z[0] - spd['surfd'] # change from depth to elevation.

    return spd


def default_parameters_from_excel(workbook_name=None, sheet_name=None):
    """Get model parameters from Excel workbook."""
    params = pd.read_excel(workbook_name,
                           sheet_name=sheet_name,
                           usecols=[0, 1, 2],
                           dtype={2:object},
                           engine='openpyxl').dropna(how='all')

    # Turn nan into None
    for k in params.index:
        value = params['Value'][k]
        try:
            if np.isnan(value):
                value = None
        except:
            pass
        try:
            if isinstance(value, bool) or k[0].lower in 'ijklmn':
                value = int(value)
        except:
            pass
        params['Value'][k] = value

    # Make a dict of the parameters per package in the data
    packages = np.unique(params['Package'].values)
    params_dict = {}
    for pck in packages:
        params_dict[pck] = params[['Parameter', 'Value']].loc[params['Package'] == pck]
        params_dict[pck] = dict(pd.Series(params_dict[pck]['Value'].values,
                                          index=params_dict[pck]['Parameter']))
    return params_dict


def check_parameters(params_dict):
    """Verify that all parameters have values.

    Parameters
    ----------
    params_dict of dicts where dict[package_nane] is a dict of parameter-value combinations.:
        dict of dict of parameter-value combinations with

    Print warnings if one or more parameters have not been given vallues.
    """
    lines = []
    wmax = 0
    for pck in params_dict:
        pp=params_dict[pck] # dict of parameters pertaining to parackage par
        for par in pp:
            if pp[par] is None:
                lines.append((pck, par))
                wmax = max(len(par), wmax)

    if len(lines)>0:
        print("These {} parameters have not been set:".format(len(lines)))
        fmt = '{{:{:d}s}} in package {{:4s}}'.format(wmax)
        for pck, par in lines:
            print(fmt.format(par, pck))
        print()
    return None


def add_trench_props_to_layers(layers=None, trench=None, soil_props=None, trench_name='TRENCH'):
    """Return layers with soil properties of trench soil appended.

    Pamameters
    ----------
    layers: the DataFrame 'layers' form Boring object
        the layers with properties and 'path'
    trench: Trench object
        the trench to include in the layers object
    soil_props: pd.Series
        soil properties of the material in the trench.
        Must contain the same soil properties as in the layers.

    Example
    -------
    newlayers = add_trench_props_to_layers(borehole['DKMG101'].layers, trenches[21], soil_props.loc['zand'])
    """
    layers = layers.copy()
    trench_props = soil_props.copy()
    trench_props['name'] = trench_name
    trench_props['path'] = trench.trench_path
    return layers.append(trench_props)


def get_mflow_parameters(workbook_name=None, sheet_name=None,
                         layers=None, new_layers=None, trench=None, gr=None):
    """Return dict with the arrays required by MODFLOW.

    Parameters
    ----------
    workbook_name: str
        excel workbook met sheet 'MFLOW' having the default parameters
    sheet_name str
        name of the spreadsheet within workbook that hold the modflow parameters
    layers : Boringen.layers object
        properties of layers of drilling hole representative for this site.
        This is a pd.DataFrame with for which each record represents a soil layer
        whose extent is obtained from the file 'layer_patch' using the get_path()
        method of the patch object that shows the layer.
    new_layers:  a list of dicts or pd.Series with layer properties and a path
        field.
        These layers will be appended to the layers to be included in the
        parameters construction. Newlayes may be contain trench_soilprops
        and the soiprops of deeper layer or of any space defined by the
        path in the 'path' field. Notice that the order is importand as parts
        of the parameter arrays will be overwritten by subsequent layers.
    trench: Trench object
        defines the trench. But it's soil_properties must already be in new_layers.
    gr: Grid object
        the modelflow network
    """
    # We will associate patches.get_path() with properties, in the layers DataFrame

    # Turn the layers DataFrame into a sht.Shape_df object. For this to work
    # The dataframe is supposed to have a file 'layer_path' that is a patch
    # showing the layer as a rectangle.

    #layers['path'] = [p.get_path() for p in layers['layer_patch']
    layers_shdf = sht.Shape_df(data=layers.append(new_layers))

    NODENR = layers_shdf.get_id_array(gr.XM, gr.ZM)

    par = default_parameters_from_excel(workbook_name, sheet_name=sheet_name)

    spd = stress_period_data_from_excel(workbook_name, sheet_name='SPD', gr=gr)

    if spd.index[0] == 1:
        spd.index = spd.index - 1

    # The packages
    dis = par['dis']
    bas = par['bas']
    lpf = par['lpf']
    upw = par['upw']
    rch = par['rch']
    evt = par['evt']
    oc  = par['oc' ]

    # Unit to write the cbc flows to
    for pck in['rch', 'evt']:
        par[pck]['ipakcb'] = lpf['ipakcb']

    #==========================================================================
    dis.update({
        'nlay': gr.nz,
        'nrow': gr.ny,
        'ncol': gr.nx,
        'delr': gr.dx,
        'delc': gr.dy,
        'top':  gr.Z[0],
        'botm': gr.Z[1:],
        'laycbd':gr.LAYCBD,
        'nper': spd.index[-1] + 1,
        'perlen': spd['perlen'].values,
        'nstp': np.asarray(spd['nstp'].values, dtype=int),
        'tsmult': spd['tsmult'].values,
        'steady': np.asarray(spd['steady'].values, dtype=int),
        })

    #==========================================================================
    L = np.ones((gr.ny, gr.nx))
    rch['rech'] = {sp: spd['rech'].loc[sp] * L for sp in spd.index}

    evt.update({
        'evtr': {sp: spd['evtr'].loc[sp] * L for sp in spd.index},
        'surf': {sp: spd['surf'].loc[sp] * L for sp in spd.index},
        'exdp': {sp: spd['exdp'].loc[sp] * L for sp in spd.index},
    })

    oc['stress_period_data'] = {(sp, 0) :
                ['save head', 'save budget'] for sp in spd.index}

    #==========================================================================
    lpf.update({ # this now includes the trench soil properties
        'hk':  layers_shdf.fill_array(gr.XM, gr.ZM, parameter='kh'),
        'vka': layers_shdf.fill_array(gr.XM, gr.ZM, parameter='kv'),
        'sy':  layers_shdf.fill_array(gr.XM, gr.ZM, parameter='Sy'),
        'ss':  layers_shdf.fill_array(gr.XM, gr.ZM, parameter='Ss'),
    })

    if lpf['thickstrt']:
        laytyp = np.zeros(gr.nz, dtype=int)
        laytyp[0] = -1
        lpf.update(laytyp=laytyp)


    upw.update({ # this now includes the trench soil properties
        'hk':  layers_shdf.fill_array(gr.XM, gr.ZM, parameter='kh'),
        'vka': layers_shdf.fill_array(gr.XM, gr.ZM, parameter='kv'),
        'sy':  layers_shdf.fill_array(gr.XM, gr.ZM, parameter='Sy'),
        'ss':  layers_shdf.fill_array(gr.XM, gr.ZM, parameter='Ss'),
    })


    #==========================================================================
    bas.update({
         'strt':   np.zeros(gr.shape),
         'ibound': np.ones(gr.shape, dtype=int)})

    for iL, ibound_layer in zip(layers.index, layers['ibound']):
        lay = layers.loc[iL]

        bas[  'strt'][:, :, 0:1][NODENR[:, :, 0:1] == iL] = lay['strthd']
        bas[  'strt'][:, :, -1:][NODENR[:, :, -1:] == iL] = lay['strthd']
        bas['ibound'][:, :, 0:1][NODENR[:, :, 0:1] == iL] = ibound_layer
        bas['ibound'][:, :, -1:][NODENR[:, :, -1:] == iL] = ibound_layer

    #==== Update ibound in the trench (GHB_block)==============================
    XZ = np.vstack((gr.XM.ravel(), gr.ZM.ravel())).T

    # Set trench to fixed head.
    ibound_trench = trench.soil_props['ibound']
    strthd_trench = trench.soil_props['strthd']
    in_trench = trench.trench_path.contains_points(XZ).reshape(gr.shape)

    # Set head in trench equal to elevation of cell center (= seepage face)
    bas['ibound'][in_trench] = ibound_trench
    if False:
        bas[  'strt'][in_trench] = strthd_trench
    else:
        bas[  'strt'][in_trench] = gr.ZM[in_trench]

    # Make the pipes inactive using their patch stored in trench
    #for pp in trench.pipe_patch:
    #    pth    = pp.get_path()
    #    transf = pp.get_transform()
    #    cpth   = transf.transform_path(pth)
    #    bas['ibound'].ravel()[cpth.contains_points(XZ)] = 0

    return par # return the updated dict of dicts

def get_mt3d_parameters(workbook_name=None, sheet_name=None, layers=None,
            new_layers=None, trench=None, tds_profile=None, gr=None, ibound=None):
    """Return dict with the arrays required by MODFLOW.

    Parameters
    ----------
    workbook_name: str
        excel workbook met sheet 'MFLOW' having the default parameters
    sheet_name: str
        name of the spreadsheet in workbook with the MT3D and seawat parameters
    layers : Boringen.layers object
        properties of layers of drilling hole representative for this site.
        This is a pd.DataFrame with for which each record represents a soil layer
        whose extent is obtained from the file 'layer_patch' using the get_path()
        method of the patch object that shows the layer.
    new_layers:  a list of dicts or pd.Series with layer properties and a path
        field.
        These layers will be appended to the layers to be included in the
        parameters construction. Newlayes may be contain trench_soilprops
        and the soiprops of deeper layer or of any space defined by the
        path in the 'path' field. Notice that the order is importand as parts
        of the parameter arrays will be overwritten by subsequent layers.
    trench: Trench object
        defines the trench. But it's soil_properties must already be in new_layers.
    tds_profile: Salinity_profile object
        the salinity profile to be used
    gr: fdm.Grid object
        the modflow network
    ibound: np.ndarray dtype int
        the boundary array used with MODFLOW
    """
    layers_shdf = sht.Shape_df(data=layers.append(new_layers))

    par = default_parameters_from_excel(workbook_name, sheet_name='MT3D')

    spd = stress_period_data_from_excel(workbook_name, sheet_name='SPD', gr=gr)

    if spd.index[0] == 1:
        spd.index = spd.index - 1

    adv = par['adv']
    btn = par['btn']
    dsp = par['dsp']
    rct = par['rct']
    ssm = par['ssm']
    vdf = par['vdf']
    gcg = par['gcg']

    #==========================================================================
    btn.update({
        'nlay': gr.nz,
        'nrow': gr.ny,
        'ncol': gr.nx,
        'nper': spd.index[-1] + 1,
        'delr': gr.dx,
        'delc': gr.dy,
        'htop': gr.Z[0],
        'dz':   gr.DZ,
        'prsity': layers_shdf.fill_array(gr.XM, gr.ZM, parameter='por'),
        'icbund': ibound,
        'sconc': tds_profile.get_conc(gr=gr),
        'perlen': spd['perlen'].values,
        'nstp':  np.asarray(spd['nstp'].values, dtype=int),
        'tsmult': spd['tsmult'].values,
        'ssflag': ['T' if flag != 0 else 'F' for flag in spd['steady'].values],
        })

    #==========================================================================
    dsp.update({
        'al': layers_shdf.fill_array(gr.XM, gr.ZM, parameter='al'),
        'trpt': np.ones(gr.nz) * par['dsp']['trpt'],
        'trpv': np.ones(gr.nz) * par['dsp']['trpv'],
        'dmcoef': (np.ones(gr.nz) * par['dsp']['dmcoef']).reshape((gr.nlay, 1))
        })

    #==========================================================================

    rct.update({
        'rhob': layers_shdf.fill_array(gr.XM, gr.ZM, parameter='rhob_dry')
            })

    ssm.update({
        'dtype': None,
        'stress_period_data': None,
        })

    #==========================================================================

    vdf.update({})

    gcg.update({})


    return par # return the updated dict of dicts



def modflow(dirs=None, case=None, par=None, bdd=None):
    """Set up the actual MODFLOW Model and run it.

    Parameters
    ----------
    dirs: Dirs object
        directory structure
    par: dict
        parameters
    spd: dict
        stress period data
    bdd: dict
        boundary data
    """
    dirs.cwd()

    exe = dirs.executables['mflow']

    model= flopy.modflow.Modflow(modelname=case, exe_name=exe,
                          model_ws=dirs.case_results, verbose=True)

    #MODFLOW packages
    packages = {}

    oc  = flopy.modflow.ModflowOc(model,
        stress_period_data=par['oc']['stress_period_data'], compact=par['oc']['compact'])

    dis = flopy.modflow.ModflowDis(model, **par['dis'])
    bas = flopy.modflow.ModflowBas(model, **par['bas'])
    rch = flopy.modflow.ModflowRch(model, **par['rch'])
    evt = flopy.modflow.ModflowEvt(model, **par['evt'])
    packages.update({'dis': dis, 'bas':bas, 'rch':rch, 'evt':evt, 'oc':oc})
    if 'usg' in exe:
        upw = flopy.modflow.ModflowUpw(model, **par['upw'])
        sms = flopy.modflow.ModflowSms(model, **par['sms'])
        packages.update({'upw':upw, 'sms':sms})
    else:
        lpf = flopy.modflow.ModflowLpf(model, **par['lpf'])
        pcg = flopy.modflow.ModflowPcg(model, **par['pcg'])
        lmt = flopy.modflow.ModflowLmt(model, output_file_name='mt3d_link.ftl')
        packages.update({'lpf':lpf, 'pcg':pcg, 'lmt':lmt})


    # Not used now:
    #ghb = flopy.modflow.ModflowGhb(model, stress_period_data=bdd['GHB'], ipakcb=par['ipakcb'])
    #riv = flopy.modflow.ModflowRiv(model, stress_period_data=bdd['RIV'], ipakcb=par['ipakcb'])
    #drn = flopy.modflow.ModflowDrn(model, stress_period_data=bdd['DRN'], ipakcb=par['ipakcb'])
    #wel = flopy.modflow.ModflowWel(model, stress_period_data=bdd['WEL'], ipakcb=par['ipakcb'])
    #packages.update({'ghb':ghb, 'riv':riv, 'drn':drn, 'wel':wel})

    print('Pakages used:''[{}]'.format(', '.join(packages.keys())))

    # write data and run modflow
    model.write_input()
    success, mfoutput = model.run_model(silent=False, report=True, pause=False)

    print('Running success = {}'.format(success))
    if not success:
        raise Exception('MODFLOW did not terminate normally.')

    return model


class HDS_obj:
    """Heads object, to store and plot head data."""

    def __init__(self, case, gr=None):
        """Return Heads_obj.

        Paeameters
        ----------
        dirs: Dir_struct object
            GGOR directory structure
        gr: fdm.mfgrid.Grid object
            holds the Modflow grid.
        """
        self.gr = gr
        #% Get the modflow-computed heads
        hds_file = case +'.hds'
        print("\nReading binary head file '{}' ...".format(hds_file))
        HDS = bf.HeadFile(hds_file)

        self.ncol = HDS.ncol
        self.nrow = HDS.nrow
        self.nlay = HDS.nlay
        self.shape = HDS.nlay, HDS.nrow, HDS.ncol
        self.kstpkper = HDS.kstpkper
        self.times = HDS.times
        self.heads = HDS.get_alldata()
        self.heads[np.logical_or(self.heads<-500, self.heads>500)] = np.nan


    def get_water_table(self, it=-1):
        """Return the water table."""
        return self.heads[it][0, 0, :]


    def plot_water_table(self, it=-1, gr=None, ax=None, color='blue', **kw):
        """Plot the water table.

        Parameters
        ----------
        it: int
            "Stress period number or zero based entry in outputs if nstp was > 1"
        color: str
            Color for the water table.
        kw: dict
            all kw are passed on to the ax.plot(.... **kw)
        """
        ax.plot(gr.xm, self.heads[it][0, 0, :], color=color, label='water_table', **kw)


    def contour(self, it=None, gr=None, ax=None, **kw):
        """Contour the heads for a given stress period."""
        if it is None:
            ValueError("You must specify it.")
        hdl = ax.contour(gr.XM[:, 0, :], gr.ZM[:, 0, :], self.heads[it][:, 0, :], **kw)
        return hdl


    def contourf(self, it=None, gr=None, ax=None, **kw):
        """Plot filled contour the heads for a given stress period."""
        if it is None:
            ValueError("You must specify it.")
        hdl = ax.contourf(gr.XM[:, 0, :], gr.ZM[:, 0, :], self.heads[it][:, 0, :], **kw)
        return hdl


    def get_pressures(self, layers=None, trench=None, gr=None):
        """Return ther array of active_pressures (expressed in m H2O.

        Parameters
        ----------
        layers: the layrs object from a borehole at the horizontal cell faces
            the layer properties
        trench: the used trench object.
            the trench with its soil_properties
        gr: fdm.Grid object
            the modeflow network.
        """
        g  = 9.81 # N/kg
        rhow = 1000 # kg/m3 water density

        XZ = np.vstack((gr.XM.ravel(), gr.ZM.ravel())).T

        hw = self.heads[-1].copy()
        hw.ravel()[trench.trench_path.contains_points(XZ)] = 0.
        # Compute total pressures
        rhob_wet = gr.const(0.)

        Zsoil = gr.Z.copy()
        Zsoil[0] = layers.loc[layers.index[0], 'NAPtop']

        for iL in layers.index:
            lay = layers.loc[iL]
            rhob_wet.ravel()[lay['path'].contains_points(XZ)] = lay['rhob_wet']

        # Points along top inside trench also have pressure zero
        in_trench = trench.trench_path.contains_points(XZ).reshape(gr.shape)
        rhob_wet[in_trench] = trench.soil_props['rhob_wet']

        ptot = np.zeros_like(Zsoil)
        ptot[1:] = np.cumsum(g * rhob_wet * np.abs(np.diff(Zsoil, axis=0)), axis=0)

        # Compute water pressures
        hw = np.zeros_like(Zsoil)
        hw = np.concatenate((np.zeros_like(Zsoil[0:1]), self.heads[-1]), axis=0)
        hw[1:-1, :, :] = 0.5 * (self.heads[-1, :-1] + hw[2:])

        pw = rhow * g * (hw - Zsoil)

        peff = ptot - pw
        return peff / (rhow * g), ptot / (rhow * g), pw / (rhow * g), Zsoil




class CBC_obj:
    """Cell by cell flows object."""

    def __init__(self, case, gr=None):
        """Return Cell-by-cell flow object.

        Paeameters
        ----------
        dirs: Dir_struct object
            GGOR directory structure
        IBOUND: ndarray of gr.shape
            Modflow's boundary array
        gr: fdm.mfgrid.Grid object
            holds the Modflow grid.
        """
        cbc_labels = {'CHD': b'   CONSTANT HEAD',
                      'FRF': b'FLOW RIGHT FACE ',
                      'FFF': b'FLOW FRONT FACE ',
                      'FLF': b'FLOW LOWER FACE ',
                      'RCH': b'        RECHARGE',
                      'EVT': b'              ET',
                      'GHB': b'HEAD DEP BOUNDS',
                      'RIV': b'  RIVER LEAKAGE',
                      'DRN': b'         DRAINS',             # also used for trenches.
                      'STO': b'        STORAGE',
                      'WEL': b'          WELLS'}
        compact_labels = ['CHD', 'GHB', 'RIV', 'DRN', 'WEL']


        self.gr = gr

        #% Get the modflow-computed heads
        cbc_file = case +'.cbc'
        print("\nReading binary cbc file '{}' ...".format(cbc_file))
        CBC = bf.CellBudgetFile(cbc_file)

        self.nper = CBC.nper
        self.nlay = CBC.nlay
        self.nrow = CBC.nrow
        self.ncol = CBC.ncol
        self.nrecords = CBC.nrecords
        self.shape = CBC.nlay, CBC.nrow, CBC.ncol
        self.labels = CBC.textlist
        self.kstpkper = CBC.kstpkper
        self.times = CBC.times

        self.cbc = dict()
        for lbl in cbc_labels:
            if cbc_labels[lbl] in self.labels:
                self.cbc[lbl] = np.zeros((len(self.kstpkper), *gr.shape))
                data = CBC.get_data(text=cbc_labels[lbl])
                if lbl in compact_labels:
                    for it, stp in enumerate(self.kstpkper):
                        self.cbc[lbl][it].ravel()[data[it]['node'] - 1] = data[it]['q']
                elif lbl in ['RCH', 'EVT']:
                    for it, stp in enumerate(self.kstpkper):
                        I = gr.NOD[0] + data[it][0] - 1
                        self.cbc[lbl][it].ravel()[I] = data[it][1].ravel()
                else:
                    for it, stp in enumerate(self.kstpkper):
                        self.cbc[lbl][it] = data[it]

    def stream(self, it=None, row=0):
        """Return stream function for cross section defined by given row.

        Parameters
        ----------
        row: int
            the row number defining the cross section for the stream function
        """
        if it is None:
            raise ValueError("You must specify it.")
        # start with lowerface, shape(nz+1, nx-1)
        strm = np.zeros((self.nlay+1, self.ncol - 1))

        strm[:-1, :] += self.cbc['FRF'][it, :, row, :-1][::-1].cumsum(axis=0)[::-1]

        return strm

    def contour_streamf(self, it=None, gr=None, ax=None, row=0, **kw):
        """Plot the stream function for the cross section defined by row.

        Parameters
        ----------
        levels: int or sequence_like
            the stream function levels [m2/d] or [L2/T]
        ax: plt.Axes object
            axes to plot the contours
        row: int
            row number defining the cross section (make sure no flow perpendicular
                                                   to the row to be valid).
        """
        if it is None:
            raise ValueError("You must specify it.")
        hdl = ax.contour(gr.Xp, gr.Zp, self.stream(it=it, row=row), **kw)
        return hdl # returns handle to contours

    def contourf_streamf(self, it=None, gr=None, ax=None, row=0, **kw):
        """Plot the stream function for the cross section defined by row.

        Parameters
        ----------
        levels: int or sequence_like
            the stream function levels [m2/d] or [L2/T]
        ax: plt.Axes object
            axes to plot the contours
        row: int
            row number defining the cross section (make sure no flow perpendicular
                                                   to the row to be valid).
        """
        if it is None:
            raise ValueError("You must specify it.")
        hdl = ax.contour(gr.XP, gr.ZP, self.stream(it=it, row=row), **kw)
        return hdl # returns handle to contours


#%% Mt3dms and Seawat


def mt3dms(dirs=None, case=None, par=None, spd=None, bdd=None, modflowmodel=None):
    """Set up the actual MT3MS Model and run it.

    Parameters
    ----------
    dirs: Dirs object
        directory structure
    par: dict
        parameters
    spd: dict
        stress period data
    bdd: dict
        boundary data
    """
    dirs.cwd()

    model  = flopy.mt3d.Mt3dms(modelname=case, exe_name=dirs.executables['mt3d'],
                model_ws=dirs.wd, external_path=None, modflowmodel=modflowmodel, verbose=True)

    adv = flopy.mt3d.Mt3dAdv(model, **par['adv'])
    btn = flopy.mt3d.Mt3dBtn(model, **par['btn'])
    dsp = flopy.mt3d.Mt3dDsp(model, **par['dsp'])
    gcg = flopy.mt3d.Mt3dGcg(model, **par['gcg'])
    #rct = flopy.mt3d.Mt3dRct(model, **par['rct'])
    ssm = flopy.mt3d.Mt3dSsm(model, **par['ssm'])

    packages = {'adv': adv, 'btn': btn, 'dsp': dsp, 'gcg': gcg, 'ssm': ssm } #'rct': rct}
    print('Pakages used:''[{}]'.format(', '.join(packages.keys())))

    model.write_input()
    success, mtoutput = model.run_model(silent=False, report=True, pause=False)

    with open(os.path.join(dirs.wd, case + '.list'), 'r') as fp:
        _ = fp.seek(0, 2)
        _ = fp.seek(fp.tell() - 200)
        line = fp.readlines()[-2]
        success = 'END OF MODEL' in line

    print('Running success = {}'.format(success))
    if not success:
        raise Exception('Mt3d did not terminate normally.')


def seawat(dirs=None, case=None, par_mf=None, par_mt=None, bdd=None, modflowmodel=None):
    """Set up the actual MODFLOW Model and run it.

    Parameters
    ----------
    dirs: Dirs object
        directory structure
    par: dict
        parameters
    spd: dict
        stress period data
    bdd: dict
        boundary data
    """
    dirs.cwd()

    model  = flopy.seawat.Seawat(modelname=case, exe_name=dirs.executables['seawat'],
                     model_ws=dirs.wd, verbose=True)

    #MODFLOW packages
    dis = flopy.modflow.ModflowDis(model, **par_mf['dis'])
    bas = flopy.modflow.ModflowBas(model, **par_mf['bas'])
    lpf = flopy.modflow.ModflowLpf(model, **par_mf['lpf'])

    #ghb = flopy.modflow.ModflowGhb(model, stress_period_data=bdd['GHB'], ipakcb=par_mf['ipakcb'])
    #riv = flopy.modflow.ModflowRiv(model, stress_period_data=bdd['RIV'], ipakcb=par_mf['ipakcb'])
    #drn = flopy.modflow.ModflowDrn(model, stress_period_data=bdd['DRN'], ipakcb=par_mf['ipakcb'])
    #wel = flopy.modflow.ModflowWel(model, stress_period_data=bdd['WEL'], ipakcb=par_mf['ipakcb'])

    rch = flopy.modflow.ModflowRch(model, **par_mf['rch'])
    evt = flopy.modflow.ModflowEvt(model, **par_mf['evt'])
    pcg = flopy.modflow.ModflowPcg(model, mxiter=200, iter1=200, hclose=0.001, rclose=0.001)
    #sms = flopy.modflow.ModflowSms(model, **par['sms'])
    oc  = flopy.modflow.ModflowOc(model,
        stress_period_data=par_mf['oc']['stress_period_data'], compact=par_mf['oc']['compact'])

    packages = {'dis': dis, 'bas':bas, 'lpf':lpf,'rch':rch, 'evt':evt,
                'oc':oc, 'pcg':pcg}

    #MT3D Packages
    adv = flopy.mt3d.Mt3dAdv(model, **par_mt['adv'])
    btn = flopy.mt3d.Mt3dBtn(model, **par_mt['btn'])
    dsp = flopy.mt3d.Mt3dDsp(model, **par_mt['dsp'])
    gcg = flopy.mt3d.Mt3dGcg(model, **par_mt['gcg'])
    #rct = flopy.mt3d.Mt3dRct(model, **par_mt['rct'])
    ssm = flopy.mt3d.Mt3dSsm(model, **par_mt['ssm'])

    # Seawat packages
    vdf = flopy.seawat.SeawatVdf(model, **par_mt['vdf'])

    packages.update({'adv': adv, 'btn': btn, 'dsp': dsp, 'gcg': gcg, 'ssm': ssm,
                     'vdf': vdf})

    print('Pakages used:''[{}]'.format(', '.join(packages.keys())))


    model.write_input()
    success, mtoutput = model.run_model(silent=False, report=True, pause=False)

    # Look at the end of the list ifle to see if the programme ended normally
    #with open(os.path.join(dirs.wd, case + '.list'), 'r') as fp:
    #    _ = fp.seek(0, 2)
    #    _ = fp.seek(fp.tell() - 200)
    #    line = fp.readlines()[-2]
    #    success = 'END OF MODEL' in line

    print('Running success = {}'.format(success))
    if not success:
        raise Exception('Seawat did not terminate normally.')


class UCN_obj:
    """Concentration object, to store and plot concentrations from MT3DMS or Seawat."""

    def __init__(self, dirs, gr=None, speciesNr=1):
        """Return Heads_obj.

        Paeameters
        ----------
        dirs: Dir_struct object
            GGOR directory structure
        gr: fdm.mfgrid.Grid object
            holds the Modflow grid.
        speciesNr: int (default 1)
            MT3DMS species number
        """
        self.gr = gr
        #% Get the modflow-computed heads
        fname = 'MT3D{:03d}.UCN'.format(speciesNr)
        ucn_file = os.path.join(dirs.case_results, fname)
        print("\nReading concentration file '{}' ...".format(fname))
        UCN = bf.UcnFile(ucn_file)

        self.text = UCN.text
        self.filename = UCN.filename
        self.ncol = UCN.ncol
        self.nrow = UCN.nrow
        self.nlay = UCN.nlay
        self.shape = UCN.nlay, UCN.nrow, UCN.ncol
        self.kstpkper = UCN.kstpkper
        self.times = UCN.times
        self.conc = UCN.get_alldata()
        self.conc[np.logical_or(self.conc<-1e10, self.conc>1e10)] = np.nan

    def contour(self, it=None, gr=None, ax=None, **kw):
        """Contour the concentration for a given stress period."""
        if it is None:
            ValueError("You must specify it.")
        hdl = ax.contour(gr.XM[:, 0, :], gr.ZM[:, 0, :], self.conc[it][:, 0, :], **kw)
        return hdl


    def contourf(self, it=None, gr=None, ax=None, **kw):
        """Plot filled contour the concentration for a given stress period."""
        if it is None:
            ValueError("You must specify it.")
        hdl = ax.contourf(gr.XM[:, 0, :], gr.ZM[:, 0, :], self.conc[it][:, 0, :], **kw)
        return hdl


    #%% __main__

if __name__ == '__main__':

    #%% Get the data

    # Parameters to generate the model. We'll use this as **kwargs

    # Get the data from a cross section
    dirs = Dir_struct(home=home, case_folder='Kruiszwin_1',
        executables={'mflow':'mf2005.mac', 'mt3d': 'mt3dms5b.mac', 'seawat':'swt_v4.mac'})

    workbook   = os.path.join(dirs.data, "Julianadorp.xlsx")
    layers_df  = pd.read_excel(workbook, sheet_name='Boringen', engine="openpyxl")
    piez_df    = pd.read_excel(workbook, sheet_name='Peilbuizen',
                            skiprows=[1], index_col=0, engine="openpyxl")
    soil_props = pd.read_excel(workbook, sheet_name='Grondsoort',
                               index_col=0, engine="openpyxl")
    spd_df     = pd.read_excel(workbook, sheet_name='SPD', index_col=0, engine="openpyxl")


    np.unique(layers_df['name'])

    # Get piezometers object (could also be a list)
    piezoms = Piezometers(piez_df)
    piezoms.plot() # show piezometes

    # Make a list of Boring objects and reset their index
    borehole = dict()
    borehole_names = np.unique(layers_df['name'])
    for name in borehole_names:
        borehole[name] = Boring(name, layers_df.loc[layers_df['name'] == name].copy())
        borehole[name].layers.index = np.arange(len(borehole[name].layers.index), dtype=int)
        # We still have to run method make_layer_patches_and_paths(xlim=(xmin,xmax))
        # Before we can plot the layers. We do than when we know the grid.


    plot_bores(borehole, title_fontsize=9, ylim=(-16, 2)) # Show boreholes

    print(soil_props)

    #%% trenches

    trenches = Trench_collection(os.path.join(GIS, 'punten.shp'),
                      os.path.join(GIS, 'Kruiswin12(345)_lijn.shp'), mv=0.76)

    ax = newfig("Pipes","x","y", size_inches=(12, 17))
    trenches.plot_pipes(ax=ax)
    trenches.plot_points(ax=ax)
    ax.set_aspect(1.)


    trenches.plot_trenches(ylim=(-2, 1))

    #%% # Show section with trench and pipe based on borehole with given name

    print(borehole.keys())

    print(borehole[name].layers)

    # Need gr to plot this. gr is defined further down.
    # name = 'DKMG010'
    # for itr in range(10,20):
    #     ax = kzw.newfig("Profile, trench {}, borehole={}".format(itr, name), "x-section", "NAP")
    #     kzw.plot_section(boring=borehole[name], trench=trenches.trenches[itr], xlim=gr.x[[0, -1]], ylim=(-16, 2), ax=ax)

    #%% Salinities

    tds_profiles = Salinity_profile_collection(workbook, sheet_name='Salinities', coords_sheet='Coords')

    ax = newfig("Salinities", "TDS [mg/L]", "NAP")

    tds_profiles.plot(ax=ax)
    ax.legend()


    # Using the individual profiles

    ax = newfig("Salinities", "TDS [mg/L]", "NAP")
    names = ['DKMG103', 'DKMG107']

    # Show these specific profiles
    tds_profiles.plot(names=names, ax=ax)

    # Interpolate and show them on top of the previous ones
    zdata=np.linspace(1, -14, 61)
    for name in names:
        ZTDS = tds_profiles[name].interp(zdata=np.linspace(1, -14, 61))
        ax.plot(ZTDS[0], ZTDS[1], 'o', label="interpolated boring {}".format(name))
    ax.legend()


    #%% Show salinities on/with a section

    names = set(borehole.keys()).intersection(tds_profiles.profiles.keys())
    print(names)


    def twoaxes(name):
        """Plot soil profile with salinity next to it."""
        titles =(f'Bodemopbouw boring {name}', f'Zoutprofiel boring {name}')
        xlabels =('x tov hart sleuf [m]', 'Totaal zoutgehalte [mg/L]')
        ylabels = ('NAP [m]', 'NAP [m]')
        fig, axes = plt.subplots(1, 2, sharey=True)
        fig.set_size_inches(15, 15)
        for ax, title, xlabel, ylabel in zip(axes, titles, xlabels, ylabels):
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid()
        return axes


    itr = 21 # should be a trench near the borehole, this only refers to the trench excavation.
    for name in sorted(names):
        ax0, ax1 = twoaxes(name)
        plot_section(boring=borehole[name], trench=trenches.trenches[itr], ylim=(-16, 2), ax=ax0, aspect='auto',
                        xlim=(-5, 5))
        tds_profiles.plot(names=name, ax=ax1)


    #%% Overview of the trenches at the chosen locations.

    trs = trenches.trenches
    for itr, tr in enumerate(trs):
        xz = tr.trench_path.vertices
        zmin = np.min(xz[:, 1])
        w = np.max(xz[:, 0]) - np.min(xz[:, 0])
        x, y = tr.pm
        d1 = tr.pipes[0]['diam']
        d2 = tr.pipes[1]['diam']
        t1 = tr.pipes[0]['type']
        t2 = tr.pipes[0]['type']
        s = f'trench {itr:2d}, xRD={x:8.0f}, yRD={y:8.0f}, w={w:6.2f}, z_min={zmin:6.2f}, d1={d2:6.0f}/{t1}'
        if tr.ip[0] != tr.ip[1]:
            s += f', d2={d2:6.0f}/{t2} mm'
        print(s)

    #%% ========= The sinulations =============================================

    # The name of the borehole on which the model is based
    use_name = 'DKMG110'

    case_mf = 'justmodflow'
    case_mt = 'justMT3d'
    case_sw = 'justSeawat'

    run_seawat = False

    # Generate the collection of trenches
    trenches = Trench_collection(os.path.join(GIS, 'punten.shp'),
                          os.path.join(GIS, 'Kruiswin12(345)_lijn.shp'), mv=0.76)

    # Choose a trench for use in the model, we use trench 21 here.
    itr = 21 # Trench number to use.
    use_trench = trenches[itr]

    # set its soil properties.
    use_trench.set_soil_props(soil_props.loc['zand_trench'].copy(), trench_name=f'SLEUF_{itr}')

    #%% ========= MODFLOW =====================================================

    # Choose grid line coordinates
    x = np.hstack((np.linspace(0, 5, 21), np.logspace(np.log10(5), np.log10(250), 60)))
    x = np.hstack((-x[::-1], x))
    z = -np.linspace(0.5, 20, 78)

    # rather base Z for MODFLOW on the borehole layers themselves and the trench
    # z = np.unique(np.hstack((borehole[use_name].layers['NAPtop'].values,
    #                borehole[use_name].layers['NAPbot'].values,
    #                borehole[use_name].layers['NAPbot'].values[:-1] + 0.05, # extra yields nicer contours
    #                trench.ghb_path.vertices.T[1])))[::-1]

    # We need the x and z coordinates of our grid lines without considering the trench and the layers exactly
    gr = generate_grid(borehole[use_name], trench=use_trench, x=x, z=[0.5], zrefine=0.25)

    # Because the top aquifer is thin and has a free water table, adapt the top of the grid to this
    # water table. (This is because wetting does not lead to convergence in MODFLOW2005 and, therefor, seawat.)
    # The problem is, of course, that we don't know the water table in advance. But if we have already run the model
    # and have the same grid, it might work to just get the told heads and use these to estimate the water table
    # of the current run. This may require that we run the model several times and verify the results
    try:
        hds = HDS_obj(os.path.join(dirs.case_folder, case_mf), gr=gr)
        Z = gr.Z; Z[0] = np.fmax(hds.get_water_table(), Z[1] + 0.01)
        gr = Grid(gr.x, gr.y, Z) # redefine gr using update top of teh grid mesh.
        # Top and bottom of the water tabel
        print(np.min(gr.Z[0]), np.max(Z[0]))
    except:
        pass

    # Make sure the borehole layer patches match with the width of the model
    for name in borehole:
        borehole[name].mk_layer_patches_and_paths(xlim=(gr.x[0], gr.x[-1]))

    print(gr.Z[0])

    # Get modflow parametres
    par_mf = get_mflow_parameters(workbook_name=workbook, sheet_name='MFLOW',
                            layers=borehole[use_name].layers, trench=use_trench, gr=gr)

    # Ghb points. No ghb (general head boundaries in nodel)
    bdd = {} # no ghb riv of other boundary conditions.

    #%% Check/inspect parameters
    if False:
        # Parameter inspection 0
        print(par_mf.keys())

        # Parameter inspection 1
        par_mf['bas']['ibound'][:, 0, [0,-1]]
        par_mf['bas'][  'strt'][:, 0, [0,-1]]
        par_mf['bas'][  'strt'][:, 0, -1]
        par_mf['bas']['ibound'][:-12, 0, 72:88]
        par_mf['bas']['strt'][:-12, 0, 72:88]

        # Parameter inspection 2
        for k in par_mf.keys():
            print()
            print('key ', k, ':')
            print(par_mf[k].keys())
            print()

        # Parameter inspection 3
        def listpar(pardict, parname):
            """Return a parameter."""
            print(f"{parname}")
            print(str(pardict[parname][:, 0, :]))
            print(f"Unique values for {parname}: ", np.unique(pardict[parname].ravel()))

        listpar(par_mf['bas'], 'ibound')
        listpar(par_mf['bas'], 'strt')
        listpar(par_mf['lpf'], 'hk')
        listpar(par_mf['lpf'], 'vka')
        listpar(par_mf['lpf'], 'ss')
        listpar(par_mf['lpf'], 'sy')

        # Parameter inspection 4
        def showpar(pardict, parname, gr=gr, xlim=None, ylim=None):
            """Plot parameter as image for inspection."""
            ax = newfig(f"{parname}", "x", "z", xlim=xlim, ylim=ylim)
            c = ax.contourf(gr.xm, gr.zm, pardict[parname][:, 0, :])
            plt.colorbar(c,ax=ax)

        kw = {'xlim': (-5, 5), 'ylim': (-16, 2)}
        kw = {'xlim': gr.x[[0, -1]], 'ylim': None}

        showpar(par_mf['bas'], 'ibound', **kw)
        showpar(par_mf['bas'], 'strt', **kw)
        showpar(par_mf['lpf'], 'hk', **kw)
        showpar(par_mf['lpf'], 'vka', **kw)
        showpar(par_mf['lpf'], 'ss', **kw)
        showpar(par_mf['lpf'], 'sy', **kw)


    if False: # skip modflow and mt3d
        #%% MODFLOW
        model = modflow(dirs=dirs, case=case_mf, par=par_mf, bdd=bdd)

        hds = HDS_obj(os.path.join(dirs.case_folder, case_mf), gr=gr)
        cbc = CBC_obj(os.path.join(dirs.case_folder, case_mf), gr=gr)


        it = par_mf['dis']['nper'] - 1
        xlim = gr.x[1], gr.x[-2]

        axs = newfig2h(["Head and stream lines {} for stress t = {:.0f} d, trench_nr {}".format(use_name, hds.times[-1], itr),
                        "Pressures below and next to trench [mH2O]"],
                        ["x [m]", "pressure mH2O"],
                        ["NAP [m]", "NAP [m]"], size_inches=(16, 8), shift_fraction=0.4)


        plot_section(boring=borehole[use_name], trench=use_trench,
                        xlim=gr.x[[0, -1]], ylim=(-16, 2), ax=axs[0], trenchcolor='white')


        hds.plot_water_table(it=it, gr=gr, ax=axs[0])

        Ch = hds.contour(it=it, levels=30, ax=axs[0], gr=gr, colors='black', linewidths=1, linestyles='solid')
        my_hd_levels=[-0.5, -1.0 -1.5, -2, -2.5, -3] # for labels
        ax.clabel(Ch, inline=True, fmt='%.3f', fontsize=10)

        cbc.contour_streamf(it=it, gr=gr, ax=axs[0], levels=40, linewidths=1., linestyles='solid', colors='brown', zorder=2)


        # Plak een witte plakker over de trench
        axs[0].add_patch(PathPatch(use_trench.trench_path, fc='white', zorder=100))


        label_color_dict=({
            'water_table':'darkblue',
            'heads':'black',
            'stream function [m2/d]':'brown',
            'Chloride concentration [g/m3]':'darkred'
            })

        my_legend(ax=axs[0], label_color_dict=label_color_dict, loc='best')

        # axs[0].set_xlim((-10, 10))
        axs[0].set_xlim((-100, 100))
        #ax.set_ylim((-2.5, 1.5))


        # Plot the pressures ========================================================================================

        peff, ptot, pw, Zsoil = pressures = hds.get_pressures(layers=borehole[use_name].layers, trench=use_trench, gr=gr)

        for ix, clr in zip([70, 80], clrs()):
            axs[1].plot(ptot[:, 0, ix], Zsoil[:, 0, ix], '--', color=clr, lw=1, label=f'ptot op x= {gr.xm[ix]:.2f} m]')
            axs[1].plot(peff[:, 0, ix], Zsoil[:, 0, ix],  '-', color=clr, lw=3, label=f'peff op x= {gr.xm[ix]:.2f} m]')
            axs[1].plot(pw[  :, 0, ix], Zsoil[:, 0, ix], '-.', color=clr, lw=2, label=f'pw[  op x= {gr.xm[ix]:.2f} m]')
        axs[1].legend()
        axs[1].set_xlim((-1, 10))

        #axs[0].set_position(axs[1].get_position())

        #%% Water budget directly from cbc

        # Water budget
        ibound = par_mf['bas']['ibound']

        XZ = np.stack((gr.XM.ravel(), gr.ZM.ravel())).T
        L = use_trench.trench_path.contains_points(XZ).reshape(gr.shape)
        Qtr1 = np.sum(cbc.cbc['CHD'][-1][L])
        Qtr2 = np.sum(cbc.cbc['CHD'][-1][ibound==-4])

        print(f'Qtr1 = {Qtr1:10.5f} m2/d\nQtr2 = {Qtr2:10.5f} m2/d\n')

        # Total recharge
        Qrch = np.sum(cbc.cbc['RCH'][-1])
        Qrch = np.sum(cbc.cbc['RCH'][-1][ibound > 0])
        # Print totals CHD per negative value of ibound
        print("Total flow constant head and recharge per negative value of ibound")
        Qtot = 0
        print(f'rch              ={Qrch:10.5f}')
        Qtot += Qrch
        for i in np.unique(ibound)[::-1][1:]:
            Qchd = np.sum(cbc.cbc['CHD'][-1][ibound == i])
            Qtot += Qchd
            print(f'ibound = {i:2d}, Qchd={Qchd:10.5f}')
        print('==============================')
        print(f'Qchd total       ={Qtot:10.5f}')


        # Total recharge (excluding fixed head and inactive nodes)
        rch = 0.0005 * gr.DX; rch[1:] = 0
        np.sum(rch[ibound > 0])

        L = 100.
        Qtrh = Qtr1 * L / 24
        print(f"Qtrench per 100m is {Qtrh:10.4f}")

        #%% ====== MT3DMS =========================================================

        par_mt3d = get_mt3d_parameters(workbook_name=workbook, sheet_name='MT3D',
                                    layers=borehole[use_name].layers,
                                    new_layers=None,
                                    trench=use_trench,
                                    ibound=par_mf['bas']['ibound'],
                                    tds_profile=tds_profiles[use_name],
                                    gr=gr)


        mt3dms(dirs=dirs, case=case_mt, par=par_mt3d, bdd=bdd, modflowmodel=model)


        ucn = UCN_obj(dirs, gr=gr)
        print(ucn.times)


        it = par_mf['dis']['nper'] - 1
        xlim = gr.x[1], gr.x[-2]

        axs = newfig2h(["Totaal zout conentraties {} voor t = {:.0f} d, sleuf_nr {}".format(use_name, ucn.times[-1], itr),
                        "Totaal zout concentratie [mg/L] onder en naast de sleuf"],
                        ["x [m]", "TDS [mg/L]"],
                        ["NAP [m]", "NAP [m]"], size_inches=(16, 8), shift_fraction=0.4)


        plot_section(boring=borehole[use_name], trench=trenches.trenches[itr],
                        xlim=gr.x[[0, -1]], ylim=(-16, 2), ax=axs[0], trenchcolor='white')

        hds.plot_water_table(it=it, gr=gr, ax=axs[0])

        clevels =[500, 1000, 2000, 4000, 8000, 16000, 24000, 32000]

        Cs = ucn.contour(it=it, gr=gr, ax=axs[0], levels=clevels, linewidths=1., colors='darkred', zorder=3)

        ax.clabel(Cs, clevels, inline=True, fmt='%.0f', fontsize=10)

        axs[0].set_xlim(-100, 100)

        # Zoutprofiel onder en naast de sleuf

        tds_profiles[use_name].plot(ax=ax)
        for ix in [80, 136]:
            axs[1].plot(ucn.conc[-1, :, 0, ix], gr.ZM[:, 0, ix], label=f"tds x={gr.xm[ix]:.0f} m")

        axs[1].legend()


    #%% ==== SEAWAT ===========================================================


    par_mt3d = get_mt3d_parameters(workbook_name=workbook, sheet_name='MT3D',
                                layers=borehole[use_name].layers,
                                new_layers=None,
                                trench=use_trench,
                                ibound=par_mf['bas']['ibound'],
                                tds_profile=tds_profiles[use_name],
                                gr=gr)


    seawat(dirs=dirs, case=case_sw, par_mf=par_mf, par_mt=par_mt3d, bdd=None )


    it = par_mf['dis']['nper'] - 1
    xlim = gr.x[1], gr.x[-2]

    ucn = UCN_obj(dirs, gr=gr)

    axs = newfig2h(["Totaal zout conentraties {} voor t = {:.0f} d, sleuf_nr {}".format(use_name, ucn.times[-1], itr),
                       "Totaal zout concentratie [mg/L] onder en naast de sleuf"],
                       ["x [m]", "TDS [mg/L]"],
                       ["NAP [m]", "NAP [m]"], size_inches=(16, 8), shift_fraction=0.4)



    plot_section(boring=borehole[use_name], trench=trenches.trenches[itr],
                     xlim=gr.x[[0, -1]], ylim=(-16, 2), ax=axs[0], trenchcolor='white')


    hds.plot_water_table(it=it, gr=gr, ax=axs[0])

    clevels =[500, 1000, 2000, 4000, 8000, 16000, 24000, 32000]

    Cs = ucn.contour(it=it, gr=gr, ax=axs[0], levels=clevels, linewidths=1., colors='darkred', zorder=3)
    ax.clabel(Cs, clevels, inline=True, fmt='%.0f', fontsize=10)

    axs[0].set_xlim(-100, 100)

    # Zoutprofiel onder en naast de sleuf

    tds_profiles[use_name].plot(ax=ax)
    for ix in [80, 136]:
        axs[1].plot(ucn.conc[-1, :, 0, ix], gr.ZM[:, 0, ix], label=f"tds x={gr.xm[ix]:.0f} m")

    axs[1].legend()


    #%% Comparing confined and unconfined 1D flow

    ax = newfig("Vergelijk freatisch - gespannen", "x [m]", "h [m]")

    b, k, n, D = 80, 5, 0.0005, 1/2.5

    x = np.linspace(0, b, 50)

    h = np.sqrt(n / k * (b ** 2 - x ** 2))
    phi = n / (2 * k * D) * (b ** 2 - x ** 2)

    ax.plot(x, h, label='freatisch')
    ax.plot(x, phi, label='gespannen')
    ax.plot(x, D * np.ones_like(x), label='D')
    ax.legend()
    print("h[0] = {:.2f}, phi[0]= {:.2f}, h[0] / phi[0] = {:.2f}".format(h[0], phi[0], h[0]/phi[0]))


# %%

# %%
