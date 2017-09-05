import numpy as np, astropy.io.fits as fits, random, scipy, pdb, glob, ipdb
import matplotlib.pyplot as plt, matplotlib.axes as ax, matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from astropy.modeling import models, fitting
from scipy.signal import medfilt
