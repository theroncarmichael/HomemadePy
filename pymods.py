import numpy as np, astropy.io.fits as fits, random, scipy, pdb, glob, ipdb
import matplotlib.pyplot as plt, matplotlib.axes as ax, matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from astropy.modeling import models, fitting
from scipy.signal import medfilt
import scipy.stats

def plot(X = [0], Y = [0], IMAGE = np.zeros((10,10)), COLOR = 'b.', MS = 2.0, LW = 2.0, FIGX = 10, FIGY = 10,  NAXIS = 1, BG = 'white', INTER = 'none', VMIN = True, VMAX = True, NO = 1):
	if NAXIS == 1:
		plt.figure(NO, figsize = (FIGX,FIGY))
		plt.plot(X, Y, COLOR, ms = MS, linewidth = LW)
		plt.gca().set_axis_bgcolor(BG)
	if NAXIS == 2:
		if VMIN == True and VMAX == True:
			mid, std = np.median(IMAGE), np.std(IMAGE)
			VMIN, VMAX = mid - 1*std, mid + 1*std
		else:
			VMIN, VMAX = VMIN, VMAX
		plt.figure(NO, figsize = (FIGX,FIGY))
		ax = plt.gca()
		im = ax.imshow(IMAGE, cmap = 'gray', vmin = VMAX, vmax = VMAX, interpolation = INTER) 
		div = make_axes_locatable(ax)
		cax = div.append_axes("right", size="5%", pad=0.05)
		plt.colorbar(im, cax = cax)
	pass


