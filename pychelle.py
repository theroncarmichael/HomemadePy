import numpy as np
import astropy.io.fits as fits
import random
import scipy
import pdb
import ipdb
import glob
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import defaultdict
from astropy.modeling import models, fitting
from scipy.signal import medfilt
import scipy.stats
from astropy.modeling.models import custom_model
import warnings
warnings.simplefilter('ignore', np.RankWarning)

######################################################################
def clean(filename, filewrite, flip, cut, scan, 
          write = True, hdr = 0, HIRES = False):
    """
    The clean() function removes NaN values and does a row-by-row 
    subtraction of the overscan region on the image. 
    The wavelength dispersion direction should approximately go 
    from left to right (use flip = T if 
    the echelle orders are vertical). 
    This function returns a 2D image with the overscan region 
    trimmed away.
    For slit-fed echelle, sky-subtraction is accounted for in 
    pychelle.trace().
    ----------
    Parameters:
    ----------
    filename: Name of raw data file
    filewrite: User-designated name of reduced data file
    flip: True: Rotated image by 90 degrees with numpy.rot90()
    cut: X-pixel value image is trimmed to
    scan: The X-pixel value of the start of the overscan region
    write: True: Save image to ``filewrite``
    hdr: Header index of raw image in ``filename``
    ----------
    Returns: 2D image with overscan region trimmed off
    ----------
    """
    print 'Processing image...'
    image_file = fits.open(str(filename))
    image = image_file[hdr].data.astype(float)
    image_file.close()
    # Remove NaN values
    image = image[[~np.any(np.isnan(image),axis = 1)]]
    if flip: # Rotate the frame so the orders run horizontally
        image = np.rot90(image, k = 1)
    nrows, ncols = image.shape[0], image.shape[1]
    bias, dark = np.zeros(nrows), np.arange(cut,nrows*ncols,ncols) 
    # dark is the last column of pixels at which this cutoff occurs
    # and only darker areas that are not part of the orders remain. 
    # For example, if there are 50 columns of darkness after the orders
    # end, then cut-cols should equal 50 to remove these dark areas.
    for i in range(nrows): # loop through the number of rows
        # take row i and look the in overscan 
        # region parsed with [scan:ncols]
        bias[i] = np.median(image[[i]][0][scan:ncols]) 
    clipped_bias = scipy.stats.sigmaclip(bias) #Remove outliers
    bias_sigma = 5.0*np.std(clipped_bias[0])
    bias_median = np.median(clipped_bias[0])
    bias[bias <= (bias_median - bias_sigma)] = bias_median
    bias[bias >= (bias_median + bias_sigma)] = bias_median    
######################################################################
    for i in range(nrows):
        # Find and subtract the median bias of each row 
        # from the overscan region
        image[i,:] -= bias[i]
    # ``cut`` is the pixel on which the orders finishes
    image = np.delete(image, np.s_[cut:ncols], axis = 1)
    if HIRES: # Trim the image according to HIRES specifications
        # axis = 0 deletes rows (681 to the top row here)
        image = np.delete(image, np.s_[681::1] , axis = 0)
        # delete the bottom rows after the top rows
        image = np.delete(image, np.s_[0:27:1], axis = 0)  
    if write:
        hdulist = fits.HDUList()
        prime_hdr = fits.PrimaryHDU()
        prime_hdr.header['Bias_med'] = round(bias_median) 
        prime_hdr.header['Bias_dev'] = round(bias_sigma,2)
        prime_hdr.header['Bias_min'] = round(np.min(bias))
        prime_hdr.header['Bias_max'] = round(np.max(bias))
        cleaned_image = fits.ImageHDU(image, name = 'Processed 2D Image')
        hdulist.append(prime_hdr)
        hdulist.append(cleaned_image)
        print 'Writing file: ', str(filewrite)+'_CLN.fits'
        hdulist.writeto(str(filewrite)+'_CLN.fits', overwrite = True)
    print '\n~-# Image processed #-~ \n'
    return image # Returns the cleaned image



def peaks(y, nsig, mean = -1, deviation = -1):
    """
    This functions returns the indices of the peaks in an array.
    The height of the peaks to be considered can be controlled with ``nsig`` (how many standard deviations
    above some mean a datum is).\n 
    Y: Y-values of data
    nsig: The number of standard deviations away from the ``mean`` a ``y`` value in ``y`` must be to qualify as a peak
    mean: Manually set a mean value. Default uses the mean of Y
    deviation: Manually set the standard deviation. Default uses the standard deviation of Y
    """
    right, left = y - np.roll(y,1), y - np.roll(y,-1) 
    #Shift the elements in ``y`` left and right and subtract this from the original to check where these values are > 0
    pk = np.where(np.logical_and(right > 0, left > 0))[0]
    if nsig <= 0.0:
        print 'Setting ``nsig`` = 1.0 in peaks()'
        nsig = 1.0
    if nsig > 0.0:
        if type(y) != type(np.array(0)): #Verify lists and arrays are not interacting so y can be indexed with pk
            y = np.array(y)
        yp = y[pk]
        if mean != -1 and deviation == -1: #Use the input mean and/or standard deviation or calculate them
            mn, std = mean, np.std(yp)
        elif deviation != -1 and mean == -1:
            mn, std = np.mean(yp), deviation
        elif mean != -1 and deviation != -1:
            mn, std = mean, deviation
        else:
            mn, std = np.mean(yp), np.std(yp)
        peak = np.where(yp > mn + nsig*std)[0] #Applies ``nsig`` constraint to maxima; how separated from the noise a maximum is
        npk = len(peak)                        #Number of maxima
        if npk > 0:                            #If ``nsig`` is not too high and npk > 0 then these ``nsig``-constrained
            peak_ind = []                      #maxima are added to an updated maximum index list
            for i in peak:
                peak_ind += [pk[i]]
        else:
            peak_ind = []
            print "Relax peak definition; reduce ``nsig`` in peaks() or adjust ``xstart`` and ``ystart`` in trace() to avoid bias region"
    else:
        peak_ind = pk
    return np.array(peak_ind) #Returns the indices of ``y`` at which nsig-constrained maxima are



def trace_fit(x,y, deg = 1):
	""""
	This function utilizes numpy's polyfit and polyval functions to return parameters of a fit to a curve\n
	x: The x data input used in numpy.polyfit()
	y: The y data input used in numpy.polyfit()
	deg: Polynomial degree
	"""
	line_params = np.polyfit(x, y, deg)
	trc_fnc = np.polyval(line_params, x)
	return trc_fnc, line_params

def sigma_clip(x,y, deg = 1, nloops = 15, sig = 5.0):
	"""
	Sigma clip data based on a fit produced by numpy's polyfit and polyval functions\n
	x: The x data input used in numpy.polyfit()
	y: The y data input used in numpy.polyfit()
	deg: Polynomial degree
	nloops: Number of loops to iterate over while clipping
	sig: Number of sigma away from the fit the data can be to be clipped
	"""
	y_sig_arr = np.arange(0,nloops,1.0)
	for i in range(1,nloops):
		line_params = np.polyfit(x, y, deg)
		y_fit = np.polyval(line_params, x)
		y_sig = sig*np.std(y-y_fit)
		y_sig_arr[i] = y_sig
		clipped_ind = np.where(np.abs(y-y_fit) <= y_sig)[0]
		y, x = y[clipped_ind], x[clipped_ind] #Reset to te newly clipped data
		if np.around(y_sig_arr[i],3) == np.around(y_sig_arr[i-1],3):
			break
	return line_params

def blaze_fit(xrng, spec):
	"""
	This function fits a 1D blaze function to each spectral order once the order has been integrated to 1D\n
	xrng: The x values of the data (typcially pixels along the dispersion direction of the detector)
	spec: The integrated values of the order at each x datum
	"""
	if spec[0] < spec[-1]: #Find the count        #The trace of each order is fit with tf and the start of this
	    mn = spec[0]       #values for the        #trace is where the rectified orders begin. This reduces the
	else:                  #edges of the order    #effect of comic rays at the beginning of the order
	    mn = spec[-1]      #to fit the blaze      #misplacing the rectified order
	blaze = peaks(spec,0.1,mean = mn) #Find top of spectrum to approximate the blaze function
	pks = spec[blaze]
	blfn_params = trace_fit(blaze, pks, deg = 7)[1]
	blfn = np.polyval(blfn_params, xrng)
	return blfn

def gauss_lorentz_hermite_prof(x, mu1 = 0.0, amp1 = 1.0, sig = 1.0, offset1 = 1.0, offset2 = 1.0,
			       c1 = 1.0, c2 = 1.0, c3 = 1.0, c4 = 1.0, c5 = 1.0, c6 = 1.0, c7 = 1.0, c8 = 1.0, c9 = 1.0,
			       amp2 = 1.0, gamma = 0.5, mu2 = 0.1):
	gauss = amp1 * np.exp(-0.5 * (( x - mu1 ) / sig )**2) + offset1
	lorentz = amp2 * (0.5 * gamma) / (( x - mu2 )**2 + (0.5 * gamma)**2) + offset2
	h_poly = c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6 + c7*x**7 + c8*x**8 + c9*x**9
	return (h_poly) * gauss + lorentz

def instrumental_profile(image, order_length, trc_fn, gauss_width):
	""" 
	This function is the algorithm for creating a super-sampled profile of each spectral order.
	Sample the trace at each column to produce a profile shape for the trace using the trace function.\n
	image: The cleaned echelle image
	order_length: The length of the dispersion direction (number of x pixels)
	trc_fn: The trace functions of each spectral order of the cleaned image
	gauss_width: The distance from the center of each order to include in the Gaussian fit\n
	"""
	"""
	-------------------------------------
	Initialize an IP based on 128 columns
	-------------------------------------
	"""
	sample_length = len(np.arange(int(10-gauss_width),int(10+gauss_width),1))
	order_sample_arr = np.zeros((order_length, sample_length))
	for x in range(order_length):
	    xdata = np.arange(int(trc_fn[x]-gauss_width),int(trc_fn[x]+gauss_width),1) 
	    #Select the area along the order to sample
	    ydata_ind = np.arange(int(trc_fn[x]-gauss_width),int(trc_fn[x]+gauss_width),1)
	    ydata = image[:,x][ydata_ind]
	    order_sample_arr[x,:] = ydata
	# Fit a Gaussian to the profile of the order at each column to normalize the height #
	    mu, sigma, amp = trc_fn[x], 1.50, np.max(ydata)
	    initial_model = models.Gaussian1D(mean = mu, stddev = sigma, amplitude = amp) 
	    #Initialize a 1D Gaussian model from Astropy
	    fit_method = fitting.LevMarLSQFitter() 
	    #instantiate a fitter, in this case the fitter that uses the Levenburg Marquardt Least-Squares algorithm
	    odr_prof = fit_method(initial_model, xdata, ydata)
	    gauss_int = np.sum(odr_prof(xdata))
	    order_sample_arr[x,:] = ydata/gauss_int # Normalize the height of the profile
	# Initialize arrays for super-sampled Y coordinates (xrng_arr) and counts at each Y coordinate #
	xrng_arr = np.zeros((len(trc_fn),2*gauss_width))
	order_prof = np.zeros((len(trc_fn),2*gauss_width))
	# Floor the trace function to get sub-pixel precision #
	pix_order = trc_fn.astype(int)
	x_shift = pix_order[0] - gauss_width
	rect_order = pix_order - trc_fn
	for i in range(len(trc_fn)):
	    xrng_fit = np.arange(rect_order[i]-gauss_width, rect_order[i]+gauss_width,1.0)
	    xrng_arr[i] = xrng_fit
	    order_prof[i] = order_sample_arr[i,:]
	xrng_arr -= np.min(xrng_arr)
	xrng_arr += x_shift
	yrng = np.arange(0,len(trc_fn),1)
	yrng_arr = np.zeros((len(yrng),2*gauss_width))
	for t in range(len(yrng_arr[:,0])):
	    yrng_arr[t,:] = t
	# Store the super-sampled profile shapes by column or as a sorted continuous function #
	x_long, y_long = xrng_arr.reshape((len(trc_fn)*2*gauss_width)), order_prof.reshape((len(trc_fn)*2*gauss_width))
	sorted_ind = np.argsort(x_long)
	x_sorted, y_sorted = x_long[sorted_ind], y_long[sorted_ind]
	profile_coordinates = (x_long,y_long, x_sorted,y_sorted)
	return profile_coordinates 




def trace(image, xstart, ystart, xstep, yrange, nsig, filewrite, sep,
	  write=False, odr=False, MINERVA = False, HIRES = False, MIKE = False, cutoff = [0]):
    """
    This function returns the coordinates of the echelle orders.\n
    image: 2-D image array containg echelle orders
    xstart/ystart: Typically 0 for the corner of the image from which the search for orders begins
    xstep: Number of X-pixels to skip subtracting 1 to include in a fit to the trace, 
    i.e. ``xstep`` = 1 uses all X-pixels (skips 0 pixels)
    yrange: Y-pixel range to search for the next part of the order as the X-pixels are looped over
    nsig: The number of standard deviations away from the ``mean`` a ``y`` value in ``y`` must be to qualitfy as a peak
    filewrite: User-designated name of traced data file
    sep: Y-pixel separation between two detected peaks; used to only take one of these adjacent peaks
    write: True: Save image to ``filewrite`` / False: Do not save to ``filewrite``
    odr: 1-D array; if the starting Y-pixel values of each order is known, then input odr\n
    """
    print 'Locating spectral orders on cleaned image...'
    xrng, yvals, counts, centroids = np.arange(1,image.shape[1]+1,xstep), [], [], []
    background_levels = []
    rect_image, blze_image = np.zeros(image.shape), np.zeros(image.shape)
    ##### Automatically locates the start of each order #####
    odr_start, odr_ind = peaks( image[ystart:image.shape[0], xstart], nsig ),[]
    if odr: #Use input list of order starting locations
        odr_start = odr
    if MINERVA == True: #Account for the curve in orders across the detector cutting off on the edge and remove them
        #cutoff_order =  np.where(np.array(odr_start) - 70 < 0)[0]
        odr_start = np.delete(odr_start, cutoff)
    #Create an empty array for the trace function of each order
    trace_arr = np.zeros((len(odr_start),image.shape[1]/xstep))
    for i in range(len(odr_start)):
        if np.abs(odr_start[i] - odr_start[i-1]) <= sep and len(odr_start) > 1: #Remove pixel-adjacent peak measurements
            odr_ind += [i]                                                      #This avoids double-counting a peak
    odr_start = list(np.delete(odr_start,odr_ind))                              
    if HIRES and np.abs(odr_start[-1] - image.shape[0]) <= 10:
		odr_start = list(np.delete(odr_start, -1))
    trc_cont = raw_input(str(len(odr_start))+" orders found, is this accurate? (y/n): ")
    while trc_cont != 'y' and trc_cont != 'n': 
		trc_cont = raw_input('Enter y or n: ')
		if trc_cont == 'y' or trc_cont == 'n': break
    if trc_cont == 'n':
		print 'Starting Y-pixel coordinates of orders: ', odr_start
		print 'Exiting pychelle.trace(). Adjust ``sep``, ``nsig``, or ``cutoff`` to locate the correct number of full orders.\n'
		return [], [], [], []
    elif trc_cont == 'y':
		print 'Starting Y-pixel coordinates of orders: ', odr_start
		print str(len(odr_start))+" orders found, scanning each column for the pixel coordinates of peaks...\n"
		# Algorithm for order definition via a trace function #
		prof_shape, end_trace = [], False
		for o in range(len(odr_start)):  #At each pixel coordinate where a peak (the top of an order) was detected
			if MINERVA:
				dy = 8 # ----------------------------- Hard coded half-width of each order; width = 2*dy
			elif HIRES:
				dy = 15
			elif MIKE:
				dy = 6
			for i in xrng:    #(the values in 'odr_start'),take slices of each column across the
				column = image[ystart:image.shape[0],i-1] #horizontal range (xrng) and begin tracing the order based on
				ypeaks = peaks(column,nsig)   #the peak coordinates found near the currently iterating loop
				if len(ypeaks) == 0:
					print 'Starting too close to edge; increase ``xstart`` in trace()\n'
					break
				for y in range(len(ypeaks)):
					if len(yvals) <= 5 and np.abs((ypeaks[y] + ystart) - odr_start[o]) <= yrange:
						ypix = ypeaks[y]   #After the first few (5 in this case) peaks are found, this
						break              #trend (X,Y pixel coordinates) is what is updated and referenced
											#for the rest of the horizontal range. The trend is an average
					else:                  #pixel coordinate value for 5 preceding peak pixels
						if len(yvals) > 5:
							ytrend = int(np.mean(yvals[len(yvals)-5:]))
							if np.abs((ypeaks[y] + ystart) - ytrend) <= yrange: #Checking that the next peaks are within
								ypix = ypeaks[y]                                #range of the trend
								break
	    # Fit a Gaussian to the cross-section of each order to find the center for the trace function  #
				initial_model = models.Gaussian1D(mean=ypix, stddev=1.2, amplitude = np.max(column[ypix-dy : ypix+dy]))
				fit_method = fitting.LevMarLSQFitter()
				xaxis, yaxis = np.arange(ypix-dy, ypix+dy, 1), column[ypix-dy : ypix+dy]
				if len(xaxis) > len(yaxis): # If column[] is indexed at an integer > it's length, the order is
					                # running off the edge of the detector
					print '\nTruncated order detected\n'
					odr_start = np.delete(odr_start, o)
					end_trace = True
					break
				odr_prof = fit_method(initial_model, xaxis, yaxis) #Fit X and Y data using the initialized 1D Gaussian
				background_level = np.median(yaxis)#odr_prof2.offset2.value + odr_prof2.offset1.value    
				'''
				if background_level <= 0.0:# or np.abs(ypix - odr_prof.mean[0]) >= 10:
					plt.plot(xaxis, yaxis, 'ko')
					xmooth = np.linspace(xaxis[0],xaxis[-1],1000)
					#plt.plot(xmooth, odr_prof2(xmooth), 'b-', label = 'Gauss-Hermite-Lorentz')
					plt.plot(xaxis, odr_prof(xaxis), 'g-', label = str(ypix)+'_'+str(odr_prof.mean[0]))
					plt.title('col_number_'+str(i+1)+'_order_'+str(o+1))
					plt.axhline(y = background_level, color = 'g')
					#plt.axhline(y = np.median(odr_prof2(xmooth)), color = 'r')
					#plt.axhline(y = np.median(yaxis), color = 'm')
					#plt.xlim(20,50), plt.ylim(0,70)
					plt.legend()
					#plt.savefig('/Users/theroncarmichael/Desktop/Exolab/diags/KH_15D_chunks/
					#col_number_'+str(i+1)+'_order_'+str(o+1))
					plt.show()
					plt.close()
				'''
				fit_centroid = float(odr_prof.mean[0])
	    # Add the centroid fit to the array used to fit a trace function #
				centroids += [fit_centroid + ystart]
				yvals += [ypix + ystart]
				counts += [column[ypix]]
				background_levels += [background_level]
				'''
	    # Diagnostic plot for difference between Gaussian centroids and peak location of cross-section of order #
			    x_fine = np.arange(ypix-dy, ypix+dy, 0.1)
			    d_peak = np.round(np.abs(fit_centroid - xaxis[len(xaxis)/2]),3)
			    plt.figure(i)
			    plt.plot(xaxis , yaxis, 'bo')
			    plt.plot(x_fine, odr_prof(x_fine), 'r-', linewidth = 1.5)
			    gp, = plt.plot(odr_prof.mean[0], odr_prof.amplitude[0], 'go')
			    plt.legend([gp] ,['Peak residuals: '+str(d_peak)+' pixels'],  loc=1 , prop = {'size':10})
			    plt.xlim(xaxis[0], xaxis[-1]), plt.show()
			    pdb.set_trace()
		        '''
			if end_trace == True:
				break #Stop creating new centroids for the trace function
	    # Use the centroid values from the Gaussian fits to create a trace of order[o] #
			#yvals = [x+1 for x in yvals] #Correct for indexing 0 to 1
			order_length, gauss_width = len(yvals), 10
			trc_fn = trace_fit(xrng, centroids, deg = 7)[0]
			trace_arr[o] = trc_fn#Store the trace function for each order
			print '\n=== Spectral order '+str(o+1)+' traced ==='
			print 'Calculating profile shape for order '+str(o+1)+'...'
			prof_shape += [instrumental_profile(image, order_length, trc_fn, gauss_width)]
			'''
		# Diagnostic plot for difference between Gaussian centroids and peak location of cross-section of order #
			plt.figure(1, figsize = (10,5))
			c, = plt.plot(xrng, trc_fn, 'b-', linewidth = 1.0, label = 'Gaussian Centroid fit')#trace_arr[o], 'r-')
			plt.plot(xrng, yvals, 'k.', markersize = 2.0)
			f, = plt.plot(xrng, trace_fit, 'r-', linewidth = 1.0, label = 'Peak of Data fit')
			plt.plot(xrng, centroids, 'g.', markersize = 2.0)
			plt.gca().invert_yaxis(), plt.xlim(xrng[0],xrng[-1])
			plt.title('Centroid-fit and peak-fit trace function'), plt.xlabel('X Pixel'), plt.ylabel('Y Pixel')
			plt.legend(handles = [c,f], loc=1, prop = {'size':11})
	
			plt.figure(2, figsize = (10,5))
			plt.plot(xrng, trc_fn - yvals, 'ko', markersize = 2.0)
			plt.xlim(xrng[0],xrng[-1])
			plt.title('Residuals between data and fit to centroid fits'), plt.xlabel('X Pixel')
			plt.ylabel('Y Pixel (Centroid - Data)')
			plt.show()
			pdb.set_trace()
			
			plt.figure(figsize = (7,7), dpi = 70)
			plt.title('Trace function of an echelle order from HIRES', fontsize = 16)
			plt.xlabel('Dispersion direction (column number)', fontsize = 14), plt.ylabel('Cross-dispersion direction (row number)', fontsize = 14)
			plt.plot(trc_fn, ls = '-', color = 'royalblue', lw = 2.0, label = 'Trace function')
			plt.plot(yvals, color = 'k', marker = '.', ls = 'None', label = 'Physical peak of order'), plt.plot(centroids, ls = '-', color = 'orange', lw = 2.0, label = 'Modeled peak of order')
			plt.legend(loc = 2), plt.show()
			pdb.set_trace()
			'''			
			counts, yvals, centroids, background_levels = [], [], [], [] #Reset the arrays for the next order in loop
		if write:
			hdulist = fits.HDUList()
			f1 = fits.ImageHDU(trace_arr, name = 'Trace function')
			f2 = fits.ImageHDU(prof_shape, name = 'Profile fitting')
			hdulist.append(f1), hdulist.append(f2)
			hdulist.writeto(str(filewrite)+'_TRC.fits', overwrite = True)
			print 'Writing file '+str(filewrite)+'_TRC.fits'
			hdulist.close()
			print '\n ~-# Spectral orders traced: '+str(len(odr_start))+' #-~\n'
		return trace_arr, prof_shape




def flat(filepath, filewrite, hdr, window, write = True):
	"""
	This function creates normalized flat images of echelle spectra.\n
	filepath: Location of raw data files
	filewrite: User-designated name of reduced data file
	hdr: Header index of raw image in ``filename``
	window: Size of the smoothing window used in scipy.signal.medfilt()
	write: True: Save image to ``filewrite`` / False: Do not save to ``filewrite``\n
	"""
	print 'Creating normalized flat image...'
	file_no = np.sort(glob.glob(filepath+'*fits'))
	flat_img = np.zeros(fits.open(file_no[0])[hdr].data.shape)
	for f in range(len(file_no)):
		flat_img += fits.open(str(file_no[f]))[hdr].data #/ float(len(file_no)) 
	flat_img /= float(len(file_no)) #Average the flat images
	model_flat = np.zeros(flat_img.shape) #Create an array that is the same shape as the image
	for i in range(flat_img.shape[1]):
		flat_col = flat_img[:,i]
		med_flat = scipy.signal.medfilt(flat_col, window)
		#plt.title('Cross section of HIRES flat image', fontsize = 16), plt.xlabel('Cross-dispersion direction (row number)', fontsize = 14), plt.ylabel('Counts', fontsize = 14)
		#plt.plot(flat_col, color = 'black', lw = '1.0'), plt.xlim(0, flat_img.shape[0])
		#plt.plot(med_flat, color = 'royalblue', lw = 2.0, label = 'Median filtered cross section'), plt.legend()
		#plt.show()
		#pdb.set_trace()
		if len(np.where(med_flat <= 0)[0]) > 0:
			med_flat[np.where(med_flat <= 0)[0]] = 1.0
			#pdb.set_trace()
		model_flat[:,i] = med_flat
	print 'Dividing the trimmed flat by the median smoothed model flat...'
	flat_img[flat_img <= 0.0] = 1.0
	norm_flat = flat_img / model_flat #The quantum efficiency based on the modelling method
	#plt.plot( np.where(flat_img == 0)[1], np.where(flat_img == 0)[0], 'go' ) 
	#plt.xlim(-100, 4200), plt.ylim(-50, 700)
	#plt.axhline(y = flat_img.shape[0]), plt.axhline(y = 0), plt.axvline(x = 0), plt.axvline(x = flat_img.shape[1])
	#plt.show()
	#pdb.set_trace()
	if write:
		hdu = fits.HDUList()
		flat = fits.ImageHDU(flat_img, name = 'Averaged flat')
		norm = fits.ImageHDU(norm_flat, name = 'Normalized Flat')
		hdu.append(norm), hdu.append(flat)
		print 'Writing file: ', str(filewrite)+'_FLT.fits'
		hdu.writeto(str(filewrite)+'_FLT.fits', overwrite = True)
	print '\n~-# Normalized flat image created #-~\n'
	return norm_flat, flat_img
    #Returns the input flat image, the two quantum efficiencies, and the two modelling methods




def spectext(image, nfib, trace_arr, yspread, filewrite, cal = False):
    """
    This function integrates an echelle image along the trace function of each order to collapse it from 2D to 1D.\n
    image: 2-D image array containing echelle orders
    nfib: Number of fibers
    trace_arr: 2-D array of number of columns x number of orders containing the coordinates of the orders
    yspread: Approximate width of order. Calibration spectra are typically wider than science spectra
    filewrite: User-designated name of reduced data file
    cal: True if the spectrum is a calibration spectrum\n
    """
    hdu = fits.HDUList()
    for f in range(1,nfib+1):
        trace_path, xrng = trace_arr[f-1::nfib,:], np.arange(0,image.shape[1],1)
        signal, spec_flat = np.zeros((len(trace_path),len(xrng))), np.zeros((len(trace_path),len(xrng)))
	if nfib == 1:
		print 'Using trace coordinates to locate spectrum along each order...'
	else:
		print 'Using trace coordinates to locate spectrum on fiber '+str(f)+' of '+str(nfib)+'...'
        # Integrate the counts over a length of the column along each point in the trace function #
	for j in range(len(trace_path)):
		for i in xrng:
			dy = np.arange(trace_path[j][i]-yspread,trace_path[j][i]+yspread,1)
			dy_int = dy.astype(int)
			bottom, top, out_rng = 0, image.shape[0], []
			for p in range(len(dy_int)): #Remove indices beyond the range of the detector
				if dy_int[p] <= bottom+1:
					out_rng += [p]
				if dy_int[p] >= top-1:
					out_rng += [p]
			dy_int, dy = np.delete(dy_int,out_rng), np.delete(dy,out_rng)
			if cal == False:
				cts_int_rng = image[dy_int,i] - np.median(image[dy_int,i]) #Remove sky background (if using slit-fed, otherwise, simply removes inter-order background)
			elif cal == True:
				cts_int_rng = image[dy_int,i]
			'''
			custom_gh_model = custom_model(gauss_lorentz_hermite_prof)
			gh_model = custom_gh_model(mu1 = len(cts_int_rng)/2., amp1 = np.max(cts_int_rng), sig = 1.0, offset1 = 0.0, offset2 = 1.0, c1 = 0.0, c2 = 0.0, c3 = 0.0, c4 = 0.0, c5 = 0.0, c6 = 0.0, c7 = 0.0, c8 = 0.0, c9 = 0.0, amp2 = np.max(cts_int_rng), gamma = 1.2, mu2 = len(cts_int_rng)/2.)
			fit_method = fitting.LevMarLSQFitter()
			xaxis, yaxis = np.arange(0, len(cts_int_rng), 1), cts_int_rng
			x_fine = np.arange(0,len(cts_int_rng), 0.001)
			odr_prof = fit_method(gh_model, xaxis, yaxis) #Fit X and Y data using the initialized 1D Gaussian
			resid = yaxis - odr_prof(xaxis)
			cosmic_detection = np.where(np.abs(resid) >= np.abs(5.0*np.std(resid))) 
			if len(cosmic_detection[0]) >= 1: 
				print xaxis[cosmic_detection]
				plt.figure(1)
				plt.plot(xaxis, yaxis, 'bo') 
				plt.plot(x_fine, odr_prof(x_fine), 'r')
#				plt.figure(2)
#				plt.plot(xaxis, resid, 'ko')
#				plt.axhline(0.0, color= 'b')
#				plt.axhline(-5.0*np.std(resid), color= 'r')
#				plt.axhline(5.0*np.std(resid), color= 'r')
				yaxis[cosmic_detection] = odr_prof(xaxis)[cosmic_detection]
				plt.figure(2)
				plt.plot(xaxis, yaxis, 'bo') 
				plt.plot(x_fine, odr_prof(x_fine), 'r')
				
				plt.show()
				pdb.set_trace()
			'''
			#print cts_int_rng
			#print image[dy_int[-1],i]
			#print image[dy_int[0],i]
			#pdb.set_trace()
			up_pix = np.abs(dy[-1]-dy_int[-1])*image[dy_int[-1],i]#image[dy_int[-1]+1,i] ########################################## 
		#Take the lower fraction of counts from the uppermost pixel cut through by dy
			low_pix = (1 - np.abs(dy[0]-dy_int[0]))*image[dy_int[0],i] 
		#Take the upper fraction of counts from the lowermost pixel cut through by dy
			if len(scipy.integrate.cumtrapz(cts_int_rng,dy_int)) == 0:
				'Spectrum integration paused'
				pdb.set_trace()
			signal[j,i] = scipy.integrate.cumtrapz(cts_int_rng,dy_int)[-1] + up_pix + low_pix
        # Blaze fitting #
		if cal == True: #Fit a blaze function to the continuum of an emission arclamp
			blaze_pars = sigma_clip(xrng, signal[j], deg = 7, nloops = 30)
			blaze = np.polyval(blaze_pars, xrng)
			low = np.where(signal[j] <= blaze)
			blaze_pars_new = sigma_clip(xrng[low], signal[j][low], deg = 7, nloops = 30)
			blfn = np.polyval(blaze_pars_new, xrng)
			spec_flat[j] = (signal[j]/blfn)
		else:
			if signal[j][0] < signal[j][-1]: #Find the count      #The trace of each order is fit with tf and the
				mn = signal[j][0]        #values for the      #start of this trace is where the rectified orders
			else:                            #edges of the order  #begin. This reduces the effect of cosmic rays at the
				mn = signal[j][-1]       #to fit the blaze    #left edge of the order misplacing the starting point
			blaze = peaks(signal[j],0.1,mean = mn) #Find top of spectrum to approximate the blaze function
			pks = signal[j][blaze]
			blazed = trace_fit(blaze, pks,7) #Fit blaze function to top of spectrum
			blfn = np.polyval(blazed[1], xrng)
			spec_flat[j] = (signal[j]/blfn)
	spec_f = fits.ImageHDU(spec_flat, name = 'Blaze-corrected spectrum') 
	spec = fits.ImageHDU(signal, name = '1D Spectrum')
	xpix = fits.ImageHDU(xrng, name = 'X pixel')
	hdu.append(xpix), hdu.append(spec), hdu.append(spec_f)
	print 'Writing file: ', str(filewrite)+'_SPEC.fits'
	hdu.writeto(str(filewrite)+'_SPEC.fits', overwrite = True)
	hdu.close()
	print '\n ~-# Spectrum extracted #-~\n'
	return xrng, signal, spec_flat

