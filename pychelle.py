from pymods import *
from astropy.modeling.models import custom_model
from scipy.special import  erf
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def clean(FILENAME, FILEWRITE, FLIP, CUT, CEILING, SCAN, WRITE=True, HDR=0, HIRES = False):
    """
    The clean() function removes NaN values and performs a row-by-row subtraction of the overscan region on the image. 
    The wavelength dispersion direction should approximately go from left to right (use FLIP = T if 
    the echelle orders are vertical). 
    This function returns a 2D image with the overscan region trimmed away.
    For slit-fed echelle, sky-subtraction is accounted for in pychelle.trace().\n
    FILENAME: Name of raw data file
    FILEWRITE: User-designated name of reduced data file
    FLIP: True: Rotated image by 90 degrees / False: Do not rotate image
    CUT: X-pixel value image is trimmed to
    CEILING: Hot pixel value to set to median of overscan/bias region
    SCAN: The X-pixel value of the start of the overscan region
    WRITE: True: Save image to FILEWRITE
    False: Do not save to FILEWRITE
    HDR: Header index of raw image in FILENAME
    """
    print 'Cleaning image...'
    image_file = fits.open(str(FILENAME))
    image = image_file[HDR].data.astype(float)
    image_file.close()
    image = image[[~np.any(np.isnan(image),axis = 1)]] #Remove NaN values
    if FLIP: #Rotate the frame so the orders run horizontally
        image = np.rot90(image, k = 1)
    nrows, ncols = image.shape[0], image.shape[1] #Number of rows, columns
    bias, dark = np.zeros(nrows), np.arange(CUT,nrows*ncols,ncols) 
    #dark is the last column of pixels at which this cutoff occurs and only darker areas that 
    #are not part of the orders remain. 
    #For example, if there are 50 columns of darkness after the orders end, 
    #then CUT-cols should equal 50 to remove these dark areas.
    for i in range(nrows): #loop through the number of rows
        bias[i] = np.median(image[[i]][0][SCAN:ncols]) #take row #i and look the in overscan region parsed with [SCAN:ncols]
	image[i,:] -= bias[i] #Find and subtract the median bias of each row from the overscan region
    image = np.delete(image, np.s_[CUT:ncols], axis = 1) #CUT is the pixel the orders end on
    image[image >= CEILING] = np.median(bias)            #Set high points to bias level of the frame
    if HIRES: #Trim the image according to HIRES specifications
        image = np.delete(image, np.s_[681::1] , axis = 0) #axis = 0 deletes rows (681 to the top row here)
        image = np.delete(image, np.s_[0:27:1], axis = 0) #delete the bottom rows after the top rows 
    if WRITE:
        hdulist = fits.HDUList()
        cleaned_image = fits.ImageHDU(image, name = 'Reduced 2D Image')
        hdulist.append(cleaned_image)
        print 'Writing file: ', str(FILEWRITE)+'_CLN.fits'
        hdulist.writeto(str(FILEWRITE)+'_CLN.fits', overwrite = True)
	print '\n~-# Image cleaned #-~ \n'
    return image #Returns the cleaned image




def peaks(Y, NSIG, MEAN = -1, STDEV = -1):
    """
    This functions returns the indices of the peaks in an array.
    The height of the peaks to be considered can be controlled with NSIG (how many standard deviations
    above some mean a datum is).\n 
    Y: Y-values of data
    NSIG: The number of standard deviations away from the MEAN a Y-value in Y must be to qualify as a peak
    MEAN: Manually set a mean value. Default uses the mean of Y
    STDEV: Manually set the standard deviation. Default uses the standard deviation of Y
    """
    right, left = Y - np.roll(Y,1), Y - np.roll(Y,-1) 
    #Shift the elements in Y left and right and subtract this from the original to check where these values are > 0
    pk = np.where(np.logical_and(right > 0, left > 0))[0]
    if NSIG <= 0.0:
        print 'Setting NSIG = 1.0 in peaks()'
        NSIG = 1.0
    if NSIG > 0.0:
        if type(Y) != type(np.array(0)): #Verify lists and arrays are not interacting so Y can be indexed with pk
            Y = np.array(Y)
        yp = Y[pk]
        if MEAN != -1 and STDEV == -1: #Use the input mean and/or standard deviation or calculate them
            mn, std = MEAN, np.std(yp)
        elif STDEV != -1 and MEAN == -1:
            mn, std = np.mean(yp), STDEV
        elif MEAN != -1 and STDEV != -1:
            mn, std = MEAN, STDEV
        else:
            mn, std = np.mean(yp), np.std(yp)
        peak = np.where(yp > mn + NSIG*std)[0] #Applies NSIG constraint to maxima; how separated from the noise a maximum is
        npk = len(peak)                        #Number of maxima
        if npk > 0:                            #If NSIG is not too high and npk > 0 then these NSIG-constrained
            peak_ind = []                      #maxima are added to an updated maximum index list
            for i in peak:
                peak_ind += [pk[i]]
        else:
            peak_ind = []
            print "Relax peak definition; reduce NSIG in peaks() or adjust XSTART and YSTART in trace() to avoid bias region"
    else:
        peak_ind = pk
    return np.array(peak_ind) #Returns the indices of Y at which NSIG-constrained maxima are




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
	blaze = peaks(spec,0.1,MEAN = mn) #Find top of spectrum to approximate the blaze function
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




def trace(IMAGE, XSTART, YSTART, XSTEP, YRANGE, NSIG, FILEWRITE, SEP,
	  WRITE=False, ODR=False, PROFILE = True, MINERVA = False, HIRES = False, CUTOFF = [0]):
    """
    This function returns the coordinates of the echelle orders.\n
    IMAGE: 2-D image array containg echelle orders
    XSTART/YSTART: Typically 0 for the corner of the image from which the search for orders begins
    XSTEP: Number of X-pixels to skip subtracting 1 to include in a fit to the trace, 
    i.e. XSTEP = 1 uses all X-pixels (skips 0 pixels)
    YRANGE: Y-pixel range to search for the next part of the order as the X-pixels are looped over
    NSIG: The number of standard deviations away from the MEAN a Y-value in Y must be to qualitfy as a peak
    FILEWRITE: User-designated name of traced data file
    SEP: Y-pixel separation between two detected peaks; used to only take one of these adjacent peaks
    WRITE: True: Save image to FILEWRITE / False: Do not save to FILEWRITE
    ODR: 1-D array; if the starting Y-pixel values of each order is known, then input ODR\n
    """
    print 'Locating spectral orders on cleaned image...'
    xrng, yvals, counts, centroids = np.arange(1,IMAGE.shape[1]+1,XSTEP), [], [], []
    background_levels = []
    rect_image, blze_image = np.zeros(IMAGE.shape), np.zeros(IMAGE.shape)
    ##### Automatically locates the start of each order #####
    odr_start, odr_ind = peaks( IMAGE[YSTART:IMAGE.shape[0], XSTART], NSIG ),[]
    if ODR: #Use input list of order starting locations
        odr_start = ODR
    if MINERVA == True: #Account for the curve in orders across the detector cutting off on the edge and remove them
	#cutoff_order =  np.where(np.array(odr_start) - 70 < 0)[0]
        odr_start = np.delete(odr_start, CUTOFF)
    #Create an empty array for the trace function of each order
    trace_arr = np.zeros((len(odr_start),IMAGE.shape[1]/XSTEP))
    for i in range(len(odr_start)):
        if np.abs(odr_start[i] - odr_start[i-1]) <= SEP and len(odr_start) > 1: #Remove pixel-adjacent peak measurements
            odr_ind += [i]                                                      #This avoids double-counting a peak
    odr_start = list(np.delete(odr_start,odr_ind))                              
    if HIRES and np.abs(odr_start[-1] - IMAGE.shape[0]) <= 10:
		odr_start = list(np.delete(odr_start, -1))
    trc_cont = raw_input(str(len(odr_start))+" orders found, is this accurate? (y/n): ")
    while trc_cont != 'y' and trc_cont != 'n': 
		trc_cont = raw_input('Enter y or n: ')
		if trc_cont == 'y' or trc_cont == 'n': break
    if trc_cont == 'n':
		print 'Starting Y-pixel coordinates of orders: ', odr_start
		print 'Exiting pychelle.trace(). Adjust SEP, NSIG, or CUTOFF to locate the correct number of full orders.\n'
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
			for i in xrng:    #(the values in 'odr_start'),take slices of each column across the
				column = IMAGE[YSTART:IMAGE.shape[0],i-1] #horizontal range (xrng) and begin tracing the order based on
				ypeaks = peaks(column,NSIG)   #the peak coordinates found near the currently iterating loop
				if len(ypeaks) == 0:
					print 'Starting too close to edge; increase XSTART in trace()\n'
					break
				for y in range(len(ypeaks)):
					if len(yvals) <= 5 and np.abs((ypeaks[y] + YSTART) - odr_start[o]) <= YRANGE:
						ypix = ypeaks[y]   #After the first few (5 in this case) peaks are found, this
						break              #trend (X,Y pixel coordinates) is what is updated and referenced
									       #for the rest of the horizontal range. The trend is an average
					else:                  #pixel coordinate value for 5 preceding peak pixels
						if len(yvals) > 5:
							ytrend = int(np.mean(yvals[len(yvals)-5:]))
						if np.abs((ypeaks[y] + YSTART) - ytrend) <= YRANGE: #Checking that the next peaks are within
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
				fit_centroid = float(odr_prof.mean[0])
	    # Add the centroid fit to the array used to fit a trace function #
				centroids += [fit_centroid + YSTART]
				yvals += [ypix + YSTART]
				counts += [column[ypix]]
				background_levels += [background_level]
	    # Diagnostic plot for difference between Gaussian centroids and peak location of cross-section of order #
			    #x_fine = np.arange(ypix-dy, ypix+dy, 0.1)
			    #d_peak = np.round(np.abs(fit_centroid - xaxis[len(xaxis)/2]),3)
			    #plt.figure(i)
			    #plt.plot(xaxis , yaxis, 'bo')
			    #plt.plot(x_fine, odr_prof(x_fine), 'r-', linewidth = 1.5)
			    #gp, = plt.plot(odr_prof.mean[0], odr_prof.amplitude[0], 'go')
			    #plt.legend([gp] ,['Peak residuals: '+str(d_peak)+' pixels'],  loc=1 , prop = {'size':10})
			    #plt.xlim(xaxis[0], xaxis[-1]), plt.show()
			    #pdb.set_trace()
		    
			if end_trace == True:
				break #Stop creating new centroids for the trace function
	    # Use the centroid values from the Gaussian fits to create a trace of order[o] #
			#yvals = [x+1 for x in yvals] #Correct for indexing 0 to 1
			order_length, gauss_width = len(yvals), 10
			trc_fn = trace_fit(xrng, centroids, deg = 7)[0]
			trace_arr[o] = trc_fn#Store the trace function for each order
			print '\n=== Spectral order '+str(o+1)+' traced ==='
			print 'Calculating profile shape for order '+str(o+1)+'...'
			prof_shape += [instrumental_profile(IMAGE, order_length, trc_fn, gauss_width)]
		
		# Diagnostic plot for difference between Gaussian centroids and peak location of cross-section of order #
			#plt.figure(1, figsize = (10,5))
			#c, = plt.plot(xrng, trc_fn, 'b-', linewidth = 1.0, label = 'Gaussian Centroid fit')#trace_arr[o], 'r-')
			#plt.plot(xrng, yvals, 'k.', markersize = 2.0)
			#f, = plt.plot(xrng, trace_fit, 'r-', linewidth = 1.0, label = 'Peak of Data fit')
			#plt.plot(xrng, centroids, 'g.', markersize = 2.0)
			#plt.gca().invert_yaxis(), plt.xlim(xrng[0],xrng[-1])
			#plt.title('Centroid-fit and peak-fit trace function'), plt.xlabel('X Pixel'), plt.ylabel('Y Pixel')
			#plt.legend(handles = [c,f], loc=1, prop = {'size':11})
	
			#plt.figure(2, figsize = (10,5))
			#plt.plot(xrng, trc_fn - yvals, 'ko', markersize = 2.0)
			#plt.xlim(xrng[0],xrng[-1])
			#plt.title('Residuals between data and fit to centroid fits'), plt.xlabel('X Pixel')
			#plt.ylabel('Y Pixel (Centroid - Data)')
			#plt.show()
			#pdb.set_trace()
		
			counts, yvals, centroids, background_levels = [], [], [], [] #Reset the arrays for the next order in loop
		if WRITE:
			hdulist = fits.HDUList()
			f1 = fits.ImageHDU(trace_arr, name = 'Trace function')
			f2 = fits.ImageHDU(prof_shape, name = 'Profile fitting')
			hdulist.append(f1), hdulist.append(f2)
			hdulist.writeto(str(FILEWRITE)+'_TRC.fits', overwrite = True)
			print 'Writing file '+str(FILEWRITE)+'_TRC.fits'
			hdulist.close()
			print '\n ~-# Spectral orders traced: '+str(len(odr_start))+' #-~\n'
		return trace_arr, prof_shape




def flat(FILENAME, FILEWRITE, HDR, WINDOW, WRITE = True):
    """
    This function creates normalized flat images of echelle spectra.\n
    FILENAME: Name of raw data file
    FILEWRITE: User-designated name of reduced data file
    HDR: Header index of raw image in FILENAME
    WINDOW: Size of the smoothing window used in scipy.signal.medfilt()
    WRITE: True: Save image to FILEWRITE / False: Do not save to FILEWRITE\n
    """
    print 'Creating normalized flat image...'
    flat_img = fits.open(str(FILENAME))[HDR].data
    model_flat = np.zeros(flat_img.shape) #Create an array that is the same shape as the image
    for i in range(flat_img.shape[1]):
		flat_col = flat_img[:,i]
		med_flat = scipy.signal.medfilt(flat_col, WINDOW)
		model_flat[:,i] = med_flat
    print 'Dividing the trimmed flat by the median smoothed model flat...'
    norm_flat = flat_img / model_flat #The quantum efficiency based on the modelling method
    if WRITE:
        hdu = fits.HDUList()
        flat = fits.ImageHDU(flat_img, name = 'Original flat')
        norm = fits.ImageHDU(norm_flat, name = 'Normalized Flat')
        hdu.append(norm), hdu.append(flat)
        print 'Writing file: ', str(FILEWRITE)+'_FLT.fits'
        hdu.writeto(str(FILEWRITE)+'_FLT.fits', overwrite = True)
    print '\n~-# Normalized flat image created #-~\n'
    return norm_flat, flat_img
    #Returns the input flat image, the two quantum efficiencies, and the two modelling methods




def spectext(IMAGE, NFIB, TRACE_ARR, YSPREAD, FILEWRITE, CAL = False):
    """
    This function integrates an echelle image along the trace function of each order to collapse it from 2D to 1D.\n
    IMAGE: 2-D image array containing echelle orders
    NFIB: Number of fibers
    TRACE_ARR: 2-D array of number of columns x number of orders containing the coordinates of the orders
    YSPREAD: Approximate width of order. Calibration spectra are typically wider than science spectra
    FILEWRITE: User-designated name of reduced data file
    CAL: True if the spectrum is a calibration spectrum\n
    """
    hdu = fits.HDUList()
    for f in range(1,NFIB+1):
        trace_path, xrng = TRACE_ARR[f-1::NFIB,:], np.arange(0,IMAGE.shape[1],1)
        signal, spec_flat = np.zeros((len(trace_path),len(xrng))), np.zeros((len(trace_path),len(xrng)))
	if NFIB == 1:
		print 'Using trace coordinates to locate spectrum along each order...'
	else:
		print 'Using trace coordinates to locate spectrum on fiber '+str(f)+' of '+str(NFIB)+'...'
        # Integrate the counts over a length of the column along each point in the trace function #
	for j in range(len(trace_path)):
		for i in xrng:
			dy = np.arange(trace_path[j][i]-YSPREAD,trace_path[j][i]+YSPREAD,1)
			dy_int = dy.astype(int)
			bottom, top, out_rng = 0, IMAGE.shape[0], []
			for p in range(len(dy_int)): #Remove indices beyond the range of the detector
				if dy_int[p] <= bottom+1:
					out_rng += [p]
				if dy_int[p] >= top-1:
					out_rng += [p]
			dy_int, dy = np.delete(dy_int,out_rng), np.delete(dy,out_rng)
			if CAL == False:
				cts_int_rng = IMAGE[dy_int,i] - np.median(IMAGE[dy_int,i])
			elif CAL == True:
				cts_int_rng = IMAGE[dy_int,i]
			up_pix = np.abs(dy[-1]-dy_int[-1])*IMAGE[dy_int[-1]+1,i] 
		#Take the lower fraction of counts from the uppermost pixel cut through by dy
			low_pix = (1 - np.abs(dy[0]-dy_int[0]))*IMAGE[dy_int[0],i] 
		#Take the upper fraction of counts from the lowermost pixel cut through by dy
			if len(scipy.integrate.cumtrapz(cts_int_rng,dy_int)) == 0:
				pdb.set_trace()
			signal[j,i] = scipy.integrate.cumtrapz(cts_int_rng,dy_int)[-1] + up_pix + low_pix
        # Blaze fitting #
		if CAL == True: #Fit a blaze function to the continuum of an emission arclamp
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
			blaze = peaks(signal[j],0.1,MEAN = mn) #Find top of spectrum to approximate the blaze function
			pks = signal[j][blaze]
			blazed = trace_fit(blaze, pks,7) #Fit blaze function to top of spectrum
			blfn = np.polyval(blazed[1], xrng)
			spec_flat[j] = (signal[j]/blfn)
	spec_f = fits.ImageHDU(spec_flat, name = 'Blaze-corrected spectrum') 
	spec = fits.ImageHDU(signal, name = '1D Spectrum')
	xpix = fits.ImageHDU(xrng, name = 'X pixel')
	hdu.append(xpix), hdu.append(spec), hdu.append(spec_f)
	print 'Writing file: ', str(FILEWRITE)+'_SPEC.fits'
	hdu.writeto(str(FILEWRITE)+'_SPEC.fits', overwrite = True)
	hdu.close()
	print '\n ~-# Spectrum extracted #-~\n'
	return xrng, signal, spec_flat

