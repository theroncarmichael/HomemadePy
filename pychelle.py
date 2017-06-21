from pymods import *
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from astropy.modeling.models import custom_model
from scipy.special import  erf

def description(func_name='name'):
    if str(func_name) == 'clean':
        details = ['FILENAME: Name of raw data file', 'FILEWRITE: User-designated name of reduced data file',
			'FLIP: True: Rotated image by 90 degrees\n False: Do not rotate image', 'CUT: X-pixel value image is trimmed to',
			'CEILING: Hot pixel value to set to median of overscan/bias region', 'SCAN: The X-pixel value of the start of the overscan region',
			'WRITE: True: Save image to FILEWRITE\n False: Do not save to FILEWRITE', 'HDR: Header index of raw image in FILENAME']
    elif str(func_name) == 'peaks':
        details = ['Y: Y-values of data', 'NSIG: The number of standard deviations away from the MEAN a Y-value in Y must be to qualify as a peak',
			'MEAN: Manually set a mean value. Default uses the mean of Y', 'STDEV: Manually set the standard deviation. Default uses the standard deviation of Y']
    elif str(func_name) == 'trace':
        details = ['IMAGE: 2-D image array containg echelle orders', 'XSTART/YSTART: Typically 0 for the corner of the image from which the search for orders begins',
			'XSTEP: Number of X-pixels to skip subtracting 1 to include in a fit to the trace, i.e. XSTEP = 1 uses all X-pixels (skips 0 pixels)',
			'YRANGE: Y-pixel range to search for the next part of the order as the X-pixels are looped over', 'NSIG: The number of standard deviations away from the MEAN a Y-value in Y must be to qualitfy as a peak',
			'FILEWRITE: User-designated name of traced data file', 'SEP: Y-pixel separation between two detected peaks; used to only take one of these adjacent peaks',
			'WRITE: True: Save image to FILEWRITE\n False: Do not save to FILEWRITE', 'ODR: 1-D array; if the starting Y-pixel values of each order is known, input ODR']
    elif str(func_name) == 'flat':
        details = ['FILENAME: Name of raw data file', 'FILEWRITE: User-designated name of reduced data fi    le' , 'HDR: Header index of raw image in FILENAME', 
			'NSIG: The number of standard deviations away from the MEAN a Y-value in Y must be to qualitfy as a peak', 'WINDOW: Size of the smoothing window used in scipy.signal.medfilt()', 
			'WINDOW2: Second, larger smoothing window for scipy.signal.medfilt()', 'WRITE: True: Save image to FILEWRITE\n False: Do not save to FILEWRITE']
    elif str(func_name) == 'spectext':
        details = ['IMAGE: 2-D image array containing echelle orders', 'NFIB: Number of fibers', 'TRACE_ARR: 2-D array of number of columns x number of orders containing the coordinates of the orders',
			'YSPREAD: Approximate width of order. Calibration spectra are typically wider than science spectra', 'FILEWRITE: User-designated name of reduced data file',
			'CAL: True if the spectrum is a calibration spectrum']
    elif str(func_name) == 'trace_fit':
        details = ['x: X data used in numpy.polyfit()', 'y: Y data used in numpy.polyfit()', 'deg: Polynomial degree']
    else:
        details = ["Specify one of pychelle's funcitons in pychelle.description(): 'clean', 'peaks', 'trace_fit', 'trace', 'flat', 'spectext'"]
    for d in details:
        print d
    pass

def clean(FILENAME, FILEWRITE, FLIP, CUT, CEILING, SCAN, WRITE=True, HDR=0, HIRES = False):
    """
    The clean() function removes NaN values and performs a row-by-row subtraction of the overscan region on the image.\n The wavelength dispersion direction should approximately go from left to right (use FLIP = T if the echelle orders are vertical).\n This function returns a 2D image with the overscan region trimmed away. For slit-fed echelle, sky-subtraction is accounted for in pychelle.trace().
    """
    image_file = fits.open(str(FILENAME)) #Read in raw .FITS file
    image = image_file[HDR].data.astype(float)
    image_file.close()
    image = image[[~np.any(np.isnan(image),axis = 1)]] #Remove NaN values and collapses from 2D to 1D
    if FLIP: #Rotate the frame so the orders run horizontally
        image = np.rot90(image, k = 1)#np.transpose(image)
    nrows, ncols = image.shape[0], image.shape[1] #Number of rows, columns
    bias = np.zeros(nrows) #Create an empty array for the bias values
    dark = np.arange(CUT,nrows*ncols,ncols) 
    #dark is the last column of pixels at which this cutoff occurs and only darker areas that are not part of the orders remain. 
    #For example,if there are 50 columns of darkness after the orders cutoff, then CUT-cols should equal 50 to remove these dark areas.
    print 'Subtracting the bias row-by-row...'
    for i in range(nrows): #i loops through the number of rows, [0] transfers from tuple to 1D array
        bias[i] = np.mean(image[[i]][0][SCAN:ncols])
        image[i,:] -= bias[i] #Find the mean bias of each row from the overscan region
    print 'Trimming off the overscan region...'
    image = np.delete(image, np.s_[CUT:ncols], axis = 1) #CUT is the user-input approximation of which horizontal pixel the orders in the image cutoff at.
    image[image >= CEILING] = np.median(bias)            #Set high points to bias level of the frame
    if HIRES: #Trim the image according to HIRES specs
        image = np.delete(image, np.s_[1030:1070:1] , axis = 0)
        image = np.delete(image, np.s_[0:7:1], axis = 0)
    if WRITE: #Write to file
        hdulist = fits.HDUList()
        hdu = fits.ImageHDU(image, name = 'Reduced 2D Image')
        hdulist.append(hdu)
        print 'Writing file: ', str(FILEWRITE)
        hdulist.writeto(str(FILEWRITE)+'.fits', overwrite = True)
    return image #Returns the trimmed image

def peaks(Y, NSIG, MEAN = -1, STDEV = -1):
    """
    This functions returns the indices of the peaks in an array.\n The height of the peaks to be considered can be controlled with NSIG (how many standard deviations above some mean a datum is). 
    """
    right, left = Y - np.roll(Y,1), Y - np.roll(Y,-1) #Shift the elements in Y left and right and subtract this from the original to check where these values are > 0
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
	line_params = np.polyfit(x, y, deg)
	trc_fnc = np.polyval(line_params, x)
	return trc_fnc, line_params

def sigma_clip(x,y, deg = 1, nloops = 15, SIG = 5.0):
	y_sig_arr = np.arange(0,nloops,1.0)
	for i in range(1,nloops):# Sigma clipping
		line_params = np.polyfit(x, y, deg)
		y_fit = np.polyval(line_params, x)
		y_sig = SIG*np.std(y-y_fit)
		y_sig_arr[i] = y_sig
		clipped_ind = np.where(np.abs(y-y_fit) <= y_sig)[0]
		y, x = y[clipped_ind], x[clipped_ind] #Reset to te newly clipped data
		if np.around(y_sig_arr[i],3) == np.around(y_sig_arr[i-1],3):
			break
	return line_params

def gh_prof(x,mean = 0.0, amp1 = 1.0, amp2 = 1.0, sig = 1.0, gamma = 1.0, offset1 = 0.0, offset2 = 0.0):
	gauss = amp1*np.exp(-0.5*((x-mean)/sig)**2) + offset1
	lorentz = amp2*(0.5*gamma)/((x-mean)**2+(0.5*gamma)**2) + offset2
	return gauss * lorentz

def gs_prof(x,mu = 0.0, amp1 = 1.0, sig = 1.0, offset1 = 0.0, alpha = 1.0):
	gauss = amp1*np.exp(-0.5*((x-mu)/sig)**2) + offset1
	skew = alpha*(1 + erf((x-mu)/sig))
	return gauss * skew

def trace(IMAGE, XSTART, YSTART, XSTEP, YRANGE, NSIG, FILEWRITE, SEP, WRITE=False, ODR=False, PROFIT = True, MINERVA = False, CUTOFF = [0]):
    #############################################################################################################
    ###################################### Initialize the trace function ########################################
    #############################################################################################################
    ##### Create empty arrays for Y pixel coordinates and counts #####
    xrng, yvals, counts, centroids = np.arange(1,IMAGE.shape[1]+1,XSTEP), [], [], []
    rect_image, blze_image = np.zeros(IMAGE.shape), np.zeros(IMAGE.shape)
    ##### Automatically locates the start of each order #####
    odr_start, odr_ind = peaks(IMAGE[YSTART:IMAGE.shape[0], XSTART], NSIG),[]
    if ODR: #Use input list of order starting locations
        odr_start = ODR
    if MINERVA == True: #Account for the curve in orders across the detector cutting off on the edge and remove them
	#cutoff_order =  np.where(np.array(odr_start) - 70 < 0)[0]
        odr_start = np.delete(odr_start, CUTOFF)
    #Create an empty array for the trace function of each order
    trace_arr = np.zeros((len(odr_start),IMAGE.shape[1]/XSTEP))
    for i in range(len(odr_start)):
        if np.abs(odr_start[i] - odr_start[i-1]) <= SEP and len(odr_start) > 1: #Remove pixel-adjacent peak measurements
            odr_ind += [i]                                                      #This avoids double-counting a peak and skewing the trace function
    odr_start = list(np.delete(odr_start,odr_ind))                              #Delete the indices (odr_ind) where a peak is double-counted
    trc_cont = raw_input(str(len(odr_start))+" orders found, is this accurate? (y/n): ")
    while trc_cont != 'y' and trc_cont != 'n': 
	    trc_cont = raw_input('Enter y or n: ')
	    if trc_cont == 'y' or trc_cont == 'n': break
    if trc_cont == 'n':
	    print '\nExiting pychelle.trace(). Adjust SEP or CUTOFF to locate the correct number of full orders.\n'
	    return [], [], [], []
    elif trc_cont == 'y':
	    print str(len(odr_start))+" orders found, scanning each column for the pixel coordinates of peaks..."
	    #Array for the integrated counts of each order
	    signal, spec_flat = np.zeros((len(odr_start),len(xrng))), np.zeros((len(odr_start),len(xrng)))
	    #############################################################################################################
	    ############################ Algorithm for order definition via a trace function ############################
	    #############################################################################################################
	    prof_shape, end_trace = [], False
	    for o in range(len(odr_start)):  #At each pixel coordinate where a peak (the top of an order) was detected
		dy = 7 # ----------------------------- Hard coded half-width of each order; width = 2*dy
		for i in xrng:    #(the values in 'odr_start'),take slices of each column across the
		    column = IMAGE[YSTART:IMAGE.shape[0],i-1] #horizontal range (xrng) and begin tracing the order based on
		    ypeaks = peaks(column,NSIG)   #the peak coordinates found near the currently iterating loop
		    if len(ypeaks) == 0:
			print "Starting too close to edge; increase XSTART in trace()"
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
	    ###################### Fit a Gaussian to the cross-section of each order to find the center for the trace function  ######################
		    initial_model = models.Gaussian1D(mean=ypix, stddev=1.2, amplitude = np.max(column[ypix-dy : ypix+dy])) #Initialize a 1D Gaussian model from Astropy
		    #custom_gauss_skew = custom_model(gs_prof)
		    #custom_gauss_lorentz = custom_model(gh_prof)
		    #initial_model = custom_gauss_lorentz(mean = ypix, amp1 = np.max(column[ypix-dy : ypix+dy]), amp2 = np.max(column[ypix-dy : ypix+dy]), sig = 1.2, gamma = 1.0, offset1 = 5.0, offset2 = 0.1)
		    #gs_initial_model = custom_gauss_skew(mu = ypix, amp1 = 1.0, sig = 1.0, offset1 = 0.0, alpha = 1.0)
		    fit_method = fitting.LevMarLSQFitter() # instantiate a fitter, in this case the fitter that uses the Levenburg Marquardt Least-Squares algorithm
		    xaxis, yaxis = np.arange(ypix-dy, ypix+dy, 1), column[ypix-dy : ypix+dy]
		    if len(xaxis) > len(yaxis): # If column[] is indexed at an integer > it's length, the order is running of the edge of the detector
			    print "Truncated order detected"
			    odr_start = np.delete(odr_start, o)
			    end_trace = True
			    break
		    odr_prof = fit_method(initial_model, xaxis, yaxis) #Fit X and Y data using the initialized 1D Gaussian
		    #g_prof = fit_method(g_model, xaxis, yaxis)
		    #g_skew_prof = fit_method(gs_initial_model, xaxis, yaxis)
		    #plt.plot(xaxis, yaxis, 'bo')
		    #plt.plot(xaxis, odr_prof(xaxis), 'k-')
		    #plt.plot(xaxis, g_prof(xaxis), 'g-')
		    #plt.plot(xaxis, g_skew_prof(xaxis), 'r-')
		    #plt.show()
		    #pdb.set_trace()
		    fit_centroid = float(odr_prof.mean[0])
	    ###################### Add the centroid fit to the array used to fit a trace function ######################
		    centroids += [fit_centroid + YSTART]
		    yvals += [ypix + YSTART]
		    counts += [column[ypix]]
		    ###################### Diagnostic plot for difference between Gaussian centroids and peak location of cross-section of order ######################
		    #x_fine = np.arange(ypix-dy, ypix+dy, 0.1)
		    #d_peak = np.round(np.abs(fit_centroid - xaxis[len(xaxis)/2]),3)
		    #plt.figure(i)
		    #plt.plot(xaxis , yaxis, 'bo')
		    #plt.plot(x_fine, odr_prof(x_fine), 'r-', linewidth = 1.5)
		    #gp, = plt.plot(odr_prof.mean[0], odr_prof.amplitude[0], 'go')
		    #plt.legend([gp] ,['Peak residuals: '+str(d_peak)+' pixels'],  loc=1 , prop = {'size':10})
		    #plt.xlim(xaxis[0], xaxis[-1]), plt.show()
		    #pdb.set_trace()
		    ##################################################################################################################################################
		if end_trace == True:
			break #Stop creating new centroids for the trace function

	    ##########################################################################################################################
	    ###################### Use the centroid values from the Gaussian fits to create a trace of order[o] ######################
	    ##########################################################################################################################
		#yvals = [x+1 for x in yvals] #Correct for indexing 0 to 1
		order_length, gauss_width = len(yvals), 10
		sample_length = len(np.arange(int(10-gauss_width),int(10+gauss_width),1))
		order_sample_arr = np.zeros((order_length, sample_length))
		trc_fn = trace_fit(xrng, centroids, deg = 7)[0]
		trace_arr[o] = trc_fn#Store the trace function for each order
		print '~-# Spectral order '+str(o+1)+' traced #-~'

	    #############################################################################################################
	    ################### Algorithm for creating a super-sampled profile of each spectral order ###################
	    #############################################################################################################
		##### Sample the trace at each column to produce a profile shape for the trace using the trace function #####
		print 'Calculating profile shape for order '+str(o+1)+'...'
		for x in range(order_length):
		    xdata = np.arange(int(trc_fn[x]-gauss_width),int(trc_fn[x]+gauss_width),1) #Select the area along the order to sample
		    ydata_ind = np.arange(int(trc_fn[x]-gauss_width),int(trc_fn[x]+gauss_width),1)
		    ydata = IMAGE[:,x][ydata_ind]
		    order_sample_arr[x,:] = ydata
		############################# Fit a Gaussian to the profile of the order at each column to normalize the height ##################################
		    if PROFIT:
			mu, sigma, amp = trc_fn[x], 1.50, np.max(ydata)
			initial_model = models.Gaussian1D(mean = mu, stddev = sigma, amplitude = amp) #Initialize a 1D Gaussian model from Astropy
			fit_method = fitting.LevMarLSQFitter() # instantiate a fitter, in this case the fitter that uses the Levenburg Marquardt Least-Squares algorithm
			#custom_gauss_lorentz = custom_model(gh_prof)
			#inital_model = custom_gauss_lorentz(mu = mu, amp1 = amp, amp2 = amp, sig = sigma, gamma = 1.0, offset1 = 0.0, offset2 = 0.1)
			odr_prof = fit_method(initial_model, xdata, ydata)
			gauss_int = np.sum(odr_prof(xdata))
			order_sample_arr[x,:] = ydata/gauss_int #Normalize the height of the profile
		############################# Initialize arrays for super-sampled Y coordinates (xrng_arr) and counts at each Y coordinate ##################################
		xrng_arr = np.zeros((len(trc_fn),2*gauss_width))
		order_prof = np.zeros((len(trc_fn),2*gauss_width))
		############################ Floor the trace function to get sub-pixel precision ############################
		pix_order = trc_fn.astype(int)
		x_shift = pix_order[0] - gauss_width
		rect_order = pix_order - trc_fn
		for i in range(len(trc_fn)):
			xrng_fit = np.arange(rect_order[i]-gauss_width, rect_order[i]+gauss_width,1.0)
			xrng_arr[i] = xrng_fit
			order_prof[i] = order_sample_arr[i,:]
		xrng_min = np.min(xrng_arr)
		xrng_arr -= xrng_min
		xrng_arr += x_shift
		yrng = np.arange(0,len(trc_fn),1)
		yrng_arr = np.zeros((len(yrng),2*gauss_width))
		for t in range(len(yrng_arr[:,0])):
			yrng_arr[t,:] = t
		###################### Store the super-sampled profile shapes by column or as a sorted continuous function ######################
		x_long, y_long = xrng_arr.reshape((len(trc_fn)*2*gauss_width)), order_prof.reshape((len(trc_fn)*2*gauss_width))
		sorted_ind = np.argsort(x_long)
		x_sorted, y_sorted = x_long[sorted_ind], y_long[sorted_ind]
		prof_shape += [(x_long,y_long, x_sorted,y_sorted)]
		###################### Diagnostic plot for difference between Gaussian centroids and peak location of cross-section of order ######################
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
		#plt.title('Residuals between data and fit to centroid fits'), plt.xlabel('X Pixel'), plt.ylabel('Y Pixel (Centroid - Data)')
		#plt.show()
		#pdb.set_trace()
		##################################################################################################################################################

		#############################################################################################################
		########## Integrate the counts over a length of the column along each point in the trace function ##########
		#############################################################################################################
		print 'Integrating spectral order '+str(o+1)+'...'
		for col_num in xrng[:-1:]:#[:-1:] exclude the end of the row  #Integrate the counts near the trace function along the orders
		    y_int_rng = np.arange(int(trace_arr[o][col_num])-dy,int(trace_arr[o][col_num])+dy,1)
			#Take a range that is 2*dy in size to integrate the counts for an X pixel coordinate at the trace function
		    y_rng = np.arange(trace_arr[o][col_num]-dy,trace_arr[o][col_num]+dy,1)
			#Take the fractional pixel coordinates to determine what fraction of light to integrate at edges of 2*dy range
		    bottom, top, out_rng = 0, IMAGE.shape[0], []
		    for p in range(len(y_int_rng)): #Remove indices beyond the range of the detector
			if y_int_rng[p] <= bottom+1:
			    out_rng += [p]
			if y_int_rng[p] >= top-1:
			    out_rng += [p]
		    y_int_rng, y_rng = np.delete(y_int_rng,out_rng), np.delete(y_rng,out_rng)
		    cts_int_rng = IMAGE[y_int_rng,col_num]
		    up_pix = np.abs(y_rng[-1]-y_int_rng[-1])*IMAGE[y_int_rng[-1]+1,col_num]
			#Take the lower fraction of counts from the uppermost pixel cut through by dy
		    low_pix = (1 - np.abs(y_rng[0]-y_int_rng[0]))*IMAGE[y_int_rng[0],col_num]
			#Take the upper fraction of counts from the lowermost pixel cut through by dy
		    signal[o,col_num] = scipy.integrate.cumtrapz(cts_int_rng,y_int_rng)[-1] + up_pix + low_pix
			#Integrate the counts and store this value for this X,Y pixel coordinate on the trace function
		#############################################################################################################
		############################################# Blaze fitting #################################################
		#############################################################################################################
		if signal[o][0] < signal[o][-1]: #Find the count        #The trace of each order is fit with tf and the start of this
		    mn = signal[o][0]            #values for the        #trace is where the rectified orders begin. This reduces the
		else:                            #edges of the order    #effect of comic rays at the beginning of the order
		    mn = signal[o][-1]           #to fit the blaze      #misplacing the rectified order

		blaze = peaks(signal[o],0.1,MEAN = mn) #Find top of spectrum to approximate the blaze function
		pks = signal[o][blaze]
		blfn_params = trace_fit(blaze, pks, deg = 7)[1]
		blfn = np.polyval(blfn_params, xrng)

		spec_flat[o] = (signal[o]/blfn)
		counts, yvals, centroids = [], [], [] #Reset the Y pixel coordinate and counts arrays for the next order
		################################################# Diagnostic plots for blaze function fit ########################################################
		#plt.plot(xrng, signal[o], 'k-'), plt.plot(xrng, blfn, 'b-')
		#plt.xlabel('Pixel'), plt.ylabel('Integrated counts')
		#plt.title('Blaze function fit to order'), plt.xlim(xrng[0], xrng[-1]), plt.show()

		#plt.plot(xrng, spec_flat[o], 'k-')
		#plt.title('Blaze-corrected order'), plt.xlabel('Pixel'), plt.ylabel('Normalized counts')
		#plt.xlim(xrng[0], xrng[-1]), plt.show()
		#pdb.set_trace()
		##################################################################################################################################################
	    if WRITE:
		hdulist = fits.HDUList()
		f1, f2 = fits.ImageHDU(trace_arr, name = 'Trace function'), fits.ImageHDU(signal, name = 'Spectrum')
		f3, f4 = fits.ImageHDU(spec_flat, name = 'No Blaze Spectrum'), fits.ImageHDU(prof_shape, name = 'Profile fitting')
		hdulist.append(f1), hdulist.append(f2), hdulist.append(f3), hdulist.append(f4)
		hdulist.writeto(str(FILEWRITE)+'.fits', overwrite = True)
		print 'File '+str(FILEWRITE)+'.fits written'
		hdulist.close()

            return trace_arr, signal, spec_flat, prof_shape

#############################################################################################################
#############################################################################################################
#############################################################################################################
#############################################################################################################

def flat(FILENAME, FILEWRITE, HDR, NSIG, WINDOW, WINDOW2, WRITE = True):
    flat_img = fits.open(str(FILENAME))[HDR].data
    model_flat = np.zeros(flat_img.shape) #Create an array that is the same shape as the image
    model_smooth = np.zeros(flat_img.shape)
    ncols, nrows = len(model_flat[0,:]), len(model_flat[:,0])
    #Use broader median filter after 10-20 pixel smooothing
    model_smooth_broad = np.zeros(flat_img.shape)
#Row median-smooth method
    print 'Applying a row-by-row median smooth filter...'
    for row in np.arange(0,nrows):  #Loop over every row to median-smooth them
        counts = flat_img[row,:][:] #Counts of a slice across a row
        xvals = np.linspace(0,len(counts),len(counts)) #The horizontal (X values) axis in pixels
        rowfilt = scipy.signal.medfilt(counts,WINDOW)
        model_smooth[row,:] = rowfilt
    print 'Applying a second row-by-row median smooth filter...'
    for row in np.arange(0,nrows):  #Loop over every row to median-smooth them
        counts = model_smooth[row,:] #Counts of a slice across a row
        xvals = np.linspace(0,len(counts),len(counts)) #The horizontal (X values) axis in pixels
        rowfilt = scipy.signal.medfilt(counts,WINDOW2)
        model_smooth_broad[row,:] = rowfilt
#Column, flat modelling method ***Requires flats to be sufficiently wider (+10 pixels on each side) than the groups of orders
#                              ***otherwise a sawtooth pattern will emerge
    print 'Modelling the tops of each flat of the median smooth filter...'
    for col in np.arange(0,ncols):  #Loop over every column, slicing down to model the top of each order of the flat
        counts = model_smooth_broad[:,col][:] #Counts of a slice down a column for the median-smoothed image
        yvals = np.linspace(0,len(counts),len(counts)) #The vertical (Y values) axis in pixels
        dy = np.diff(counts, n=1) #Slope of the column of counts
        cts_std = NSIG*np.std(dy) #Level above the mean at which to include a peak in the slope
        yplot = np.delete(yvals, -1) #Adjust the size of yvals to match the array of slopes, dy
        dy_pks = np.where(np.abs(dy) > cts_std)[0] #Slope values higher than the deviation above the mean, cts_std
        pks = dy[dy_pks]

        D = defaultdict(list)
        for i,item in enumerate(pks):
            D[item].append(i)
        D = {k:v for k,v in D.items() if len(v)>1} #Array of duplicate slopes
        false_pks = []
        for key, value in D.items():
            if value[0]+1 == value[1]:
                false_pks += value #Indices of adjacent duplicate values that disguise a peak as normal data
                #print 'Duplicate slopes found in column '+str(col)
                #print [false_pks]
        false_pks = np.array(false_pks)

        if len(false_pks) > 1:
            pks[false_pks[::2]] -= 1000 #Offset successive points of equal slope to create a true peak for peaks()
        ypks = yplot[dy_pks] #Y pixels for X-axis array

        tp, bp = peaks(pks,NSIG, MEAN = 0.0, STDEV = 300), peaks(-1*pks, NSIG, MEAN = 0.0, STDEV = 300) #Y pixels where slopes are steep
        turns = np.sort(list(tp)+ list(bp)) #Sort the arrays of steepest slopes into one array

        while pks[turns][-1] > 0: #Account for partial orders
            turns = np.delete(turns,-1)
            tp = np.delete(tp,-1) #tp is positive slopes
        while pks[turns][0] < 0:
            turns = np.delete(turns,0)
            bp = np.delete(bp,0) #bp is negative slopes

        if len(bp) < len(tp): #Account for partial orders
            slope_sides = np.arange(0,len(bp))
        elif len(tp) < len(bp):
            slope_sides = np.arange(0,len(tp))
        elif len(bp) == len(tp):
            slope_sides = np.arange(0,len(bp))

	side_ind = ypks[turns].astype(int)
	#Remove mis-counted orders
	yt, yb = np.diff(ypks[tp]).astype(int), np.diff(ypks[bp]).astype(int)
	xt = np.linspace(0,len(yt), len(yt))
	xb = np.linspace(0,len(yb), len(yb))
	c1 = np.polyval(sigma_clip(xt,yt, SIG = 2.), xt)
	c2 = np.polyval(sigma_clip(xb,yb, SIG = 2.), xb)
	#c1 = trace_fit(xt, yt)[0]
	#c2 = trace_fit(xb, yb)[0]
	res1, res2 = c1-yt, c2-yb
	#plt.plot(yt,'bo'), plt.plot(yb, 'r.')
	#plt.plot(c1,'b'), plt.plot(c2,'r')
	#plt.show()

	## USe trace solution to flat field instead of slopes
	######Current solution for dead pixels
	rm_tp = np.where(np.abs(res1) >= 5.0)[0]
	rm_bp = np.where(np.abs(res2) >= 5.0)[0]
	#tp, bp = np.delete(tp, rm_tp), np.delete(bp, rm_bp)
	######

	#plt.plot(yt,'bo'), plt.plot(yb, 'r.')
	#plt.plot(c1,'b'), plt.plot(c2,'r')
	#plt.show()
	#pdb.set_trace()

	bad_pix = side_ind[np.where(np.diff(side_ind) < 10)[0]]
	#side_x = np.linspace(0,len(side_ind),len(side_ind)-1)
	#res = trace_fit(side_x, np.diff(side_ind))[0] - np.diff(side_ind)
	#rm_ind = np.where(np.abs(res) > 2.0*np.std(res))[0]
	#new_side_ind = np.delete(turns, rm_ind)
	#tp, bp = new_side_ind[::2], new_side_ind[1::2]
	slope_sides = np.arange(0,len(bp))

	if (len(tp) + len(bp)) % 2 == 1:
		print len(tp), len(bp)
		print 'Odd number of slopes detected'
		pdb.set_trace()
		side_x = np.linspace(0,len(side_ind),len(side_ind)-1)
		plt.figure(1)
		plt.plot(trace_fit(side_x, np.diff(side_ind))[0], 'k--')
		plt.plot(np.diff(side_ind), 'bo')
		plt.figure(2)
		plt.plot(trace_fit(side_x, np.diff(side_ind))[0] - np.diff(side_ind), 'ro')
		plt.plot([side_x[0],side_x[-1]], [2.0*np.std(res), 2.0*np.std(res)], 'k--')
		plt.plot([side_x[0],side_x[-1]], [-2.0*np.std(res), -2.0*np.std(res)], 'k--')
		res = trace_fit(side_x, np.diff(side_ind))[0] - np.diff(side_ind)
		plt.show()
		pdb.set_trace()

	plt.figure(figsize = (11,11))
        plt.plot(yvals,counts,'w-'), plt.gca().set_axis_bgcolor('black'),plt.title('Column '+str(col)+'')
        plt.plot(yvals[side_ind], counts[side_ind], 'go')
	plt.plot(yvals[bad_pix], counts[bad_pix], 'co')
        plt.xlim(0,len(yvals)),plt.xlabel('Y axis (Pixels)'),plt.ylabel('Z axis (Counts)')
        ##################################################################################################################################################
        for i in range(len(slope_sides)):
            #print int(ypks[tp[i]]), int(ypks[bp[i]])
            top_flt_yvs, top_flt_cts = yvals[int(ypks[tp[i]]):int(ypks[bp[i]])], counts[int(ypks[tp[i]]):int(ypks[bp[i]])] #top of the counts
            top_cts_cutoff = np.median(top_flt_cts) - 5*np.sqrt(top_flt_cts) #3 sigma cutoff above which data is included in the median
            top_flt_yvs, top_flt_cts = top_flt_yvs[np.where(top_flt_cts >= top_cts_cutoff)[0]], top_flt_cts[np.where(top_flt_cts >= top_cts_cutoff)[0]]

            if len(top_flt_yvs) == 0:
                print 'Partial order potentially found'
                print 'Number of rising slopes: ', len(tp), 'Number of falling slopes: ',len(bp)
                print 'Column number: ',col
                pdb.set_trace()

            parms = np.polyfit(top_flt_yvs, top_flt_cts, 1) #Model the top of each order
            fit_cts = np.polyval(parms, top_flt_yvs)#parms[0]*top_flt_yvs + parms[1]
            top_flt_yvs = np.round(top_flt_yvs).astype(int)

            mid_flt = np.median(top_flt_yvs)
            model_flat[top_flt_yvs,col] = fit_cts #Store the model in a 2D array

            plt.plot(top_flt_yvs, top_flt_cts, 'y.')
            plt.plot(top_flt_yvs, fit_cts, 'r-', linewidth= 2.25)
            plt.plot(top_flt_yvs, top_flt_cts/fit_cts, 'w.')
        plt.show()
	pdb.set_trace()

    print 'Dividing the trimmed flat by the median smoothed model flat...'
    QE = flat_img/model_flat #The quantum efficiency based on the modelling method
    QE_SM = flat_img/model_smooth_broad #The quantum efficiency based on the median-smooth method
    if WRITE:
        hdu = fits.HDUList()
        flat = fits.ImageHDU(flat_img, name = 'Original flat')
        qe_sm = fits.ImageHDU(QE_SM, name = 'Quantum Efficiency, Median Smooth Flat')
        qe = fits.ImageHDU(QE, name = 'Quantum Efficiency, Model Flat')
        flat_s = fits.ImageHDU(model_smooth_broad, name = 'Median Smooth flat')
        flat_m = fits.ImageHDU(model_flat, name = 'Median Smooth and Modelled Flat')
        hdu.append(flat), hdu.append(qe_sm), hdu.append(qe), hdu.append(flat_s), hdu.append(flat_m)
        print 'Writing file: ', str(FILEWRITE)
        hdu.writeto(str(FILEWRITE), clobber = True)

    return flat_img, QE_SM, QE, model_smooth_broad, model_flat
    #Returns the input flat image, the two quantum efficiencies, and the two modelling methods

def spectext(IMAGE, NFIB, TRACE_ARR, YSPREAD, FILEWRITE, CAL = False):
    hdu = fits.HDUList()
    for f in range(1,NFIB+1):
        trace_path, xrng = TRACE_ARR[f-1::NFIB,:], np.arange(0,IMAGE.shape[1],1)
        signal, spec_flat = np.zeros((len(trace_path),len(xrng))), np.zeros((len(trace_path),len(xrng)))
	print 'Using trace coordinates to locate spectrum on fiber '+str(f)+' of '+str(NFIB)+'...'
        ########## Integrate the counts over a length of the column along each point in the trace function ##########
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
                cts_int_rng = IMAGE[dy_int,i]
                up_pix = np.abs(dy[-1]-dy_int[-1])*IMAGE[dy_int[-1]+1,i] #Take the lower fraction of counts from the uppermost pixel cut through by dy
                low_pix = (1 - np.abs(dy[0]-dy_int[0]))*IMAGE[dy_int[0],i] #Take the upper fraction of counts from the lowermost pixel cut through by dy
                if len(scipy.integrate.cumtrapz(cts_int_rng,dy_int)) == 0:
                    pdb.set_trace()
                signal[j,i] = scipy.integrate.cumtrapz(cts_int_rng,dy_int)[-1] + up_pix + low_pix
        ############################################# Blaze fitting #################################################
	    if CAL == True: #Fit a blaze function to the continuum of an emission arclamp
	    	blaze_pars = sigma_clip(xrng, signal[j], deg = 7, nloops = 30)
		blaze = np.polyval(blaze_pars, xrng)
		low = np.where(signal[j] <= blaze)
		blaze_pars_new = sigma_clip(xrng[low], signal[j][low], deg = 7, nloops = 30)
		blfn = np.polyval(blaze_pars_new, xrng)
		spec_flat[j] = (signal[j]/blfn)
	    else:
	    	if signal[j][0] < signal[j][-1]: #Find the count        #The trace of each order is fit with tf and the start of this
        		mn = signal[j][0]            #values for the        #trace is where the rectified orders begin. This reduces the
            	else:                            #edges of the order    #effect of comic rays at the beginning of the order
                  	mn = signal[j][-1]           #to fit the blaze      #misplacing the rectified order
          	blaze = peaks(signal[j],0.1,MEAN = mn) #Find top of spectrum to approximate the blaze function
                pks = signal[j][blaze]
                blazed = trace_fit(blaze, pks,7) #Fit blaze function to top of spectrum
	        blfn = np.polyval(blazed[1], xrng)
                spec_flat[j] = (signal[j]/blfn)
    spec_f, spec, xpix = fits.ImageHDU(spec_flat, name = 'Blazr-corrected spectrum'), fits.ImageHDU(signal, name = '1D Spectrum'), fits.ImageHDU(xrng, name = 'X pixel')
    hdu.append(xpix), hdu.append(spec), hdu.append(spec_f)
    print 'Writing file: ', str(FILEWRITE)
    hdu.writeto(str(FILEWRITE), overwrite = True)
    hdu.close()
    return xrng, signal, spec_flat
