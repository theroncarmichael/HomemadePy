"""
The script contains the primary functions to reduce 2D echelle images
to 1D spectra.
-----------------
List of functions:
-----------------
clean(), peaks(), trace(), flat(), specext(),
instrumental_profile(), trace_fit(), blaze_fit(), sigma_clip()
"""
import numpy as np
import random
import scipy.stats
import scipy
from scipy.signal import medfilt
import astropy.io.fits as fits
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib as mpl
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb
import glob
######################################################################
import warnings
warnings.simplefilter('ignore', np.RankWarning)

def clean(filename, filewrite, flip, cut, scan, 
          write=True, hdr=0, HIRES=False):
    """
    The clean() function removes NaN values and does a row-by-row 
    subtraction of the overscan region of the image. 
    The wavelength dispersion direction should approximately go 
    from left to right (use flip = True if 
    the echelle orders are vertical). 
    For slit-fed echelle, sky-subtraction is accounted for in 
    pychelle.trace().
    ----------
    Parameters:
    ----------
    filename: String, name or path of raw data file
    filewrite: String, user-designated name or path of 
    reduced data file
    flip: Boolean, True: Rotated image with numpy.rot90()
    cut: Integer, X-pixel value image is trimmed to
    scan: Integer, the X-pixel start of the overscan region
    write: Boolean, True: Save image to ``filewrite``
    hdr: Integer, FITS header index of raw image in ``filename``
    -------
    Returns:
    -------
    2D array, image with overscan region trimmed off
    """
    print('Processing image...')
    image_file = fits.open(str(filename))
    image = image_file[hdr].data.astype(float)
    image_file.close()
    # Remove NaN values
    image = image[[~np.any(np.isnan(image),axis = 1)]]
    if flip: # Rotate the frame so the orders run horizontally
        image = np.rot90(image, k = 1)
    nrows, ncols = image.shape[0], image.shape[1]
    bias, dark = np.zeros(nrows), np.arange(cut,nrows*ncols,ncols) 
    # ``dark`` is the last column of pixels at which this cutoff occurs
    # and only darker areas that are not part of the orders remain. 
    # For example, if there are 50 columns of darkness after the orders
    # end, then ``cut``-``cols`` should equal 50 to remove these areas.
    for i in range(nrows): # loop through the number of rows
        # take row ``i`` and look the in overscan 
        # region parsed with ``[scan:ncols]``
        bias[i] = np.median(image[[i]][0][scan:ncols]) 
    clipped_bias = scipy.stats.sigmaclip(bias) #Remove outliers
    bias_sigma = 5.0*np.std(clipped_bias[0])
    bias_median = np.median(clipped_bias[0])
    bias[bias <= (bias_median-bias_sigma)] = bias_median
    bias[bias >= (bias_median+bias_sigma)] = bias_median    
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
        cleaned_image = fits.ImageHDU(image,
                                      name='Processed 2D Image')
        hdulist.append(prime_hdr)
        hdulist.append(cleaned_image)
        print('Writing file: ', str(filewrite)+'_CLN.fits')
        hdulist.writeto(str(filewrite)+'_CLN.fits', overwrite=True)
    print('\n~-# Image processed #-~ \n')
    return image


def peaks(y, nsig, mean=-1, deviation=-1):
    """
    This functions returns the indices of the maxima of an array.
    The height of the peaks/maxima to be considered can be 
    controlled with ``nsig`` (how many standard deviations
    above some mean a datum is). 
    ----------
    Parameters:
    ----------
    Y: 1D array, Y-values of data
    nsig: Float, the number of standard deviations away from 
    the ``mean`` a ``y`` value in ``y`` must be to qualify as a peak
    mean: Float, manually set a mean value. Default uses the mean of Y
    deviation: Float, manually set the standard deviation default uses
    the standard deviation of Y.
    -------
    Returns:
    -------
    1D array, indices at which maxima occur
    """
    right, left = y-np.roll(y,1), y-np.roll(y,-1) 
    # Shift the elements in ``y`` left and right and subtract 
    # this from the original to check where these values are > 0
    pk = np.where(np.logical_and(right > 0, left > 0))[0]
    if nsig <= 0.0:
        print('Setting ``nsig`` = 1.0 in peaks()')
        nsig = 1.0
    if nsig > 0.0:
        # Verify lists and arrays are not interacting
        if type(y) != type(np.array(0)): 
            y = np.array(y)
        yp = y[pk]
        # Use the input mean or standard deviation or calculate them
        if mean != -1 and deviation == -1:
            mn, std = mean, np.std(yp)
        elif deviation != -1 and mean == -1:
            mn, std = np.mean(yp), deviation
        elif mean != -1 and deviation != -1:
            mn, std = mean, deviation
        else:
            mn, std = np.mean(yp), np.std(yp)
        # Applies ``nsig`` constraint to maxima
        peak = np.where(yp > mn+nsig*std)[0]
        npk = len(peak)                        
        if npk > 0:                            
            # If ``nsig`` is not too high and npk > 0, these
            # maxima are added to an updated maximum index list
            peak_ind = []                      
            for i in peak:
                peak_ind += [pk[i]]
        else:
            peak_ind = []
            print('Relax peak definition; reduce ``nsig`` in peaks()' 
            'or adjust ``xstart`` and ``ystart``' 
            'in trace() to avoid bias region')
    else:
        peak_ind = pk
    return np.array(peak_ind)

def trace_fit(x, y, deg=1):
    """
    This function utilizes numpy's ``polyfit`` and 
    ``polyval`` functions to return parameters of a 
    fit to a curve
    ----------
    Parameters:
    ----------    
    x: 1D array, the x data input used in numpy.polyfit()
    y: 1D array, the y data input used in numpy.polyfit()
    deg: Integer, polynomial degree
    -------
    Returns:
    -------
    Two 1D arrays, 1) polynomial fit, 
    2) parameters for numpy.polyval()    
    """
    line_params = np.polyfit(x, y, deg)
    trc_fnc = np.polyval(line_params, x)
    return trc_fnc, line_params

def sigma_clip(x, y, deg=1, nloops=15, sig=5.0):
    """
    Sigma clip data with criteria from a fit produced
    by numpy's ``polyfit`` and ``polyval`` functions
    ----------
    Parameters:
    ----------      
    x: 1D array, the x data input used in numpy.polyfit()
    y: 1D array, the y data input used in numpy.polyfit()
    deg: Integer, polynomial degree
    nloops: Integer, number of loops to iterate over while clipping
    sig: Integer, number of sigma away from the fit the 
    data are clipped
    -------
    Returns:
    -------
    1D array, parameters for numpy.polyval() of the 
    clipped data    
    """
    y_sig_arr = np.arange(0,nloops,1.0)
    for i in range(1,nloops):
        line_params = np.polyfit(x, y, deg)
        y_fit = np.polyval(line_params, x)
        y_sig = sig*np.std(y-y_fit)
        y_sig_arr[i] = y_sig
        clipped_ind = np.where(np.abs(y-y_fit) <= y_sig)[0]
        # Reset to the newly clipped data
        y, x = y[clipped_ind], x[clipped_ind]
        if np.around(y_sig_arr[i],3) == np.around(y_sig_arr[i-1],3):
            break
    return line_params

def blaze_fit(xrng, spec):
    """
    !!! Non-ideal method of fitting blaze, new one underway !!!
    This function fits a 1D blaze function to each spectral order
    once the order has been integrated to 1D
    ----------
    Parameters:
    ----------
    xrng: 1D array, the x values of the data (typcially pixels along 
    the dispersion direction of the detector)
    spec: 1D array, the integrated values of the order at each x datum
    -------
    Returns:
    -------
    1D array, polynomial fit to blaze function
    """
    # Find the count values for the edges of the order
    if spec[0] < spec[-1]: 
        mn = spec[0]       
    else:                  
        mn = spec[-1]      
    # Find top of spectrum to approximate the blaze function
    blaze = peaks(spec,0.1,mean = mn)
    pks = spec[blaze]
    blfn_params = trace_fit(blaze, pks, deg = 7)[1]
    blfn = np.polyval(blfn_params, xrng)
    return blfn
######################################################################
def gauss_lorentz_hermite_prof(x, mu1=0.0, amp1=1.0, sig=1.0,
                               offset1=1.0, offset2=1.0, c1=1.0,
                               c2=1.0, c3=1.0, c4=1.0, c5=1.0,
                               c6=1.0, c7=1.0, c8=1.0, c9=1.0, 
                               amp2=1.0, gamma=0.5, mu2=0.1):
    gauss = (amp1 * np.exp(-0.5 * ((x - mu1) / sig)**2) 
             + offset1)
    lorentz = (amp2 * (0.5 * gamma) / ((x - mu2)**2 + (0.5 * gamma)**2) 
               + offset2)
    h_poly = (c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
              + c6*x**6 + c7*x**7 + c8*x**8 + c9*x**9)
    return h_poly * gauss + lorentz
######################################################################
def instrumental_profile(image, order_length, trc_fn, gauss_width):
    """ 
    --------------------------------------------
    To Do: Initialize an IP based on 128 columns
    --------------------------------------------
    This function is the algorithm for creating a super-sampled profile 
    of each spectral order. Sample the trace at each column to produce
    a profile shape for the trace using the trace function.
    ----------
    Parameters:
    ----------    
    image: 2D array, The cleaned echelle image
    order_length: Integer, the length of the dispersion direction 
    (number of x pixels)
    trc_fn: 2D array, the trace functions of each spectral order of
    the cleaned image
    gauss_width: Integer, the distance from the center of each order
    -------
    Returns:
    -------
    Tuple with 4 arrays: 1) & 2) A super-sampled instrumental 
    profile in its X/Y coordinates, unsorted X/Y coordinates allow 
    for plotting column-bye-column 
    (each 1-pixel wide, ``gauss_width`` tall cross section),
    3) & 4) A sorted version of 1)/2), necessary for modeling
    """
    sample_length = len(np.arange(int(10-gauss_width),
                                  int(10+gauss_width),1))
    order_sample_arr = np.zeros((order_length, sample_length))
    for x in range(order_length):
        xdata = np.arange(int(trc_fn[x]-gauss_width),
                          int(trc_fn[x]+gauss_width),1) 
        # Select the area along the order to sample
        ydata_ind = np.arange(int(trc_fn[x]-gauss_width),
                              int(trc_fn[x]+gauss_width),1)
        ydata = image[:,x][ydata_ind]
        order_sample_arr[x,:] = ydata
        # Fit a Gaussian to the profile of the order at each column 
        # to normalize the height
        mu, sigma, amp = trc_fn[x], 1.50, np.max(ydata)
        initial_model = models.Gaussian1D(mean = mu, stddev = sigma,
                                          amplitude = amp) 
        # Initialize a 1D Gaussian model from Astropy
        fit_method = fitting.LevMarLSQFitter() 
        # instantiate a fitter, in this case the fitter that uses 
        # the Levenburg Marquardt Least-Squares algorithm
        odr_prof = fit_method(initial_model, xdata, ydata)
        gauss_int = np.sum(odr_prof(xdata))
        order_sample_arr[x,:] = ydata/gauss_int # Normalize
    # Initialize arrays for super-sampled Y coordinates ``xrng_arr`` 
    # and counts at each Y coordinate 
    xrng_arr = np.zeros((len(trc_fn), 2*gauss_width))
    order_prof = np.zeros((len(trc_fn), 2*gauss_width))
    # Floor the trace function to get sub-pixel precision
    pix_order = trc_fn.astype(int)
    x_shift = pix_order[0] - gauss_width
    rect_order = pix_order - trc_fn
    for i in range(len(trc_fn)):
        xrng_fit = np.arange(rect_order[i]-gauss_width,
                             rect_order[i]+gauss_width,1.0)
        xrng_arr[i] = xrng_fit
        order_prof[i] = order_sample_arr[i,:]
    xrng_arr -= np.min(xrng_arr)
    xrng_arr += x_shift
    yrng = np.arange(0,len(trc_fn), 1)
    yrng_arr = np.zeros((len(yrng), 2*gauss_width))
    for t in range(len(yrng_arr[:,0])):
        yrng_arr[t,:] = t
    # Store the super-sampled profile shapes by column and
    # as a sorted continuous function 
    x_long = xrng_arr.reshape((len(trc_fn)*2*gauss_width))
    y_long = order_prof.reshape((len(trc_fn)*2*gauss_width))
    sorted_ind = np.argsort(x_long)
    x_sorted = x_long[sorted_ind]
    y_sorted = y_long[sorted_ind]
    profile_coordinates = (x_long,y_long, x_sorted,y_sorted)
    return profile_coordinates 

def trace(image, xstart, ystart, xstep, yrange, nsig, filewrite, sep,
          write=False, odr=False, MINERVA=False, HIRES=False,
          MIKE=False, cutoff=[0]):
    """
    This function returns the coordinates of the echelle orders.
    First, it sets up the necessary variables and arrays. Next,
    it locates the starting pixel of each order and confirms with
    the user if it has successfully located the correct number of
    orders. After this, each order is looped over, tracing out
    its path accros the image using the approximate center of the
    1D cross-sectional profile to guide it. A Gaussian is fit to
    the 1D profile to re-trace a mean as the final cetner point of
    the order. The instrumental profile is fit in detail with
    ``pychelle.instrumental_profile()``.
    ----------
    Parameters:
    ----------
    image: 2-D image array containg echelle orders
    xstart/ystart: Typically 0 for the corner of the image from which 
    the search for orders begins
    xstep: Number of X-pixels to skip subtracting 1 to include in 
    a fit to the trace, i.e. ``xstep`` = 1 uses all X-pixels
    yrange: Y-pixel range to search for the next part of the order 
    as the X-pixels are looped over
    nsig: The number of standard deviations away from the 
    ``mean`` a ``y`` value in ``y`` must be to qualitfy as a peak
    filewrite: User-designated name of traced data file
    sep: Y-pixel separation between two detected peaks; used to only 
    take one of these adjacent peaks
    write: True: Save image to ``filewrite``
    odr: 1-D array; if the starting Y-pixel of each order is 
    known, input odr
    -------
    Returns:
    -------
    Tuple with 2 arrays: 1) an N x L size array where N = number 
    of orders, L = length of the image, this array contains the X/Y
    coordinates of the centers of the orders. 2) returns the same
    information as ``pychelle.instrumental_profile()``
    """
    print('Locating spectral orders on cleaned image...')
    xrng, yvals = np.arange(1, image.shape[1]+1, xstep), []
    counts, centroids = [], []
    background_levels = []
    rect_image = np.zeros(image.shape) 
    blze_image = np.zeros(image.shape)
    odr_start = peaks(image[ystart:image.shape[0], xstart], nsig)
    odr_ind = []
    if odr: # Use input list of order starting locations
        odr_start = odr
    if MINERVA == True: 
        # Account for the curve in orders across the detector 
        # cutting off on the edge and remove them
        #cutoff_order = np.where(np.array(odr_start) - 70 < 0)[0]
        odr_start = np.delete(odr_start, cutoff)
    # Create an empty array for the trace function of each order
    trace_arr = np.zeros((len(odr_start),image.shape[1]//xstep))
    for i in range(len(odr_start)):
        if (np.abs(odr_start[i] - odr_start[i-1]) <= sep and
            len(odr_start) > 1): # Remove pixel-adjacent peak measurements
            odr_ind += [i]       # This avoids double-counting a peak
    odr_start = list(np.delete(odr_start,odr_ind))                              
    if HIRES and np.abs(odr_start[-1] - image.shape[0]) <= 10:
        odr_start = list(np.delete(odr_start, -1))
    trc_cont = input(str(len(odr_start))
                         +' orders found, is this accurate? (y/n): ')
    while trc_cont != 'y' and trc_cont != 'n': 
        trc_cont = input('Enter y or n: ')
        if trc_cont == 'y' or trc_cont == 'n': break
    if trc_cont == 'n':
        print('Starting Y-pixel coordinates of orders: ', odr_start)
        print('Exiting pychelle.trace(). Adjust ``sep``, ``nsig``,'
        'or ``cutoff`` to locate the correct number of full orders.\n')
        return [], [], [], []
    elif trc_cont == 'y':
        print('Starting Y-pixel coordinates of orders: ', odr_start)
        print(str(len(odr_start))+' orders found, scanning each' 
        'column for the pixel coordinates of peaks...\n')
        # Algorithm for order definition via a trace function
        prof_shape, end_trace = [], False
        for o in range(len(odr_start)):
        # Detector-specific hard coded half-width of each order
            if MINERVA:
                dy = 8
            elif HIRES:
                dy = 15
            elif MIKE:
                dy = 6
            # horizontal range ``xrng`` and begin tracing the order 
            # based on the peak coordinates found near the current loop
            for i in xrng:    
                column = image[ystart:image.shape[0],i-1] 
                ypeaks = peaks(column,nsig)
                if len(ypeaks) == 0:
                    print('Starting too close to edge;'
                          'increase ``xstart`` in trace()\n')
                    break
                for y in range(len(ypeaks)):
                    # After the first few (5) peaks are found, this
                    # trend (X,Y pixel coordinates) is what is updated 
                    # and referenced for the rest of the horizontal range 
                    # The trend is an average pixel coordinate value for
                    # 5 preceding peak pixels
                    if (len(yvals) <= 5 and 
                        np.abs((ypeaks[y]+ystart) 
                               - odr_start[o]) <= yrange):
                        ypix = ypeaks[y]   
                        break              
                    else:                  
                        if len(yvals) > 5:
                            ytrend = int(np.mean(yvals[len(yvals)-5:]))
                            # Checking that the next peaks are within
                            # range of the trend
                            if np.abs((ypeaks[y] + ystart) 
                                      - ytrend) <= yrange:
                                ypix = ypeaks[y]         
                                break
                # Fit a Gaussian to the cross-section of each order 
                # to find the center for the trace function
                initial_model = models.Gaussian1D(mean=ypix,
                                                  stddev=1.2,
                                                  amplitude=np.max(
                                                  column[ypix-dy:
                                                         ypix+dy]))
                fit_method = fitting.LevMarLSQFitter()
                xaxis = np.arange(ypix-dy, ypix+dy, 1)
                yaxis = column[ypix-dy:ypix+dy]
                # If ``column`` is indexed at an integer > it's length, 
                # the order is running off the edge of the detector
                if len(xaxis) > len(yaxis):
                    print('\nTruncated order detected\n')
                    odr_start = np.delete(odr_start, o)
                    end_trace = True
                    break
                odr_prof = fit_method(initial_model, xaxis, yaxis)
                background_level = np.median(yaxis)    
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
                centroids += [fit_centroid+ystart]
                yvals += [ypix+ystart]
                counts += [column[ypix]]
                background_levels += [background_level]
                '''
                # Diagnostic plot for difference between Gaussian centroids 
                # and peak location of cross-section of order
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
            # Stop creating new centroids for the trace function    
            if end_trace == True:
                break 
            # Use the centroid values from the Gaussian fits to create 
            # a trace of ``order[o]``
            #yvals = [x+1 for x in yvals] # Correct for indexing 0 to 1
            order_length, gauss_width = len(yvals), 10
            trc_fn = trace_fit(xrng, centroids, deg = 7)[0]
            trace_arr[o] = trc_fn
            print('\n=== Spectral order '+str(o+1)+' traced ===')
            print('Calculating profile shape for order '+str(o+1)+'...')
            prof_shape += [instrumental_profile(image, order_length, 
                                                trc_fn, gauss_width)]
            '''
            # Diagnostic plot for difference between Gaussian centroids 
            # and peak location of cross-section of order
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
            plt.xlabel('Dispersion direction (column number)', fontsize = 14), 
            plt.ylabel('Cross-dispersion direction (row number)', fontsize = 14)
            plt.plot(trc_fn, ls = '-', color = 'royalblue', lw = 2.0, label = 'Trace function')
            plt.plot(yvals, color = 'k', marker = '.', ls = 'None', label = 'Physical peak of order')
            plt.plot(centroids, ls = '-', color = 'orange', lw = 2.0, label = 'Modeled peak of order')
            plt.legend(loc = 2), plt.show()
            pdb.set_trace()
            '''
            # Reset the arrays for the next order in loop
            counts, yvals = [], []
            centroids, background_levels = [], []
        if write:
            hdulist = fits.HDUList()
            trace_array = fits.ImageHDU(trace_arr, 
                                        name='Trace function')
            profile_shape = fits.ImageHDU(prof_shape, 
                                          name='Profile fitting')
            hdulist.append(trace_array)
            hdulist.append(profile_shape)
            hdulist.writeto(str(filewrite)
                            +'_TRC.fits', overwrite=True)
            print('Writing file '+str(filewrite)+'_TRC.fits')
            hdulist.close()
            print('\n ~-# Spectral orders traced: '
            +str(len(odr_start))+' #-~\n')
        return trace_arr, prof_shape

    
def flat(filepath, filewrite, hdr, window, write=True):
    """
    This function creates normalized flat images of echelle spectra.
    ----------
    Parameters:
    ----------    
    filepath: String, name or path of cleaned flat images
    filewrite: String, user-designated name or path of reduced data file
    hdr: Integer, FITS header index of image data in ``filename``
    window: Integer, size of the smoothing window used 
    in scipy.signal.medfilt()
    write: Boolean, True: Save image to ``filewrite``
    -------
    Returns:
    -------
    Two 2D arrays, 1) Normalized flat field image
    2) Averaged flat field image
    """
    print('Creating normalized flat image...')
    file_no = np.sort(glob.glob(filepath+'*CLN.fits'))
    flat_img = np.zeros(fits.open(file_no[0])[hdr].data.shape)
    for f in range(len(file_no)):
        flat_img += fits.open(str(file_no[f]))[hdr].data
    flat_img /= float(len(file_no)) # Average the flat images
    model_flat = np.zeros(flat_img.shape)
    for i in range(flat_img.shape[1]):
        flat_col = flat_img[:,i]
        med_flat = scipy.signal.medfilt(flat_col, window)
        #pdb.set_trace()
        if len(np.where(med_flat <= 0)[0]) > 0:
            med_flat[np.where(med_flat <= 0)[0]] = 1.0
            #pdb.set_trace()
        model_flat[:,i] = med_flat    
    print('Dividing the trimmed flat by '
          'the median smoothed model flat...')
    flat_img[flat_img <= 0.0] = 1.0
    # The quantum efficiency based on the modelling method
    norm_flat = flat_img/model_flat
    #pdb.set_trace()
    if write:
        hdulist = fits.HDUList()
        flat = fits.ImageHDU(flat_img, name='Averaged flat')
        norm = fits.ImageHDU(norm_flat, name='Normalized Flat')
        hdulist.append(norm)
        hdulist.append(flat)
        print('Writing file: ', str(filewrite)+'_norm.fits')
        hdulist.writeto(str(filewrite)+'_norm.fits', overwrite=True)
    print('\n~-# Normalized flat image created #-~\n')
    return norm_flat, flat_img


def spectext(image, nfib, trace_arr, yspread, filewrite, cal=False):
    """
    This function integrates an echelle image along the trace function 
    of each order to collapse it from 2D to 1D.
    ----------
    Parameters:
    ----------    
    image: 2D array, image containing echelle orders
    nfib: Integer, number of fibers (not slit fed)
    trace_arr: 2D array, number of columns timmes number of orders 
    containing the coordinates of the orders
    yspread: Integer, approximate width of order. 
    Calibration spectra can be wider than science spectra
    filewrite: String, user-designated name or path of reduced data file
    cal: Boolean, True: the spectrum is a calibration spectrum
    -------
    Returns:
    -------
    One 1D array & two 2D arrays, 1) The integer pixel values along the
    dispersion direction (length) of the image
    2) The spectrum by each order; size N x L array 
    where N = number of orders, L = length of the image
    3) The blaze-normalized spectrum by each order; size N x L array 
    where N = number of orders, L = length of the image
    """
    hdulist = fits.HDUList()
    for f in range(1,nfib+1):
        trace_path = trace_arr[f-1::nfib,:]
        xrng = np.arange(0,image.shape[1],1)
        signal = np.zeros((len(trace_path),len(xrng)))
        spec_flat = np.zeros((len(trace_path),len(xrng)))
    if nfib == 1:
        print('Using trace coordinates to locate ' 
              'spectrum along each order...')
    else:
        print('Using trace coordinates to locate spectrum on fiber '
              +str(f)+' of '+str(nfib)+'...')
    # Integrate the counts over a length of the column along 
    # the trace function
    for j in range(len(trace_path)):
        for i in xrng:
            dy = np.arange(trace_path[j][i]-yspread,
                           trace_path[j][i]+yspread,1)
            dy_int = dy.astype(int)
            bottom, top, out_rng = 0, image.shape[0], []
            # Remove indices beyond the range of the detector
            for p in range(len(dy_int)):
                if dy_int[p] <= bottom+1:
                    out_rng += [p]
                if dy_int[p] >= top-1:
                    out_rng += [p]
            dy_int = np.delete(dy_int,out_rng) 
            dy = np.delete(dy,out_rng)
            if cal == False:
                # Remove sky background if using slit-fed, otherwise, 
                # simply removes inter-order background
                cts_int_rng = image[dy_int,i]-np.median(image[dy_int,i])
            elif cal == True:
                cts_int_rng = image[dy_int,i]
            '''
            ------------------------------------------------
            To Do: Reject cosmic rays based on 128-column IP
            ------------------------------------------------
            '''
            # Take the lower fraction of counts from 
            # the uppermost pixel cut through by ``dy``
            up_pix = (np.abs(dy[-1]-dy_int[-1])
                      *image[dy_int[-1],i])
            #          image[dy_int[-1]+1,i]
            # Take the upper fraction of counts from 
            # the lowermost pixel cut through by ``dy``
            low_pix = ((1-np.abs(dy[0]-dy_int[0]))
                       *image[dy_int[0],i])
            if len(scipy.integrate.cumtrapz(cts_int_rng,dy_int)) == 0:
                print('Spectrum integration paused')
                pdb.set_trace()
            signal[j,i] = (scipy.integrate.cumtrapz(cts_int_rng,dy_int)[-1]
                           + up_pix + low_pix)
        # Fit a blaze function to the continuum of an emission arclamp
        if cal == True:
            blaze_pars = sigma_clip(xrng, signal[j], deg=7, nloops=30)
            blaze = np.polyval(blaze_pars, xrng)
            low = np.where(signal[j] <= blaze)
            blaze_pars_new = sigma_clip(xrng[low], signal[j][low],
                                        deg=7, nloops=30)
            blfn = np.polyval(blaze_pars_new, xrng)
            spec_flat[j] = (signal[j]/blfn)            
        else:
            '''
            ---------------------------------------------------------
            To Do: Substitute lines 707-714 with pychelle.blaze_fit()
            ---------------------------------------------------------
            '''
            if signal[j][0] < signal[j][-1]:
                mn = signal[j][0]
            else:
                mn = signal[j][-1]
            blaze = peaks(signal[j],0.1,mean = mn)
            pks = signal[j][blaze]
            blazed = trace_fit(blaze, pks,7)
            blfn = np.polyval(blazed[1], xrng)
            spec_flat[j] = (signal[j]/blfn)
    spec_f = fits.ImageHDU(spec_flat, name='Blaze-corrected spectrum') 
    spec = fits.ImageHDU(signal, name='1D Spectrum')
    xpix = fits.ImageHDU(xrng, name='X pixel')
    hdulist.append(xpix)
    hdulist.append(spec)
    hdulist.append(spec_f)
    print('Writing file: ', str(filewrite)+'_SPEC.fits')
    hdulist.writeto(str(filewrite)+'_SPEC.fits', overwrite=True)
    hdulist.close()
    print('\n ~-# Spectrum extracted #-~\n')
    return xrng, signal, spec_flat

######################################################################
