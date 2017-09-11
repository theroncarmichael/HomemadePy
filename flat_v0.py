
'''
#def flat(FILENAME, FILEWRITE, HDR, NSIG, WINDOW, WINDOW2, WRITE = True):
    ncols, nrows = len(model_flat[0,:]), len(model_flat[:,0])
    model_smooth = np.zeros(flat_img.shape)
    model_smooth_broad = np.zeros(flat_img.shape)
#Row median-smooth method
    print 'Applying a row-by-row median smooth filter...'
    for row in np.arange(0,nrows):  #Loop over every row to median-smooth them
        counts = flat_img[row,:][:] #Counts of a slice across a row
        xvals = np.linspace(0,len(counts),len(counts)) #The horizontal (X values) axis in pixels
        rowfilt = scipy.signal.medfilt(counts,WINDOW)
        model_smooth[row,:] = rowfilt
    print 'Applying a second row-by-row median smooth filter...'
    #Use broader median filter after 10-20 pixel smooothing
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

