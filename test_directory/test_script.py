import pychelle
from pymods import *
import os

os.makedirs(os.path.expanduser('~/HomemadePy/test_directory/Flats/clean/'), exist_ok=True)
os.makedirs(os.path.expanduser('~/HomemadePy/test_directory/Flats/normalized/'), exist_ok=True)

################################################################################
target_file = os.path.expanduser('~/HomemadePy/test_directory/Target/raw_spectrum.fits')
hires_chip = 3 # 3: red chip, 2: green chip, 1: blue_chip
target_directory = os.path.expanduser('~/HomemadePy/test_directory/Target/')
flat_directory = os.path.expanduser('~/HomemadePy/test_directory/Flats/')
cleaned_flats = os.path.expanduser('~/HomemadePy/test_directory/Flats/clean/')
normalized_flats = os.path.expanduser('~/HomemadePy/test_directory/Flats/normalized/')
arclamp_directory = os.path.expanduser('~/HomemadePy/test_directory/ThAr/')
bright_star_directory = os.path.expanduser('~/HomemadePy/test_directory/Bright/')

# Clean flat field images and create a normalized flat
flat_list = np.sort(glob.glob(flat_directory+'*.fits'))
for i in range(len(flat_list)):
    flat_clean = pychelle.clean(flat_list[i], cleaned_flats+flat_list[i][-22:-5]+'_'
                                +str(hires_chip), flip=True, cut=4040, scan=4080,
                                write=True, hdr=hires_chip, HIRES=True)
normalized_flat, average_flat = pychelle.flat(cleaned_flats, normalized_flats
                                              +str(hires_chip), hdr=1, window=9, 
                                              write=True)
# Clean and flat field the ThAr arclamp image
arclamp_clean = pychelle.clean(arclamp_directory+'arclamp.fits', arclamp_directory+'arclamp',
                              flip=True, cut=4040, scan=4080, write=True,
                              hdr=hires_chip, HIRES=True)
arclamp_flat_fielded = arclamp_clean / normalized_flat

# Clean and flat field the bright star image used to define the trace function                                       
bright_clean = pychelle.clean(bright_star_directory+'bright_spectrum.fits', 
                              bright_star_directory+'bright', flip=True, cut=4040, 
                              scan=4080, write=True, hdr=hires_chip, HIRES=True)
bright_flat_fielded = bright_clean / normalized_flat
# Use the bright star image to define the trace function
trace_solution, profile_shape = pychelle.trace(bright_flat_fielded, xstart=0, ystart=0, 
                                               xstep=1, yrange=6, nsig=0.05, 
                                               filewrite=bright_star_directory+
                                               'bright_spectrum_'+str(hires_chip), sep=10, 
                                               write=True, HIRES=True)

# Clean and flat field the target spectrum
target_clean = pychelle.clean(target_file, target_directory+'target', flip=True,
                              cut=4040,  scan=4080, write=True, hdr=hires_chip,
                              HIRES=True)
target_flat_fielded = target_clean / normalized_flat
# Use the bright star's trace solution to extract the target's spectrum to 1D
x_pixel, target_counts, targ_norm_counts = pychelle.spectext(target_flat_fielded, nfib=1,
                                                             trace_arr=trace_solution, yspread=15,
                                                             filewrite = target_directory+'target_'
                                                             +str(hires_chip), cal=False)

# Use the bright star's trace solution to extract the arclamp's spectrum to 1D
x_pixel, arclamp_counts, arc_norm_counts = pychelle.spectext(arclamp_flat_fielded, nfib=1,
                                                             trace_arr=trace_solution, yspread=15,
                                                             filewrite = arclamp_directory+'arclamp_'
                                                             +str(hires_chip), cal=True)

# Plot and save images to their respective directories
number_orders = len(target_counts)
for i in range(number_orders):
    plt.figure(i+1, figsize = (12,10))
    plt.plot(x_pixel, target_counts[i], 'k-', lw = 0.8)
    plt.xlim(0,4040)
    plt.title('Target spectrum at order #'+str(i+1), fontsize = 20)
    plt.ylabel('Counts', fontsize=18)
    plt.xlabel('X pixel', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.savefig(target_directory+'Order_'+str(i+1))
    plt.close()

for i in range(number_orders):
    plt.figure(i+1, figsize = (12,10))
    plt.plot(x_pixel, arclamp_counts[i], 'k-', lw = 0.8)
    plt.xlim(0,4040)
    plt.title('ThAr arclamp spectrum at order #'+str(i+1), fontsize = 20)
    plt.ylabel('Counts', fontsize=18)
    plt.xlabel('X pixel', fontsize=18)
    plt.tick_params(labelsize=15)
    plt.savefig(arclamp_directory+'Order_'+str(i+1))
    plt.close()
