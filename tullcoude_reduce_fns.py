###################################################################################################
##### Functions for reducing Tull coude data #####
##### Daniel Krolikowski, Aaron Rizzuto #####
##### Optical echelle spectra #####
###################################################################################################

##### Imports #####

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.time import Time
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages

import glob
import os
import pickle
import mpyfit
import pdb
import astroscrappy

###################################################################################################
###### FUNCTIONS ######
###################################################################################################

## Function: Make the header information file ##
def Header_Info( Conf ):

    files   = np.sort( glob.glob( '*.fits' ) ) # Get list of fits file names

    outfile = open( Conf.InfoFile, 'w' ) # Open up the header info file to write to

    # Make the column name row for the header info file
    heading = 'File,Object,RA,DEC,Type,ExpTime,UTdate,UT,Airmass,gain,rdn,zenith,emflux,Order\n'
    outfile.write( heading )

    # Loop through all of the files
    for f in range( len( files ) ):
        head  = fits.open(files[f])[0].header # Read in the header

        itype = head['imagetyp'] # First up image type (like object, comp, etc)
        if len( files[f].split('.') ) > 2:
            itype = 'unknown' # Don't quite know what this is a catch for, some other fits file

        ra, dec = '', '' # Coordinates now!
        if 'RA'  in head.keys():
            ra  = head['RA']
        if 'DEC' in head.keys():
            dec = head['DEC']

        air = '' # Airmass now!
        if 'AIRMASS' in head.keys():
            air = str( head['airmass'] )

        exp_time = str( head['exptime'] ) # Get the exposure time

        if 'GAIN3' in head.keys(): # Get the gain and read noise
            gain, rdn = str( head['gain3'] ), str( head['rdnoise3'] )
        else:
            gain, rdn = str( head['gain2'] ), str( head['rdnoise2'] )

        emflux = '' # The emeter flux if it is there!
        if 'EMFLUX' in head.keys():
            emflux = str( head['emflux'] )

        line = ','.join( [ files[f], head['object'], ra, dec, itype, exp_time, head['DATE-OBS'], head['UT'], air, gain, rdn, head['ZD'], emflux, head['order'] ] )
        outfile.write( line + '\n' )

    outfile.close()

    return None

##### Section: Basic calibration functions (bias, flat, bad pixel mask) #####

## Function: Make the super bias frame ##
def Build_Bias( files, read_noise ):

    for i_file, file in enumerate( files ):
        frame = fits.open( file )[0].data
        if i_file == 0:
            bias_cube = np.zeros( ( len( files ), frame.shape[0], frame.shape[1] ) )
        bias_cube[i_file] = frame

    super_bias = {}

    super_bias['vals'] = np.nanmedian( bias_cube, axis = 0 )
    super_bias['errs'] = np.sqrt( super_bias['vals'] + read_noise ** 2.0 )

    return super_bias

## Function: Make the combined flat frame ##
def Build_Flat( files, read_noise, super_bias ):

    for i_file, file in enumerate( files ):
        frame = fits.open( file )[0].data
        if i_file == 0:
            flat_cube = np.zeros( ( len( files ), frame.shape[0], frame.shape[1] ) )
        flat_cube[i_file] = frame

    super_flat = {}

    flat_median = np.nanmedian( flat_cube, axis = 0 )
    flat_values = flat_median - super_bias['vals']     # Subtract off the bias from the flat!

    super_flat['errs'] = np.sqrt( flat_median + read_noise ** 2.0 + super_bias['errs'] ** 2.0 ) # Photon noise and bias error propagation
    super_flat['vals'] = flat_values - flat_values.min()

    super_flat['errs'] = super_flat['errs'] / super_flat['vals'].max()
    super_flat['vals'] = super_flat['vals'] / super_flat['vals'].max()

    return super_flat

## Function: Make the bad pixel mask ##
def Make_BPM( super_bias, super_flat, Conf ):

    cut_bias    = np.percentile( super_bias, Conf.bpm_limit )
    bad_pix_map = np.where( ( super_bias > cut_bias ) | ( super_flat <= 0.0001 ) )

    return bad_pix_map

## Function: Call sub-functions and return bias, flat, BPM ##
def Basic_Cals( bias_inds, flat_inds, file_info, Conf ):

    if Conf.doCals: # If we need to generate calibration files
        # Create super bias
        if Conf.verbose: print( 'Reading Bias Files' )
        bias_rdn   = file_info['rdn'].values[bias_inds] / file_info['gain'].values[bias_inds]
        super_bias = Build_Bias( file_info['File'].values[bias_inds], bias_rdn[0] )
        pickle.dump( super_bias, open( Conf.rdir + 'cal_files/bias.pkl', 'wb' ) )

        # Create super flat
        if Conf.verbose: print( 'Reading Flat Files' )
        flat_rdn   = file_info['rdn'].values[flat_inds] / file_info['gain'].values[flat_inds]
        super_flat = Build_Flat( file_info['File'].values[flat_inds], flat_rdn[0], super_bias )
        pickle.dump( super_flat, open( Conf.rdir + 'cal_files/flat.pkl', 'wb' ) )

        # Create the bad pixel mask
        if Conf.verbose: print( 'Creating the bad pixel mask' )
        bad_pix_map = Make_BPM( super_bias['vals'], super_flat['vals'], Conf )
        pickle.dump( bad_pix_map, open( Conf.rdir + 'cal_files/bpm.pkl', 'wb' ) )

        # Plots!
        plt.clf() # Plot the bias
        plt.imshow( np.log10( super_bias['vals'] ), cmap = 'gray', aspect = 'auto', interpolation = 'none', vmin = np.median( np.log10( super_bias['vals'] ) ),
                    vmax = np.percentile( np.log10( super_bias['vals'] ), Conf.bpm_limit ) )
        plt.colorbar()
        plt.title( 'Bias ' + str( np.nanmedian( super_bias['vals'] ) ) )
        plt.savefig( Conf.rdir + 'cal_files/bias.pdf' )
        plt.clf()

        plt.clf() # Plot the flat
        plt.imshow( np.log10( super_flat['vals'] ), cmap = 'gray', aspect = 'auto', interpolation = 'none' )
        plt.colorbar(); plt.title( 'Flat Field' )
        plt.savefig( Conf.rdir + 'cal_files/flat.pdf' )
        plt.clf()

        plt.clf() # Plot the BPM
        plt.imshow( np.log10( super_bias['vals'] ), cmap = 'gray', aspect = 'auto', interpolation = 'none', vmin = np.median( np.log10( super_bias['vals'] ) ),
                    vmax = np.percentile( np.log10( super_bias['vals'] ), Conf.bpm_limit ) )
        plt.plot( bad_pix_map[1], bad_pix_map[0], ',', c = '#dfa5e5', ms = 1 ) # Invert x,y for imshow
        plt.colorbar(); plt.title( 'Bad Pixel Map on Bias' )
        plt.savefig( Conf.rdir + 'cal_files/bpm.pdf' )
        plt.clf()

    elif not Conf.doCals: # If we're reading in already generated cal files
        if Conf.verbose: print( 'Reading in already generated Bias, Flat, and BPM files' )
        super_bias  = pickle.load( open( Conf.rdir + 'cal_files/bias.pkl', 'rb' ), encoding = 'latin' )
        super_flat  = pickle.load( open( Conf.rdir + 'cal_files/flat.pkl', 'rb' ), encoding = 'latin' )
        bad_pix_map = pickle.load( open( Conf.rdir + 'cal_files/bpm.pkl', 'rb' ), encoding = 'latin' )

    return super_bias, super_flat, bad_pix_map

##### Section: Generate data cubes -- 2D spectral cubes for arc and object exposures #####

## Function: Make the data cubes! With cosmic subtraction, error propagation ##
def Make_Cube( files, read_noise, gain, dark_curr, Conf, bias = None, flat = None, bpm = None ):

    for i_file, file in enumerate( files ): # Loop through all files
        frame = fits.open( file )[0].data

        if i_file == 0: # If it is the first file set the empty arrays for the cubes
            val_cube = np.zeros( ( len( files ), frame.shape[0], frame.shape[1] ) )
            snr_cube = np.zeros( ( len( files ), frame.shape[0], frame.shape[1] ) )

        val_cube[i_file] = frame - dark_curr[0] # Cube value, subtract dark if there is one (which I'm not doing)
        err_vals = np.sqrt( val_cube[i_file] + dark_curr[0] + read_noise[i_file] ** 2.0 ) # Noise

        # Perform cosmic subtraction, if specified. Using astroscrappy now instead of cosmics (which works in py3 and is about 2x faster it seems)
        if Conf.cosmic_sub:
            cos_mask, clean_frame = astroscrappy.detect_cosmics( val_cube[i_file], gain = gain[i_file], readnoise = read_noise[i_file], niter = Conf.niter_cosmic_sub,
                                                                 sigclip = 5, sigfrac = 0.3, objlim = 5, satlevel = np.inf )
            val_cube[i_file] = clean_frame

        CBerrVal = err_vals ** 2.0 / val_cube[i_file] ** 2.0
        FerrVal  = 0.0

        if bias is not None: # If we're subtracting off the bias
            val_cube[i_file] -= bias['vals']
            CBerrVal = ( err_vals ** 2.0 + bias['errs'] ** 2.0 ) / val_cube[i_file] ** 2.0
        if flat is not None: # If we're dividing out the flat
            val_cube[i_file] /= flat['vals']
            FerrVal = ( flat['errs'] / flat['vals'] ) ** 2.0

        full_err = np.sqrt( val_cube[i_file] ** 2.0 * ( CBerrVal + FerrVal ) ) # Total Error

        snr_cube[i_file] = val_cube[i_file] / full_err # Compute the signal to noise cube

        if bpm is not None: # If there's a bad pixel
            val_cube[i_file,bpm[0],bpm[1]] = np.nanmedian( val_cube[i_file] )  # Is setting a bad pixel to the median right? (Should it just be a nan?)
            snr_cube[i_file,bpm[0],bpm[1]] = 1e-4 # Setting the S/N to effectively 0 (so shouldn't matter what the value of a bad pixel is?)

        nan_loc = np.where( np.isnan( val_cube[i_file] ) )
        val_cube[i_file,nan_loc[0],nan_loc[1]] = np.nanmedian( val_cube[i_file] ) # I might as well just do the same as for the BPM
        snr_cube[i_file,nan_loc[0],nan_loc[1]] = 1e-4 # Set a nan pixel to have an effectively 0 S/N

    return val_cube, snr_cube

## Function: Call Make_Cube to return arc and object cubes
def Return_Cubes( arc_inds, obj_inds, file_info, dark_curr_arr, super_bias, super_flat, bad_pix_map, Conf ):

    suffix = ''
    if Conf.cosmic_sub: suffix = '_cossub'

    if Conf.doCubes:
        if Conf.verbose: print( 'Generating arc spectral cubes' )
        rdn_vals  = file_info['rdn'].values[arc_inds] / file_info['gain'].values[arc_inds]
        dark_vals = dark_curr_arr[arc_inds] / file_info['gain'].values[arc_inds]
        gain_vals = file_info['gain'].values[arc_inds]
        arc_val_cube, arc_snr_cube = Make_Cube( file_info['File'].values[arc_inds], rdn_vals, gain_vals, dark_vals, Conf, bias = super_bias )

        if Conf.verbose: print( 'Generating object spectral cubes' )
        rdn_vals  = file_info['rdn'].values[obj_inds] / file_info['gain'].values[obj_inds]
        dark_vals = dark_curr_arr[obj_inds] / file_info['gain'].values[obj_inds]
        gain_vals = file_info['gain'].values[obj_inds]
        obj_val_cube, obj_snr_cube = Make_Cube( file_info['File'].values[obj_inds], rdn_vals, gain_vals, dark_vals, Conf, bias = super_bias, flat = super_flat, bpm = bad_pix_map )

        pickle.dump( arc_val_cube, open( Conf.rdir + 'spec_cubes/cube_arc_val' + suffix + '.pkl', 'wb' ) ) # Write out the spectral cubes for ease (and only have to run cosmics once)
        pickle.dump( arc_snr_cube, open( Conf.rdir + 'spec_cubes/cube_arc_snr' + suffix + '.pkl', 'wb' ) )
        pickle.dump( obj_val_cube, open( Conf.rdir + 'spec_cubes/cube_obj_val' + suffix + '.pkl', 'wb' ) )
        pickle.dump( obj_snr_cube, open( Conf.rdir + 'spec_cubes/cube_obj_snr' + suffix + '.pkl', 'wb' ) )

    else:
        arc_val_cube = pickle.load( open( Conf.rdir + 'spec_cubes/cube_arc_val' + suffix + '.pkl', 'rb' ) ) # Read in spectral cubes if they have already been generated
        arc_snr_cube = pickle.load( open( Conf.rdir + 'spec_cubes/cube_arc_snr' + suffix + '.pkl', 'rb' ) )
        obj_val_cube = pickle.load( open( Conf.rdir + 'spec_cubes/cube_obj_val' + suffix + '.pkl', 'rb' ) )
        obj_snr_cube = pickle.load( open( Conf.rdir + 'spec_cubes/cube_obj_snr' + suffix + '.pkl', 'rb' ) )

    return arc_val_cube, arc_snr_cube, obj_val_cube, obj_snr_cube

##### Section: Get the trace! #####

## Function: Get peak values of trace for a slice of the flat ##
def Start_Trace( flat_slice, grad_perc_cut ):

    flat_slice_grad = np.gradient( flat_slice )
    grad_cut_val    = np.nanpercentile( abs( flat_slice_grad ), grad_perc_cut )

    order_zeros = []
    last_peak   = 0

    # Find peaks based on the gradient of the flat slice
    for i_pix in range( 6, flat_slice.shape[0] ): # Start at 6 to cut out the very start, although probably doesn't matter if it is 0 or 6 or 12
        if flat_slice_grad[i_pix] > grad_cut_val or last_peak == 0:
            if 100 > i_pix - last_peak > 20 or last_peak == 0:
                order_zeros.append( i_pix + 11 )
                last_peak = i_pix
    order_zeros = np.array( order_zeros )

    # Here we will recenter the peaks that have been found!
    for i_ord, ord_loc in enumerate( order_zeros ):
        flat_cut_val = flat_slice[ord_loc] * ( 0.7 ) # The value for like the edges of the order peak to center on (70% of the current peak value)

        left  = ord_loc - 15 + np.where( flat_slice[ord_loc-15:ord_loc] <= flat_cut_val )[-1] # Use 15 as a rough peak width
        right = ord_loc + np.where( flat_slice[ord_loc:ord_loc+20] <= flat_cut_val )[-1]

        if len( left ) == 0 or len( right ) == 0:
            order_zeros[i_ord] = ord_loc # If it just doesn't find the edges use what is already there!
        else:
            order_zeros[i_ord] = ( right[0] + left[-1] ) / 2 # Otherwise use the midpoint of the left and right edges

    order_vals = flat_slice[order_zeros]

    return order_zeros, order_vals

## Function: Use flat slices to find starting values for the trace ##
def Find_Orders( super_flat, order_start, Conf ):
    # Uses flat slices at edge and in middle and uses that to refine initial points

    mid_point = ( ( super_flat.shape[1] + order_start ) // 2 ) + 100 # Integer division if it is odd, add 100 for uh some reason.

    start_zeros, start_vals = Start_Trace( super_flat[:,order_start], Conf.start_grad_perc ) # Get peaks for edge of flat
    mid_zeros, mid_vals     = Start_Trace( super_flat[:,mid_point], Conf.mid_grad_perc )   # Get peaks for middle of flat

    # Plot the start and mid trace
    plt.clf()
    with PdfPages( Conf.rdir + 'trace/prelim_trace_start_mid.pdf' ) as pdf:
        for x_range in [ ( 0, super_flat.shape[0] ), ( 0, 900 ), ( 800, super_flat.shape[0] ) ]:
            plt.plot( super_flat[:,mid_point], '#dfa5e5', label = 'Mid Order' ); plt.plot( mid_zeros, mid_vals, '+', c = '#bf3465' )
            plt.plot( super_flat[:,order_start], '#21bcff', label = 'Order Edge' ); plt.plot( start_zeros, start_vals, '+', c = '#1c6ccc' )
            plt.xlim( x_range[0] - 10, x_range[1] ); plt.ylim( 0, np.nanmax( super_flat[x_range[0]:x_range[1],mid_point] ) + 0.015 )
            plt.xlabel( 'Pixel across orders' ); plt.ylabel( 'Flat Field Value' )
            plt.legend(); pdf.savefig(); plt.close()
    plt.clf()

    # By hand remove extra orders that are present at the midpoint (just for here it is always 2, but that could be a problem for not our setup)
    mid_zeros = mid_zeros[2:]
    mid_vals  = mid_vals[2:]

    # Calculate slopes between the two to refine and smooth out trace starting points across orders (potential for outliers with spiky trace peaks)
    slopes    = []
    x_diff    = super_flat.shape[1] + order_start - mid_point
    fit_range = range( 3, start_zeros.size - 2 ) # Fit with the orders found from the 4th to the number of edge orders minus 2 (used to be index = 5 to 50)

    for i_ord in fit_range:
        y_diff = float( start_zeros[i_ord] - mid_zeros[i_ord] )
        slopes.append( y_diff / x_diff )
    slope_fit = np.polyfit( fit_range, slopes, 2 ) # Now do a 2nd order polynomial fit of the slopes!

    # Plot the slope thing
    plt.clf()
    plt.plot( fit_range, slopes, 'o', c = '#dfa5e5' )
    plt.plot( np.linspace( fit_range[0], fit_range[-1], 1000 ), np.polyval( slope_fit, np.linspace( fit_range[0], fit_range[-1], 1000 ) ), '#bf3465' )
    plt.xlabel( 'Order Number' ); plt.ylabel( 'Edge to Mid Slope' )
    plt.savefig( Conf.rdir + 'trace/prelim_trace_slopes.pdf' ); plt.clf()

    # Apply the fit 2nd order polynomial to get the final order locations at the edge!
    final_zeros = np.round( mid_zeros + np.polyval( slope_fit, range( len( mid_zeros ) ) ) * x_diff ).astype( int )
    final_vals  = super_flat[final_zeros, order_start]

    return final_zeros, final_vals

######## INSERT THE OTHER FUNCTIONS ######

## Function: Use order starting points to calculate full trace from the flat field ##
def Full_Trace( super_flat, order_zeros, order_start ):

    num_ord = order_zeros.size
    trace   = np.zeros( ( num_ord, super_flat.shape[1] + order_start ), dtype = int )

    for pix in range( 1, super_flat.shape[1] + order_start + 1 ):

        if pix > 1:
            prev = trace[:,-pix+1]
        else:
            prev = order_zeros

        flat_slice = super_flat[:,-pix+order_start]

        for i_ord in range( num_ord ):

            flat_cut_val = flat_slice[prev[i_ord]] * 0.7
            left_edge    = prev[i_ord] - 15 + np.where( flat_slice[prev[i_ord]-15:prev[i_ord]] <= flat_cut_val )[-1]
            right_edge   = prev[i_ord] + np.where( flat_slice[prev[i_ord]:prev[i_ord]+20] <= flat_cut_val )[-1]

            if len( left_edge ) == 0 or len( right_edge ) == 0:
                trace[i_ord,-pix] = int( prev[i_ord] )
            else:
                trace[i_ord,-pix] = int( 0.5 * ( left_edge[-1] + right_edge[0] ) )

    return trace

## Function: Fit the full trace and correct outlier bad orders ##
def Fit_Trace( trace, super_flat, Conf ):
    # Fit the trace with a 2nd order polynomial (the cubic term in the 3rd order fit was basically 0 for all orders)
    # Also go through and fit the linear and quadratic terms as function of order number -- fix bad orders!

    fit_trace  = np.zeros( ( trace.shape[0], 2048 ) )
    trace_pars = np.zeros( ( trace.shape[0], 3 ) )

    for i_ord in range( trace.shape[0] ): # Do initial fit for the trace along each order
        # trace_pars[i_ord] = np.polyfit( np.arange( trace.shape[1] ), trace[i_ord], 2 )
        trace_pars[i_ord] = np.polyfit( np.arange( 750, trace[i_ord].size ), trace[i_ord,750:], 2 )

    plt.clf()
    with PdfPages( Conf.rdir + 'trace/initial_fit_full_trace.pdf' ) as pdf:
        for y_range in [ ( 2048, 0 ), ( 2048, 1000 ), ( 950, 0 ) ]:
            plt.imshow( np.log10( super_flat ), aspect = 'auto', cmap = 'gray' )
            for i_ord in range( fit_trace.shape[0] ):
                plt.plot( trace[i_ord], '.', c = '#dfa5e5', ms = 2 )
                plt.plot( np.arange( 750, trace[i_ord].size ), trace[i_ord,750:], '.', c = '#50b29e', ms = 2 )
                plt.plot( np.polyval( trace_pars[i_ord], np.arange( trace.shape[1] ) ), '#bf3465', lw = 1 )
            plt.xlim( 0, 2048 ); plt.ylim( y_range ); pdf.savefig(); plt.close()
    plt.clf()

    ## Title string values
    title_arr = [ '2nd order polynomial coefficient', '']
    bad_orders = []
    hyper_par_fit_pars = np.zeros( ( 3, 4 ) )
    with PdfPages( Conf.rdir + 'trace/full_trace_hyperpars.pdf' ) as pdf:
        for i_coeff in [ 0, 1, 2 ]: # Redetermine linear and quadratic terms in the fits (leave zero point alone)

            hyper_pars = np.polyfit( np.arange( trace_pars.shape[0] ), trace_pars[:,i_coeff], 2 )
            hyper_fit  = np.polyval( hyper_pars, np.arange( trace_pars.shape[0] ) )

            med_diff = np.median( np.abs( trace_pars[:,i_coeff] - hyper_fit ) )
            mask     = np.where( np.abs( trace_pars[:,i_coeff] - hyper_fit ) <= 5 * med_diff )[0] # Correct orders more than 5 "sigma" bad
            bad      = np.where( np.abs( trace_pars[:,i_coeff] - hyper_fit ) > 5 * med_diff )[0]

            print( i_coeff, bad )

            for bad_ord in bad:
                bad_orders.append( bad_ord )

            hyper_pars_2 = np.polyfit( mask, trace_pars[mask,i_coeff], 3 )
            hyper_fit_2  = np.polyval( hyper_pars_2, np.arange( trace_pars.shape[0] ) )

            hyper_par_fit_pars[i_coeff] = hyper_pars_2

            # Plot the hyper parameter stuff!
            plt.plot( trace_pars[:,i_coeff], 'o', c = '#dfa5e5', label = 'Order Parameter Values' )
            plt.plot( mask, trace_pars[mask,i_coeff], '*', c = '#874310', label = 'Good Orders' )

            plot_x = np.linspace( 0, trace_pars.shape[0], 200 )
            plt.plot( plot_x, np.polyval( hyper_pars, plot_x ), '#50b29e', label = 'Initial Hyper Par Fit' )
            plt.plot( plot_x, np.polyval( hyper_pars_2, plot_x ), '#bf3465', label = 'Final Hyper Par Fit' )
            plt.xlabel( 'Order Number' ); plt.ylabel( 'Polynomial coefficient value' )
            plt.title( str( 2 - i_coeff ) + ' order polynomial coefficient' )
            plt.legend(); pdf.savefig(); plt.close()

            # trace_pars[:,i_coeff] = hyper_fit_2

            if i_coeff == 2:
                unique_bad_orders = np.unique( bad_orders )
                trace_pars[unique_bad_orders,i_coeff] = hyper_fit_2[unique_bad_orders]
            else:
                trace_pars[:,i_coeff] = hyper_fit_2
    plt.clf()

    if Conf.extend_to_58:
        if trace.shape[0] < 58:
            fit_trace = np.zeros( ( 58, 2048 ) )
            for i_ord in range( 58 ):
                if i_ord < trace.shape[0]:
                    fit_trace[i_ord] = np.polyval( trace_pars[i_ord], np.arange( 2048 ) )
                else:
                    c0 = np.polyval( hyper_par_fit_pars[0], i_ord )
                    c1 = np.polyval( hyper_par_fit_pars[1], i_ord )
                    c2 = np.polyval( hyper_par_fit_pars[2], i_ord )
                    fit_trace[i_ord] = np.polyval( np.array( [ c0, c1, c2 ] ), np.arange( 2048 ) )
    else:
        for i_ord in range( trace.shape[0] ): # Calculate fitted trace from corrected/final polynomial fits
            fit_trace[i_ord] = np.polyval( trace_pars[i_ord], np.arange( 2048 ) )

    return fit_trace

## Function: Call above functions to get initial trace, calculate full trace, and return fitted trace ##
def Get_Trace( super_flat, Conf ):

    if Conf.doTrace: # If we need to calculate the trace

        if Conf.verbose: print( 'Performing preliminary trace' )
        order_zeros, order_vals = Find_Orders( super_flat, Conf.order_start, Conf ) # Find initial values for trace

        # Plot the preliminary trace
        plt.clf()
        with PdfPages( Conf.rdir + 'trace/prelim_trace.pdf' ) as pdf:
            for x_range in [ ( 0, super_flat.shape[0] ), ( 0, 900 ), ( 800, super_flat.shape[0] ) ]:
                plt.plot( super_flat[:,Conf.order_start], '#d9d9d9' ); plt.plot( order_zeros, order_vals, '+', c = '#bf3465' )
                plt.xlim( x_range[0] - 10, x_range[1] ); plt.ylim( 0, np.nanmax( super_flat[x_range[0]:x_range[1],Conf.order_start] ) + 0.015 );
                plt.xlabel( 'Pixel across orders' ); plt.ylabel( 'Flat Field Value' ); pdf.savefig(); plt.close()
        plt.clf()

        # Write out the preliminary trace
        pickle.dump( { 'zeros': order_zeros, 'vals': order_vals }, open( Conf.rdir + 'trace/prelim_trace.pkl', 'wb' ) )

        # Get rid of the first order if it isn't a full order/would spill over the top of the image (15 pixels roughly is half an order)
        if order_zeros[0] < 15:
            order_zeros = order_zeros[1:]
            order_vals  = order_vals[1:]

        # Get the full trace and fit it with polynomials per order
        full_trace = Full_Trace( super_flat, order_zeros, Conf.order_start ) # Get full trace
        fit_trace  = Fit_Trace( full_trace, super_flat, Conf ) # Fit the full trace

        # # Make sure the top order is a full order and doesn't spill over top of image
        # if fit_trace[0,-1] <= 10.0: # 10 pixels is rough width of an order, be a bit conservative
        #     full_trace = full_trace[1:]
        #     fit_trace = fit_trace[1:]

        if Conf.verbose: print( 'Saving full and fitted trace to file' )
        pickle.dump( full_trace, open( Conf.rdir + 'trace/full_trace.pkl', 'wb' ) )
        pickle.dump( fit_trace, open( Conf.rdir + 'trace/fit_trace.pkl', 'wb' ) )

        pickle.dump( full_trace, open( Conf.rdir + 'trace/full_trace_py2.pkl', 'wb' ), protocol = 2 )
        pickle.dump( fit_trace, open( Conf.rdir + 'trace/fit_trace_py2.pkl', 'wb' ), protocol = 2 )

        plt.clf()
        with PdfPages( Conf.rdir + 'trace/final_trace.pdf' ) as pdf:
            for y_range in [ ( 2048, 0 ), ( 2048, 900 ), ( 950, 0 ) ]:
                plt.imshow( np.log10( super_flat ), aspect = 'auto', cmap = 'gray' )
                for i_ord in range( full_trace.shape[0] ):
                    plt.plot( full_trace[i_ord], '.', c = '#dfa5e5', ms = 2 )
                for i_ord in range( fit_trace.shape[0] ):
                    plt.plot( fit_trace[i_ord], '#bf3465', lw = 1 )
                plt.xlim( 0, 2048 ); plt.ylim( y_range ); plt.title( full_trace.shape ); pdf.savefig(); plt.close()
        plt.clf()

    else: # If trace has already been calculated

        if Conf.verbose: print( 'Reading in premade Trace and plotting on Flat:' )
        full_trace = pickle.load( open( Conf.rdir + 'trace/full_trace.pkl', 'rb' ) )
        fit_trace  = pickle.load( open( Conf.rdir + 'trace/fit_trace.pkl', 'rb' ) )

    return full_trace, fit_trace

##### Section: Extraction! #####

## Function: Calculate difference between data and model for mpyfit ##
def Least( p, args ):

    X, vals, err, func = args # Unpack arguments

    if err is not None: # If there is an error array provided #### I will change this to just pass an array of ones if there isn't error to put in
        dif = ( vals - func( X, p ) ) / err
    else:
        dif = vals - func( X, p )

    return dif.ravel() # Use ravel() to turn multidimensional arrays to 1D

## Function: 2D model of an order (each pixel is a gaussian with variation along dispersion direction) ##
def OrderModel( X, p, return_full = False ):

    x, y = X

    means  = p[2] * x ** 2.0 + p[1] * x + p[0] # Trace of the order -- parabola
    peaks  = p[5] * x ** 2.0 + p[4] * x + p[3] # Peak shape curve -- parabola
    sigmas = p[9] * x ** 3.0 + p[8] * x ** 2.0 + p[7] * x + p[6] # Sigma curve -- cubic

    # Full model
    model  = peaks * np.exp( - ( y - means ) ** 2.0 / ( 2.0 * sigmas ** 2.0 ) ) + p[10]

    if return_full == False: return model    # If we just want the model
    else: return model, means, peaks, sigmas # If we need all of the model constituents

## Function: Simple 1D Gaussian model ##
def GaussModel( x, p ):
    # Order of the parameters: amplitude, mean, sigma, background offset

    model = p[0] * np.exp( - ( x - p[1] ) ** 2.0 / ( 2.0 * p[2] ** 2.0 ) ) + p[3]

    return model
