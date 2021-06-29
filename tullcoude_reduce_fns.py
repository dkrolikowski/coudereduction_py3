###################################################################################################
##### Functions for reducing Tull coude data #####
##### Daniel Krolikowski, Aaron Rizzuto #####
##### Optical echelle spectra #####
###################################################################################################

##### Imports #####

import glob, os, pickle, mpyfit, pdb
import astroscrappy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.time import Time
from scipy import signal

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
        if len( files[f].split('.') ) > 2: itype = 'unknown' # Don't quite know what this is a catch for, some other fits file

        ra, dec = '', '' # Coordinates now!
        if 'RA'  in head.keys(): ra  = head['RA']
        if 'DEC' in head.keys(): dec = head['DEC']

        air = '' # Airmass now!
        if 'AIRMASS' in head.keys(): air = str( head['airmass'] )

        exp_time = str( head['exptime'] ) # Get the exposure time

        if 'GAIN3' in head.keys(): # Get the gain and read noise
            gain, rdn = str( head['gain3'] ), str( head['rdnoise3'] )
        else:
            gain, rdn = str( head['gain2'] ), str( head['rdnoise2'] )

        emflux = ''
        if 'EMFLUX' in head.keys(): emflux = str( head['emflux'] )

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
        pickle.dump( super_bias, open( Conf.rdir + 'bias.pkl', 'wb' ) )

        # Create super flat
        if Conf.verbose: print( 'Reading Flat Files' )
        flat_rdn   = file_info['rdn'].values[flat_inds] / file_info['gain'].values[flat_inds]
        super_flat = Build_Flat( file_info['File'].values[flat_inds], flat_rdn[0], super_bias )
        pickle.dump( super_flat, open( Conf.rdir + 'flat.pkl', 'wb' ) )

        # Create the bad pixel mask
        if Conf.verbose: print( 'Creating the bad pixel mask' )
        bad_pix_map = Make_BPM( super_bias['vals'], super_flat['vals'], Conf )
        pickle.dump( bad_pix_map, open( Conf.rdir + 'bpm.pkl', 'wb' ) )

    elif not Conf.doCals: # If we're reading in already generated cal files
        if Conf.verbose: print( 'Reading in already generated Bias, Flat, and BPM files' )
        super_bias  = pickle.load( open( Conf.rdir + 'bias.pkl' ) )
        super_flat  = pickle.load( open( Conf.rdir + 'flat.pkl' ) )
        bad_pix_map = pickle.load( open( Conf.rdir + 'bpm.pkl' ) )

    plt.clf() # Plot the bias
    plt.imshow( np.log10( super_bias['vals'] ), cmap = 'gray', aspect = 'auto', interpolation = 'none', vmin = np.median( np.log10( super_bias['vals'] ) ),
                vmax = np.percentile( np.log10( super_bias['vals'] ), Conf.bpm_limit ) )
    plt.colorbar()
    plt.title( str( np.nanmedian( super_bias['vals'] ) ) )
    plt.savefig( Conf.rdir + 'plots/bias.pdf' )
    plt.clf()

    plt.clf() # Plot the flat
    plt.imshow( np.log10( super_flat['vals'] ), cmap = 'gray', aspect = 'auto', interpolation = 'none' )
    plt.colorbar()
    plt.savefig( Conf.rdir + 'plots/flat.pdf' )
    plt.clf()

    plt.clf() # Plot the BPM
    plt.imshow( np.log10( super_bias['vals'] ), cmap = 'gray', aspect = 'auto', interpolation = 'none', vmin = np.median( np.log10( super_bias['vals'] ) ),
                vmax = np.percentile( np.log10( super_bias['vals'] ), Conf.bpm_limit ) )
    plt.plot( bad_pix_map[1], bad_pix_map[0], ',', c = '#dfa5e5', ms = 1 ) # Invert x,y for imshow
    plt.colorbar()
    plt.savefig( Conf.rdir + 'plots/bpm.pdf' )
    plt.clf()

    return super_bias, super_flat, bad_pix_map

##### Section: Generate data cubes -- 2D spectral cubes for arc and object exposures #####

## Function: Make the data cubes! With cosmic subtraction, error propagation ##
def Make_Cube( files, read_noise, gain, dark_curr, Conf, bias = None, flat = None, bpm = None, cos_sub = False ):

    for i_file, file in enumerate( files ): # Loop through all files
        frame = fits.open( file )[0].data

        if i_file == 0: # If it is the first file set the empty arrays for the cubes
            val_cube = np.zeros( ( len( files ), frame.shape[0], frame.shape[1] ) )
            snr_cube = np.zeros( ( len( files ), frame.shape[0], frame.shape[1] ) )

        val_cube[i_file] = frame - dark_curr[0] # Cube value, subtract dark if there is one (which I'm not doing)
        err_vals = np.sqrt( val_cube[i_file] + dark_curr[0] + read_noise[i_file] ** 2.0 ) # Noise

        # Perform cosmic subtraction, if specified. Using astroscrappy now instead of cosmics (which works in py3 and is about 2x faster it seems)
        if cos_sub:
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
        obj_val_cube, obj_snr_cube = Make_Cube( file_info['File'].values[obj_inds], rdn_vals, gain_vals, dark_vals, Conf, bias = super_bias, flat = super_flat, bpm = bad_pix_map, cos_sub = Conf.cosmic_sub )

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
def Start_Trace( flatslice, percent ):

    fgrad    = np.gradient( flatslice )
    cutvalue = np.nanpercentile( abs( fgrad ), percent )

    orderzeros = []
    last       = 0

    # Find peaks based on the gradient of the flat slice
    for i in range( 6, flatslice.shape[0] ):
        if fgrad[i] > cutvalue or last == 0:
            if 100 > i - last > 20 or last == 0:
                orderzeros.append( i + 11 )
                last = i

    orderzeros = np.array( orderzeros )

    for i in range( len( orderzeros ) ): # Go through and recenter the peaks found
        o      = orderzeros[i]
        cutoff = flatslice[o] * ( 0.7 )
        left   = o - 15 + np.where( flatslice[o-15:o] <= cutoff )[-1]
        right  = o + np.where( flatslice[o:o+20] <= cutoff )[-1]
        if len( left ) == 0 or len( right ) == 0:
            orderzeros[i] = o
        else:
            orderzeros[i] = ( right[0] + left[-1] ) / 2

    ordervals = flatslice[orderzeros]

    return orderzeros, ordervals
