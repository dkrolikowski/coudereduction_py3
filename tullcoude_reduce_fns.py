###################################################################################################
##### Functions for reducing Tull coude data #####
##### Daniel Krolikowski, Aaron Rizzuto #####
##### Optical echelle spectra #####
###################################################################################################

##### Imports #####

import glob, os, pickle, mpyfit, pdb
# import cosmics

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

    files   = glob.glob( '*.fits' ) # Get list of fits file names

    outfile = open( Conf.InfoFile, 'wb' ) # Open up the header info file to write to

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
        if 'airmass' in head.keys(): air = str( head['airmass'] )

        exp_time = str( head['exptime'] ) # Get the exposure time

        if 'gain3' in head.keys(): # Get the gain and read noise
            gain, rdn = str( head['gain3'] ), str( head['rdnoise3'] )
        else:
            gain, rdn = str( head['gain2'] ), str( head['rdnoise2'] )

        emflux = ''
        if 'emflux' in head.keys(): emflux = str( head['emflux'] )

        line = ','.join( [ files[f], head['object'], ra, dec, itype, exp_time, head['DATE-OBS'], head['UT'], air, gain, rdn, head['ZD'], emflux, head['order'] ] )
        outfile.write( line + '\n' )

        outfile.flush()

    outfile.close()

    return None

##### Section: Basic calibration functions (bias, flat, bad pixel mask) #####

## Function: Make the super bias frame ##
def Build_Bias( files, read_noise ):

    test_data = fits.open( files[0] )[0].data # Read in one of the frames to get shape for the bias cube
    bias_cube = np.zeros( ( len( files ), test_data.shape[0], test_data.shape[1] ) )

    for f in range( len( files ) ):
        bias_cube[f] = fits.open( files[f] )[0].data

    super_bias = {}

    super_bias['vals'] = np.nanmedian( bias_cube, axis = 0 )
    super_bias['errs'] = np.sqrt( super_bias['vals'] + read_noise ** 2.0 )

    return super_bias

## Function: Make the combined flat frame ##
def Build_Flat( files, read_noise, super_bias ):

    test_data = fits.open( files[0] )[0].data
    flat_cube = np.zeros( ( len( files ), test_data.shape[0], test_data.shape[1] ) )

    for f in range( len( files ) ):
        flat_cube[f] = fits.open( files[f] )[0].data

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

    cut_bias    = np.percentile( super_bias, Conf.BPMlimit )
    bad_pix_map = np.where( ( super_bias > cut_bias ) | ( super_flat <= 0.0001 ) )

    return bad_pix_map

## Function: Call sub-functions and return bias, flat, BPM ##
def Basic_Cals( bias_inds, flat_inds, file_info, Conf ):

    if Conf.doCals == True: # If we need to generate calibration files
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

    elif Conf.doCals == False: # If we're reading in already generated cal files
        if Conf.verbose: print( 'Reading in already generated Bias, Flat, and BPM files' )
        super_bias  = pickle.load( open( Conf.rdir + 'bias.pkl' ) )
        super_flat  = pickle.load( open( Conf.rdir + 'flat.pkl' ) )
        bad_pix_map = pickle.load( open( Conf.rdir + 'bpm.pkl' ) )

    plt.clf() # Plot the bias
    plt.imshow( np.log10( super_bias['vals'] ), cmap = 'gray', aspect = 'auto', interpolation = 'none', vmin = np.median( np.log10( super_bias['vals'] ) ),
                vmax = np.percentile( np.log10( super_bias['vals'] ), Conf.BPMlimit ) )
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
                vmax = np.percentile( np.log10( super_bias['vals'] ), Conf.BPMlimit ) )
    plt.plot( bad_pix_map[1], bad_pix_map[0], ',', c = '#dfa5e5', ms = 1 ) # Invert x,y for imshow
    plt.colorbar()
    plt.savefig( Conf.rdir + 'plots/bpm.pdf' )
    plt.clf()

    return super_bias, super_flat, bad_pix_map
