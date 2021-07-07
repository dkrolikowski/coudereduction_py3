import tullcoude_reduce_fns as Fns
import numpy as np
import pandas as pd
import os, pickle, pdb

##### Set the names of the directories you want to reduce! #####

nightarr = [ 20200903, 20201130 ]

if not isinstance( nightarr[0], str ):
    nightarr = [ str(night) for night in nightarr ]

#################################################################################################################

##### Now we'll loop through all the nights and do the reduction! #####

for night in nightarr:

    ##### Set the configurations for the reduction! #####

    class Configs():

        def __init__( self ):

            ## Set directories ##
            self.dir     = os.getenv("HOME") + '/Research/coude_data/' + night + '/'
            self.rdir    = self.dir + 'reduction/'
            self.codedir = os.getenv("HOME") + '/codes/coudereduction_py3/'

            ## Set which things to be done! ##
            self.doCals   = True    # Extract and reduce calibration files
            self.doCubes  = False    # Make the arc/object spectra cubes
            self.doTrace  = True    # Do the trace!
            self.doArcEx  = False    # Extract arc spectra -- simple extraction
            self.doObjEx  = False    # Extract object spectra -- full extraction
            self.doArcWav = False    # Determine arc spectra wavelength solutions
            self.doObjWav = False    # Apply wavelength solutions to object spectra

            ## Set other important parameters ##
            self.cosmic_sub  = False   # Create object spectral cube with cosmic ray subtraction
            self.ObjExType  = 'arc'  # Set the extraction method for objects: 'full' or 'arc'
            self.verbose    = True    # Basically just have as much printing of what's going on to the terminal
            self.WavPolyOrd = 2       # Polynomial order for the wavelength solution fit
            self.niter_cosmic_sub  = 2       # Set the number of iterations for the cosmic subtraction

            self.InfoFile   = 'headstrip.csv'   # Name for the header info file
            self.PrelimWav  = 'prelim_wsol_new.pkl' # Name for the preliminary wavelength solution (initial guess)

            self.dark_curr_val = 0.0    # Value of the dark current
            self.bpm_limit     = 99.95  # Percentile to mark above as a bad pixel
            self.MedCut     = 85.0   # Flux percentile to cut at when making final trace using object spectra
            self.order_start = -33

            ## Other thing to do ##
            self.doContFit  = True   # Continuum fit the object spectra

    Conf = Configs() # Set the configurations to a variable to pass to functions

    assert Conf.ObjExType in [ 'full', 'arc' ], 'Object extraction type must be either full or arc'

    ##### Some directory and file setups! #####

    ## Make sure that the reduction directory actually exist! That would be a problem
    if not os.path.exists( Conf.rdir ):
        os.mkdir( Conf.rdir )
    ## And now do the same for any necessary sub directories
    sub_dirs = [ 'cal_files/', 'spec_cubes/', 'trace/' ]
    for sub_dir in sub_dirs:
        if not os.path.exists( Conf.rdir + sub_dir ):
            os.mkdir( Conf.rdir + sub_dir )

    print( '\nYou are reducing directory', Conf.dir, 'Better be right!\n' )

    os.chdir( Conf.dir ) # Get into the night's directory!

    ## Create the header info file
    if True:
    # if not os.path.exists( Conf.dir + Conf.InfoFile ):
        Fns.Header_Info( Conf )

    file_info = pd.read_csv( Conf.InfoFile )

    ##### Get file indices from header file #####

    bias_inds = np.where( file_info.Type == 'zero' )[0] ## Bias indicies
    flat_inds = np.where( file_info.Type == 'flat' )[0] ## Flat indicies

    ## Arcs and Objs require some checking against frames that shouldn't be included... ensure arc is called an arc and that an object isn't a solar port or test!
    arc_hdrnames    = [ 'Thar', 'ThAr', 'THAR', 'A' ]
    notobj_hdrnames = [ 'solar', 'SolPort', 'solar port', 'Solar Port', 'test', 'SolarPort', 'Solport', 'solport', 'Sol Port', 'Solar Port Halpha' ]

    # Exclude arc frames with exposure times below a certain amount (where there just won't be enough flux for a good solution)
    min_arc_exp = 30.0

    arc_inds = np.where( np.logical_and( ( ( file_info.Type.values == 'comp' ) & ( file_info.ExpTime.values > min_arc_exp ) ), np.any( [ file_info.Object == hdrname for hdrname in arc_hdrnames ], axis = 0 ) ) )[0]
    obj_inds = np.where( np.logical_and( file_info.Type.values == 'object', np.all( [ file_info.Object != hdrname for hdrname in notobj_hdrnames ], axis = 0 ) ) )[0]

    ##### Get the calibration files -- bias, flat, and bad pixel map #####

    super_bias, super_flat, bad_pix_map = Fns.Basic_Cals( bias_inds, flat_inds, file_info, Conf )

    ##### Now get the image cubes -- objects and arcs #####

    dark_curr_arr = file_info['ExpTime'].values * Conf.dark_curr_val ## Make a dark current array

    ## Make the image cubes! Outputs images and SNR images
    arc_val_cube, arc_snr_cube, obj_val_cube, obj_snr_cube = Fns.Return_Cubes( arc_inds, obj_inds, file_info, dark_curr_arr, super_bias, super_flat, bad_pix_map, Conf )

    ##### Now do the trace! This is basically all in the functions file #####

    full_trace, fit_trace = Fns.Get_Trace( super_flat['vals'], Conf )

    ## Funky thing to make sure the same orders (at least mostly) are extracted every time. Might wanna change this later but...
    ## It is basically the same for this set up always
    fit_trace = fit_trace[:58]

    # ##### Extraction time! For both arcs and objects #####
    #
    # ## Arc extraction!
    # if not Conf.doArcEx: # If the extraction is already done, read in the files
    #     wspec     = pickle.load( open( Conf.rdir + 'extracted_wspec.pkl', 'rb' ) )
    #     sig_wspec = pickle.load( open( Conf.rdir + 'extracted_sigwspec.pkl', 'rb' ) )
    #
    # else: # If we need to do the extraction, do the extraction!
    #     wspec, sig_wspec = Fns.Extractor( ArcCube, ArcSNR, FitTrace, Conf, quick = True, arc = True, nosub = True )
    #
    #     wspec     = wspec[:,::-1]     # Reverse orders so it goes from blue to red!
    #     sig_wspec = sig_wspec[:,::-1]
    #
    #     pickle.dump( wspec, open( Conf.rdir + 'extracted_wspec.pkl', 'wb' ) )
    #     pickle.dump( sig_wspec, open( Conf.rdir + 'extracted_sigwspec.pkl', 'wb' ) )
    #
    # ## Object extraction!
    #
    # obj_filename = ''
    # if Conf.ObjExType == 'arc': obj_filename += '_quick'
    # if Conf.CosmicSub: obj_filename += '_cossub'
    #
    # if not Conf.doObjEx: # If the extraction is already done, read in the files
    #
    #     spec      = pickle.load( open( Conf.rdir + 'extracted_spec' + obj_filename + '.pkl', 'rb' ) )
    #     sig_spec  = pickle.load( open( Conf.rdir + 'extracted_sigspec' + obj_filename + '.pkl', 'rb' ) )
    #
    # else: # If we need to do the extraction, do the extraction!
    #
    #     if Conf.ObjExType == 'full':
    #         spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, Conf, quick = False, arc = False, nosub = False )
    #     elif Conf.ObjExType == 'arc':
    #         spec, sig_spec   = Fns.Extractor( ObjCube, ObjSNR, FitTrace, Conf, quick = True, arc = False, nosub = True )
    #
    #     spec     = spec[:,::-1]     # Reverse orders so it goes from blue to red!
    #     sig_spec = sig_spec[:,::-1]
    #
    #     pickle.dump( spec, open( Conf.rdir + 'extracted_spec' + obj_filename + '.pkl', 'wb'  ) )
    #     pickle.dump( sig_spec, open( Conf.rdir + 'extracted_sigspec' + obj_filename + '.pkl', 'wb' ) )
    #
    # ##### Wavelength calibration now! #####
    #
    # arcwavsol = Fns.Get_WavSol( wspec, sig_wspec, Conf ) ## Arc wavelength solution
    #
    # objwavsol = Fns.Interpolate_Obj_WavSol( arcwavsol, FileInfo, ArcInds, ObjInds, Conf ) ## Object wavelength solution
    #
    # ##### Additional things to do! #####
    #
    # ## Continuum normalization
    #
    # if Conf.doContFit:
    #     print( 'Fitting the continuum!' )
    #
    #     cont, spec_cf, sigspec_cf = Fns.doContinuumFit( spec, sig_spec, Conf, obj_filename )
