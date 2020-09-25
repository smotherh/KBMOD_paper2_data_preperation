import numpy as np
import mpmath
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import os
import healpy as hp
import requests

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io.votable import parse
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import NoConvergence
from astropy.table import Table, vstack
from astroquery.imcce import Skybot
from astroquery.jplhorizons import Horizons
from skimage import measure

pd.options.mode.chained_assignment = None  # default='warn'

class FindObjects():
    def __init__(self):
        self.testVisit = None
        self.ccdNum = None
        self.dataPath = None
        self.fileType = 'DeepDiff'
        
    def matchCcds(self, ccd):
        """This is a wrapper for matchSingleVisit.
        It allows it to easily run in parallel with a simple
        pool.map() call.
        """
        #visitDF = cutDF.query('visit_id == %i' % visit)
        #visitDF = visitDF.reset_index(drop=True)
        visitLocal=self.testVisit
        visitDF = self.matchSingleVisitCcd(self.cutDF,visitLocal,ccd,self.dataPath)
        #print('Processed ccd {}'.format(ccd))
        return(visitDF)

    def matchVisits(self, visit):
        """This is a wrapper for matchSingleVisit.
        It allows it to easily run in parallel with a simple
        pool.map() call.
        """
        #visitDF = cutDF.query('visit_id == %i' % visit)
        #visitDF = visitDF.reset_index(drop=True)
        ccd=self.ccdNum
        visitDF = self.matchSingleVisitCcd(self.cutDF,visit,ccd,self.dataPath)
        #print('Processed visit {}'.format(visit))
        return(visitDF)

    def matchSingleVisitCcd(self, visitDF, visit, ccd, dataPath, uniqueWCS=True, verbose=False):
        """Match the MPC object RA and DEC to pixel coorinates in the DECam NEO Survey
        Inputs-
            visitDF : Dataframe from runMPCRequests.
                It should only contain values from a single visit, although it may
                have multiple objects in that visit.
                Can be cut on magnitude, etc. if desired
            dataPath : Path to the DECam NEO Survey warps.
            verbose : Verbosity flag for print output

        Outputs-
            visitDF : Updated Dataframe with object->pixel relationships.
        """    

        i=ccd
        if uniqueWCS:
            fits_visit = visit
        else:
            fits_visit = visitDF['visit'][0]
        if self.fileType == 'DeepDiff':
            fitsPath = '%s/warps/%02i/%s.fits' % (dataPath,i,fits_visit)
        elif self.fileType == 'Calexp':
            fitsPath = '{path}/processed_data/rerun/rerun_processed_data/{visit_id:07d}/calexp/calexp-{visit_id:07d}_{ccd_num:02d}.fits'.format(
                path=dataPath,visit_id=fits_visit,ccd_num=i)
        else:
            ValueError('Please select DeepDiff or Calexp fileType')
        if verbose:
            print('Processing visit {}.'.format(visit))
        # Iterate over all moving objects (rows) in the visitDF
        obj_row = visitDF[visitDF['visit']==visit]
        # Set ra and dec and use them to generate a SkyCoord object
        ra = obj_row['RA']
        dec = obj_row['DEC']
        c = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg))
        # Iterate over CCDs
        # CCD 2 and 61 are broken on DECam and should be skipped
        if (i==2 or i==61):
            return(obj_row)
        if verbose:
            print('Processing ccd {} of 62.'.format(i))
        # Calculate the pixel values for the objects in the visit
        try:
            # Load only the fits header, changing the path for varying CCDs
            fitsHeader = fits.getheader(fitsPath,1)
            # Load the world coordinate system and find the pixel values
            w = WCS(fitsHeader)

            x_pix, y_pix = c.to_pixel(w)
            # If the returned pixel values are on the given CCD, save the object
            if (x_pix < 2010) and (x_pix > 0) and (y_pix < 4100) and (y_pix > 0):
                if verbose:
                    print(obj_row['name'], ra, dec)
                    print(x_pix, y_pix)
                obj_row['x_pixel'] = x_pix
                obj_row['y_pixel'] = y_pix
                obj_row['ccd'] = i
        except:
            return(obj_row)
        return(obj_row)
    
    def getKBOList(self, pgNum):
        with open('PickledPointings.pkl', 'rb') as f:
            PointingGroups = pickle.load(f)
        PointingGroups = PointingGroups[pgNum]
        PointingGroups.drop_duplicates('visit_id',inplace=True)
        df = PointingGroups
        allTimes = []
        for i in range(len(PointingGroups)):
            date_obs = df['date_obs'].iloc[i].decode()#[2:-1]
            time_obj = Time(date_obs, format='iso', scale='utc')
            allTimes.append(time_obj.jd)
        allTimes = np.array(allTimes)
        #timesMask = np.logical_and(allTimes>2458576.714,allTimes<2458577.70862527)
        times = allTimes

        ra = PointingGroups['ra'].values
        dec = PointingGroups['dec'].values
        i=0
        field = (ra[i],dec[i])
        epoch = Time(times[i], format='jd')
        Results = Skybot.cone_search(field, 90*u.arcmin, epoch, location='W84')

        RA_rate = Results['RA_rate']
        DEC_rate = Results['DEC_rate']
        pixel = u.arcsec/.26
        RA_rate = RA_rate.to(pixel/u.day)
        DEC_rate = DEC_rate.to(pixel/u.day)

        totalRate = np.linalg.norm([RA_rate,DEC_rate],axis=0)
        Type = np.array([classtype[0:3] for classtype in np.array(Results['Type'])])
        KBOList = Results[Type=='KBO']#['Name','RA','DEC','V','DEC_rate','RA_rate']
        return(PointingGroups, KBOList)

def searchKnownObject(
    singleObject, imagePath,numCols=5, stampSize=[31,31], fileType='DeepDiff',
    useSeeing=False, doMask=True, doStaticMask=True):
    
    defaults = {
            'im_filepath':None, 'res_filepath':None, 'time_file':None,
            'v_arr':[92.,526.,256], 'ang_arr':[np.pi/15,np.pi/15,128], 
            'output_suffix':'search', 'mjd_lims':None, 'average_angle':None,
            'do_mask':True, 'mask_num_images':2, 'mask_threshold':120.,
            'lh_level':10., 'psf_val':1.4, 'num_obs':10, 'num_cores':30,
            'visit_in_filename':[0,6], 'file_format':'{0:06d}.fits',
            'sigmaG_lims':[25,75], 'chunk_size':500000, 'max_lh':1000.,
            'filter_type':'clipped_sigmaG', 'center_thresh':0.03,
            'peak_offset':[2.,2.], 'mom_lims':[35.5,35.5,2.0,0.3,0.3],
            'stamp_type':'sum', 'eps':0.03}
    
    mask_bits_dict = {'BAD': 0, 'CLIPPED': 9, 'CR': 3, 'DETECTED': 5, 'DETECTED_NEGATIVE': 6, 'EDGE': 4,
             'INEXACT_PSF': 10, 'INTRP': 2, 'NOT_DEBLENDED': 11, 'NO_DATA': 8, 'REJECTED': 12,
             'SAT': 1, 'SENSOR_EDGE': 13, 'SUSPECT': 7}
    
    flag_keys = ['BAD', 'CR', 'INTRP', 'NO_DATA', 'SENSOR_EDGE', 'SAT',
                 'SUSPECT', 'CLIPPED', 'REJECTED', 'DETECTED_NEGATIVE']
    
    static_mask_keys = ['DETECTED']
    
    master_mask_threshold = 10
    
    mask_bit = 0
    for key in flag_keys:
        mask_bit += 2**mask_bits_dict[key]
    
    filterSearch = Filter(defaults)

    totalLength = len(singleObject)
    # Find the number of subplots to make. Add one for the coadd.
    numPlots = len(singleObject)
    #numPlots=15
    # Compute number of rows for the plot
    numRows = numPlots // numCols
    # Add a row if numCols doesn't divide evenly into numPlots
    if (numPlots % numCols):
        numRows+=1
    numRows+=1
    # Generate the subplots, setting the size with figsize
    fig,ax = plt.subplots(nrows=numRows,ncols=numCols,
                          figsize=[3.4*numCols,4.0*numRows])
    try:
        objectMag = np.max(singleObject['V'])
    except:
        objectMag = -99.0
    # Find object velocity in pixels/day and the object angle in radians
    # total_motion is in arcsec/hr. DECam has .26arcsec/pixel ccd's. 24 hr/day.
    # Load initial and final object positions and calculate the trajectory angle
    findMotion = singleObject[singleObject['ccd']==singleObject['ccd'][0]]
    xi = np.array([findMotion['x_pixel'][0],
                   findMotion['y_pixel'][0]])
    xf = np.array([findMotion['x_pixel'][-1],
                   findMotion['y_pixel'][-1]])
    dx = xf-xi
    objectAngle = np.arctan2(dx[1],dx[0])
    dr = np.linalg.norm(dx)
    dt= findMotion['times'][-1] - findMotion['times'][0]
    objectVel = dr/dt
    x_position = findMotion['x_pixel']
    y_position = findMotion['y_pixel']
    times = findMotion['times']
    xVel = np.polyfit(times, x_position, 1)[0]
    yVel = np.polyfit(times, y_position, 1)[0]
    xResidual = np.sum(np.abs(np.poly1d(np.polyfit(times,x_position, 1))(times)-x_position))
    yResidual = np.sum(np.abs(np.poly1d(np.polyfit(times,y_position, 1))(times)-y_position))

    
    if objectAngle<0:
        objectAngle += 2*np.pi
    # Turn off all axes. They will be turned back on for proper plots.
    for row in ax[1:]:
        for column in row:
            column.axis('off')

    # Set the axis indexes. These are needed to plot the stamp in the correct subplot
    axi=1
    axj=0
    #print(totalLength)
    #mask = np.array(sorted(random.sample(range(1,totalLength),14)))
    #print(mask)
    maskedObject = singleObject#[mask]
    psiArray = []
    phiArray = []
    coaddData=[]
    all_hdul = []
    master_mask_initialized = False
    for i,row in enumerate(singleObject):
        # Get the Lori Allen visit id from the single object list
        visit_id = row['visit']
        ccd = row['ccd']
        diffPath = os.path.join(imagePath,'warps/{:02}/{}.fits'.format(ccd,visit_id))
        calexpPath = (
            '{path}/processed_data/rerun/rerun_processed_data/'
            +'{visit_id:07d}/calexp/calexp-{visit_id:07d}_{ccd_num:02d}'
            +'.fits').format(path=imagePath,visit_id=visit_id,ccd_num=ccd)
        if fileType=='DeepDiff':
            fitsPath = diffPath
        elif fileType=='Calexp':
            fitsPath = calexpPath
        else:
            ValueError('Please select Calexp or DeepDiff fileType.')
        # Open up the fits file of interest using the pre-defined filepath string
        hdul = fits.open(fitsPath)
        all_hdul.append(hdul)
        
        mask = hdul[2].data.astype(int)
        if not master_mask_initialized:
            master_mask = np.zeros(np.shape(mask))
            master_mask_initialized = True
        static_mask_bits = 0
        for key in static_mask_keys:
            static_mask_bits += 2**mask_bits_dict[key]
        master_mask[(mask & static_mask_bits) != 0] += 1
        
    for i,row in enumerate(singleObject):
        # Get the Lori Allen visit id from the single object list
        mask_counter_dict = {}
        visit_id = row['visit']
        ccd = row['ccd']
        # Get the x and y values from the first object in the cut list. Round to an integer.
        #objectLoc = np.round([row['x_pixel'],row['y_pixel']])
        # Ensure that the positions are on a line
        xLoc = xi[0] + xVel*(times[i]-times[0])
        yLoc = xi[1] + yVel*(times[i]-times[0])
        objectLoc = np.round([xLoc,yLoc])
        #calexpHeader = fits.getheader(calexpPath)
        #seeing = calexpHeader.get('DIMMSEE') #FWHM in arcsec
        hdul = all_hdul[i]
        seeing = 0.0
        if type(seeing)!=float:
            seeing=0.0
        # Generate the minimum and maximum pixel values for the stamps using stampSize
        xmin = int(objectLoc[0]-(stampSize[0]-1)/2+0.5)-1
        xmax = int(objectLoc[0]+(stampSize[0]-1)/2+0.5)
        ymin = int(objectLoc[1]-(stampSize[1]-1)/2+0.5)-1
        ymax = int(objectLoc[1]+(stampSize[1]-1)/2+0.5)
        im_dims = np.shape(hdul[1].data)
        # Plot the stamp
        stampData = hdul[1].data[ymin:ymax,xmin:xmax]
        
        maskData = hdul[2].data[ymin:ymax,xmin:xmax].astype(int)
        masterMaskStamp = master_mask[ymin:ymax,xmin:xmax]
        constantMask = maskData & mask_bit
        if doStaticMask:
            stampData[masterMaskStamp > master_mask_threshold] = 0
            counter = int(np.shape(np.where(masterMaskStamp>master_mask_threshold))[1])
            if counter != 0:
                mask_counter_dict['Static'] = counter
            
        if doMask:
            for key in flag_keys:
                counter = np.shape(np.where(maskData & 2**mask_bits_dict[key]))[1]
                if counter:
                    mask_counter_dict[key] = counter
            stampData[constantMask!=0] = 0
        variance = hdul[3].data[int(objectLoc[1]),int(objectLoc[0])]
        #print(np.isnan(stampData))
        stampData[np.isnan(stampData)] = 0.0
        coaddData.append(stampData)

        stampEdge = (stampSize[0]-1)/2
        size = stampSize[0]
        x = np.linspace(-stampEdge, stampEdge, size)
        y = np.linspace(-stampEdge, stampEdge, size)
        if useSeeing:
            if seeing!=0:
                sigma_x = seeing/(2.355*0.26)
                sigma_y = seeing/(2.355*0.26)
            else:
                sigma_x=1.4
                sigma_y=1.4
        else:
            sigma_x=1.4
            sigma_y=1.4

        x, y = np.meshgrid(x, y)
        gaussian_kernel = (1/(2*np.pi*sigma_x*sigma_y) 
            * np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2))))
        sum_pipi = np.sum(gaussian_kernel**2)
        noise_kernel = np.zeros(stampSize)
        mask_lims = 7
        x_mask = np.logical_or(x>mask_lims, x<-mask_lims)
        y_mask = np.logical_or(y>mask_lims, y<-mask_lims)
        mask = np.logical_or(x_mask,y_mask)
        noise_kernel[mask] = 1
        psi = np.sum(stampData*gaussian_kernel/variance)
        phi = np.sum(gaussian_kernel*gaussian_kernel/variance)
        psiArray.append(psi)
        phiArray.append(phi)
        SNR = psi/np.sqrt(phi)
        if (True):
            mask_row = ''
            for i, key in enumerate(mask_counter_dict):
                if i%2==0 and i>0:
                    mask_row += '\n'
                elif i>0:
                    mask_row += ' | '
                mask_row += '{}: {}'.format(key, mask_counter_dict[key])

            im = ax[axi,axj].imshow(stampData,cmap=plt.cm.bone)
            ax[axi,axj].set_title(
                'ccd={} | visit={}\nSNR={:.2f} | FWHM={:.2f}"\n{}'.format(
                    ccd,visit_id,SNR,seeing,mask_row), fontsize=14)
            ax[axi,axj].axis('on')
            # Compute the axis indexes for the next iteration
            if axj<numCols-1:
                axj+=1
            else:
                axj=0
                axi+=1
    psiArray = np.array(psiArray)
    phiArray = np.array(phiArray)
    goodIndex, newLh = filterSearch.apply_sigmaG(psiArray,phiArray)
    coaddStamp = np.median(coaddData,axis=0)
    momLims, peakArray = filterSearch.apply_stamp(coaddStamp)
    im = ax[0,0].imshow(coaddStamp,cmap=plt.cm.bone)
    SNR = np.sum(psiArray)/np.sqrt(np.sum(phiArray))
    flux = psiArray/phiArray
    x_values = np.linspace(1,len(psiArray),len(psiArray))
    _=ax[0,0].set_title('Coadd | SNR={:.4f}'.format(SNR), fontsize=14)
    ax[0,1] = plt.subplot2grid((numRows,numCols), (0,1),colspan=4,rowspan=1)
    ax[0,1].plot(x_values, flux, 'b')
    ax[0,1].plot(x_values[goodIndex], flux[goodIndex], 'r.', ms=15)
    ax[0,1].xaxis.set_ticks(x_values)
    ax[0,1].set_title(
        ('Filtered SNR = {:.4f}\n'
        +'Cental Moments: [{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}], '
        +'Peak: [{},{}]').format(newLh,*momLims,*peakArray), fontsize=14)
    ax[0,1].set_xlabel('Visit number', fontsize=14)
    ax[0,1].set_ylabel('Object flux', fontsize=14)
    targetLoc = [findMotion['x_pixel'][0], findMotion['y_pixel'][0]]
    objectData = [singleObject['targetname'][0],objectMag,[xVel,yVel],
                  objectVel,objectAngle,targetLoc]
    figTitle = ('{} image\nv_mag={}, velocity=[{:.2f},{:.2f}]={:.2f} px/day,'
                +'angle={:.2f}\npixel=[{:.2f},{:.2f}], residual=[{:.2f},{:.2f}]')
    fig.suptitle(figTitle.format(
        singleObject['targetname'][0],objectMag,xVel,yVel,objectVel,
        objectAngle,*targetLoc,xResidual,yResidual),fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return(coaddData,objectData,findMotion)

def makeStamps(singleObject,imagePath,imagePlane='science',numCols=5,
               stampSize=[31,31],fileType='DeepDiff'):
    """Generate postage stamps of an MPC object in the Lori Allen Dataset.
    
    INPUT-
        name: This is the name of the object for which to find the image.
            Names come from the query_MPC notebook that pulls data down from
            the Minor Planets Center.

        objectList: A pandas dataframe as generated by query_MPC.

        imagePath: The path to the stack of images from which to make stamps.

        numCols: The number of columns in the postage stamp subplot.

        imagePlane : From which plane of the fits file should the stamps be
            made? Acceptable options:
            'science' : The science image plane
            'mask' : The mask image plane
            'variance' : The varience image plane
    """
    totalLength = len(singleObject)
    # Set the plane number used for loading the data from a fits file
    if imagePlane == 'science':
        imagePlaneNum = 1
    elif imagePlane == 'mask':
        imagePlaneNum = 2
    elif imagePlane == 'variance':
        imagePlaneNum = 3

    # Find the number of subplots to make. Add one for the coadd.
    numPlots = len(singleObject)+1
    #numPlots=15
    # Compute number of rows for the plot
    numRows = numPlots // numCols
    # Add a row if numCols doesn't divide evenly into numPlots
    if (numPlots % numCols):
        numRows+=1
    # Add a row if numRows=1. Avoids an error caused by ax being 1D.
    if (numRows==1):
        numRows+=1
    # Generate the subplots, setting the size with figsize
    fig,ax = plt.subplots(nrows=numRows,ncols=numCols,
                          figsize=[3*numCols,3.75*numRows])
    try:
        objectMag = np.max(singleObject['V'])
    except:
        objectMag = -99.0
    # Find object velocity in pixels/day and the object angle in radians
    # total_motion is in arcsec/hr. DECam has .26arcsec/pixel ccd's. 24 hr/day.
    # Load initial and final object positions and calculate the trajectory angle
    findMotion = singleObject[singleObject['ccd']==singleObject['ccd'][0]]
    xi = np.array([findMotion['x_pixel'][0],
                   findMotion['y_pixel'][0]])
    xf = np.array([findMotion['x_pixel'][-1],
                   findMotion['y_pixel'][-1]])
    dx = xf-xi
    objectAngle = np.arctan2(dx[1],dx[0])
    dr = np.linalg.norm(dx)
    dt= findMotion['times'][-1] - findMotion['times'][0]
    objectVel = dr/dt
    xVel = dx[0]/dt
    yVel = dx[1]/dt
    
    if objectAngle<0:
        objectAngle += 2*np.pi
    # Turn off all axes. They will be turned back on for proper plots.
    for row in ax:
        for column in row:
            column.axis('off')

    # Set the axis indexes. These are needed to plot the stamp in the correct subplot
    axi=0
    axj=1
    #print(totalLength)
    #mask = np.array(sorted(random.sample(range(1,totalLength),14)))
    #print(mask)
    maskedObject = singleObject#[mask]
    for i,row in enumerate(singleObject):
        # Get the Lori Allen visit id from the single object list
        visit_id = row['visit']
        ccd = row['ccd']
        # Get the x and y values from the first object in the cut list. Round to an integer.
        objectLoc = np.round([row['x_pixel'],row['y_pixel']])
        if fileType=='DeepDiff':
            fitsPath = os.path.join(imagePath,'warps/{:02}/{}.fits'.format(ccd,visit_id))
        elif fileType=='Calexp':
            fitsPath = '{path}/processed_data/rerun/rerun_processed_data/{visit_id:07d}/calexp/calexp-{visit_id:07d}_{ccd_num:02d}.fits'.format(
                path=imagePath,visit_id=visit_id,ccd_num=ccd)
        else:
            ValueError('Please select Calexp or DeepDiff fileType.')
        # Open up the fits file of interest using the pre-defined filepath string
        hdul = fits.open(fitsPath)
        seeing = hdul[0].header.get('DIMMSEE') #FWHM in arcsec
        if type(seeing)!=float:
            seeing=0.0
        # Generate the minimum and maximum pixel values for the stamps using stampSize
        xmin = int(objectLoc[0]-(stampSize[0]-1)/2+0.5)-1
        xmax = int(objectLoc[0]+(stampSize[0]-1)/2+0.5)
        ymin = int(objectLoc[1]-(stampSize[1]-1)/2+0.5)-1
        ymax = int(objectLoc[1]+(stampSize[1]-1)/2+0.5)

        im_dims = np.shape(hdul[imagePlaneNum].data)
        # Plot the stamp
        stampData = hdul[imagePlaneNum].data[ymin:ymax,xmin:xmax]
        #print(np.isnan(stampData))
        stampData[np.isnan(stampData)] = 0.0
        if i==0:
            coaddData=stampData
        else:
            coaddData+=stampData
        stampEdge = (stampSize[0]-1)/2
        size = stampSize[0]
        x = np.linspace(-stampEdge, stampEdge, size)
        y = np.linspace(-stampEdge, stampEdge, size)
        if seeing!=0:
            sigma_x = seeing/(2.355*0.26)
            sigma_y = seeing/(2.355*0.26)
        else:
            sigma_x=1.4
            sigma_y=1.4

        x, y = np.meshgrid(x, y)
        gaussian_kernel = (1/(2*np.pi*sigma_x*sigma_y) 
            * np.exp(-(x**2/(2*sigma_x**2) + y**2/(2*sigma_y**2))))
        sum_pipi = np.sum(gaussian_kernel**2)
        noise_kernel = np.zeros(stampSize)
        mask_lims = 7
        x_mask = np.logical_or(x>mask_lims, x<-mask_lims)
        y_mask = np.logical_or(y>mask_lims, y<-mask_lims)
        mask = np.logical_or(x_mask,y_mask)
        noise_kernel[mask] = 1
        signal = np.sum(stampData*gaussian_kernel)
        noise = np.var(stampData*noise_kernel)
        SNR = signal/np.sqrt(noise*sum_pipi)
        #if (mask==i).any():
        if (True):

            im = ax[axi,axj].imshow(stampData,cmap=plt.cm.bone)
            ax[axi,axj].set_title(
                'ccd={} | visit={}\nSNR={:.2f} | FWHM={:.2f}"'.format(
                    ccd, visit_id, SNR, seeing))
            ax[axi,axj].axis('on')
            # Compute the axis indexes for the next iteration
            if axj<numCols-1:
                axj+=1
            else:
                axj=0
                axi+=1
    im = ax[0,0].imshow(coaddData,cmap=plt.cm.bone)
    signal = np.sum(coaddData*gaussian_kernel)
    noise = np.var(coaddData*noise_kernel)
    SNR = signal/np.sqrt(noise*sum_pipi)
    ax[0,0].axis('on')
    _=ax[0,0].set_title('Coadd | SNR={:.2f}'.format(SNR))
    targetLoc = [findMotion['x_pixel'][0], findMotion['y_pixel'][0]]
    objectData = [singleObject['targetname'][0], objectMag, [xVel,yVel],
                  objectVel, objectAngle, targetLoc]
    figTitle = ('{}: {} image\nv_mag={}, velocity=[{:.2f},{:.2f}]={:.2f}'
                + ' px/day, angle={:.2f}\npixel=[{},{}]')
    fig.suptitle(figTitle.format(
        singleObject['targetname'][0], imagePlane, objectMag, xVel, yVel,
        objectVel, objectAngle, *targetLoc), fontsize=16)
    return(coaddData,objectData)

def getPlots(PointingGroups, pgNum, KBOList, findObjects, fileType='DeepDiff',
             dataPath='/astro/store/epyc/users/smotherh/DECAM_Data_Reduction/pointing_groups_hyak/Pointing_Group_{0:03}'):
    pgObjectData = {}
    dupObjectNum = 0
    allFindMotion=[]
    for KBONum,KBO in enumerate(KBOList):
        objectName = KBO['Name']
        df = PointingGroups
        allTimes = []
        for i in range(len(PointingGroups)):
            date_obs = df['date_obs'].iloc[i].decode()#[2:-1]
            time_obj = Time(date_obs, format='iso', scale='utc')
            allTimes.append(time_obj.jd)
        allTimes = np.array(allTimes)
        times = allTimes
        obj = Horizons(id=objectName, location='W84', epochs=times) #ccd 43
        orbits = obj.ephemerides(quantities='1, 9')
        orbits['visit'] = [int(visit) for visit in df['visit_id']]
        orbits['x_pixel'] = -99
        orbits['y_pixel'] = -99
        orbits['ccd'] = -99
        orbits['times'] = times
        #visitMask = [np.logical_and(orbits['visit']>=845580,orbits['visit']<=845682)]
        #orbits = orbits[visitMask]
        findObjects.dataPath = (dataPath.format(pgNum))

        findObjects.cutDF = orbits
        findObjects.fileType = fileType
        nightVisits = np.array(orbits['visit'])

        findObjects.testVisit = nightVisits[20]
        with mp.Pool(20) as pool:
            results = pool.map(findObjects.matchCcds,range(1,63))

        for j in range(62):
            foo = results[j]
            onCcd = foo[foo['ccd']>0]
            if len(onCcd)>0:
                findObjects.ccdNum = onCcd['ccd'][0]
                print(objectName+' is on ccd '+str(findObjects.ccdNum))
        if findObjects.ccdNum is None:
            print(objectName+' is not on any ccd')
        else:
            with mp.Pool(20) as pool:
                results = pool.map(findObjects.matchVisits,nightVisits)

            allResults = vstack(results)
            allResults = allResults[allResults['ccd']>0]
            if (int(allResults[0]['visit'])==int(PointingGroups['visit_id'][0])):
                coaddData,objectData,findMotion = searchKnownObject(
                    allResults, findObjects.dataPath, stampSize=[21,21],
                    numCols=5, fileType=fileType)
                pgKey = 'pg{:03}_ccd{:02}'.format(pgNum, findObjects.ccdNum)
                allFindMotion.append(findMotion)
                if pgKey not in pgObjectData:
                    pgObjectData[pgKey] = objectData
                else:
                    dupObjectNum += 1
                    pgObjectData[pgKey+'_'+str(dupObjectNum)] = objectData
                plt.savefig('known_objects/pg{:03}_ccd{:02}_{}'.format(
                    pgNum,findObjects.ccdNum,objectName.replace(" ", "_")))
                plt.close()
            else:
                print('Object {} is not present in the first visit'.format(
                    objectName))
        findObjects.ccdNum = None
    return(pgObjectData, allFindMotion)

class Filter():
    def __init__(self, config):
        self.percentiles = config['sigmaG_lims']
        self.lc_filter_type = 'lh'
    def _find_sigmaG_coeff(self, percentiles):
        z1 = percentiles[0]/100
        z2 = percentiles[1]/100

        x1 = self._invert_Gaussian_CDF(z1)
        x2 = self._invert_Gaussian_CDF(z2)
        coeff = 1/(x2-x1)
        return(coeff)

    def _invert_Gaussian_CDF(self, z):
        if z < 0.5:
            sign = -1
        else:
            sign = 1
        x = sign*np.sqrt(2)*mpmath.erfinv(sign*(2*z-1))
        return(float(x))

    def _compute_lh(self, psi_values, phi_values):
        """
        This function computes the likelihood that there is a source along
        a given trajectory with the input Psi and Phi curves.
        INPUT-
            psi_values : numpy array
                The Psi values along a trajectory.
            phi_values : numpy array
                The Phi values along a trajectory.
        OUTPUT-
            lh : float
                The likelihood that there is a source along the given
                trajectory.
        """
        if (psi_values==0).all():
            lh = 0
        else:
            lh = np.sum(psi_values)/np.sqrt(np.sum(phi_values))
        return(lh)

    def apply_stamp(self, stamp):
        center_thresh = 0.03
        x_peak_offset, y_peak_offset = [2,2]
        s = stamp - np.min(stamp)
        s /= np.sum(s)
        s[np.isnan(s)] = 0
        s = np.array(s, dtype=np.dtype('float64')).reshape(21,21)
        mom = measure.moments_central(s, center=(10,10))
        mom_list = [mom[2, 0], mom[0, 2], mom[1, 1], mom[1, 0], mom[0, 1]]
        peak_1, peak_2 = np.where(s == np.max(s))
        peak_1 = np.max(np.abs(peak_1-10))
        peak_2 = np.max(np.abs(peak_2-10))
        return(mom_list,[peak_1,peak_2])

    def apply_sigmaG(
        self, psi_curve, phi_curve, n_sigma=2):
        """
        This function applies a clipped median filter to a set of likelihood
        values. Points are eliminated if they are more than n_sigma*sigmaG away
        from the median.
        INPUT-
            psi_curve : numpy array
                A single Psi curve, likely a single row of a larger matrix of
                psi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['psi_curves'].
            phi_curve : numpy array
                A single Phi curve, likely a single row of a larger matrix of
                phi curves, such as those that are loaded in from
                Interface.load_results() and stored in keep['phi_curves'].
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            n_sigma : integer
                The number of standard deviations away from the median that
                the largest likelihood values (N=num_clipped) must be in order
                to be eliminated.
        OUTPUT-
            index : integer
                The row value of the larger Psi and Phi matrixes from which
                psi_values and phi_values are extracted. Used to keep track
                of processing while running multiprocessing.
            good_index : numpy array
                The indices that pass the filtering for a given set of curves.
            new_lh : float
                The new maximum likelihood of the set of curves, after
                max_lh_index has been applied.
        """
        self.coeff = self._find_sigmaG_coeff(self.percentiles)
        masked_phi = np.copy(phi_curve)
        masked_phi[masked_phi==0] = 1e9
        if self.lc_filter_type=='lh':
            lh = psi_curve/np.sqrt(masked_phi)
        elif self.lc_filter_type=='flux':
            lh = psi_curve/masked_phi
        else:
            print('Invalid filter type, defaulting to likelihood', flush=True)
            lh = psi_curve/np.sqrt(masked_phi)
        lower_per, median, upper_per = np.percentile(
            lh, [self.percentiles[0], 50, self.percentiles[1]])
        sigmaG = self.coeff*(upper_per-lower_per)
        nSigmaG = n_sigma*sigmaG
        good_index = np.where(np.logical_and(
            lh > median-nSigmaG, lh < median+nSigmaG))[0]
        if len(good_index)==0:
            new_lh = 0
            good_index=[-1]
        else:
            new_lh = self._compute_lh(
                psi_curve[good_index], phi_curve[good_index])
        return(good_index,new_lh)


