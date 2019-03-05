import numpy as np
import pickle
from astropy.io.votable import parse
import matplotlib.pyplot as plt
import pandas as pd

import healpy as hp
import pointing_groups as pg

import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import NoConvergence

def createObsTable(df):
    """Create the input to the MPC post request.
    
    Input-
        df : pandas dataframe
            The dataframe of observations for a particular request.
    
    Output-
        textarea : string
            The string that should be the 'textarea' value in the payload dictionary
    """
    
    textarea = ''
    
    for idx in range(len(df)):
        c = SkyCoord(df['ra'].iloc[idx]*u.degree, df['dec'].iloc[idx]*u.degree)
        ra_hms = c.ra.hms
        dec_dms = c.dec.dms
        date_obs = df['date_obs'].iloc[idx].decode()#[2:-1]
        time_obj = Time(date_obs, format='iso', scale='utc')
        if dec_dms.d != 0:
            name = ("     %07i   %s %s %s.%s %02i %02i %06.3f%+03i %02i %05.2f                     W84\n" %
                        (df['visit_id'].iloc[idx], date_obs[:4], date_obs[5:7], 
                         date_obs[8:10], str(time_obj.mjd)[6:11],
                         ra_hms.h, ra_hms.m, ra_hms.s,
                         dec_dms.d, np.abs(dec_dms.m), np.abs(dec_dms.s)))
        else:
            if copysign(1, dec_dms.d) == -1.0:
                dec_dms_d = '-00'
            else:
                dec_dms_d = '+00'
            name = ("     %07i   %s %s %s.%s %02i %02i %06.3f%s %02i %05.2f                     W84\n" %
                        (df['visit_id'].iloc[idx], date_obs[:4], date_obs[5:7],
                         date_obs[8:10], str(time_obj.mjd)[6:11],
                         ra_hms.h, ra_hms.m, ra_hms.s,
                         dec_dms_d, np.abs(dec_dms.m), np.abs(dec_dms.s)))
        textarea += name
        
    return textarea

def parseResults(result):
    
    results_cut = result.text.split('\n')[12:-49]
    
    visit_id = []
    name = []
    ra_hour = []
    ra_min = []
    ra_sec = []
    dec_deg = []
    dec_min = []
    dec_sec = []
    v_mag = []
    ra_motion = []
    dec_motion = []
    
    for line in results_cut:
        #print(line)
        visit_id.append(int(line[6:12]))
        name.append(line[12:36])
        ra_hour.append(int(line[38:40]))
        ra_min.append(int(line[41:43]))
        ra_sec.append(float(line[44:48]))
        dec_deg.append(int(line[49:52]))
        dec_min.append(int(line[53:55]))
        dec_sec.append(int(line[56:58]))
        try:
            v_mag.append(float(line[60:64]))
        except ValueError:
            v_mag.append(-99.0)
        ra_motion.append('%s%i' % (line[84], int(line[82:84])))
        dec_motion.append('%s%i' % (line[91], int(line[89:91])))
            
    results_df = pd.DataFrame(np.array([visit_id, name, ra_hour, ra_min, ra_sec, 
                                       dec_deg, dec_min, dec_sec, v_mag, 
                                       ra_motion, dec_motion]).T, 
                             columns=['visit_id', 'name', 'ra_hour', 'ra_min', 'ra_sec', 
                                       'dec_deg', 'dec_min', 'dec_sec', 'v_mag', 
                                       'ra_motion', 'dec_motion'])
    
    results_df['visit_id'] = pd.to_numeric(results_df['visit_id'])
    results_df['ra_hour'] = pd.to_numeric(results_df['ra_hour'])
    results_df['ra_min'] = pd.to_numeric(results_df['ra_min'])
    results_df['ra_sec'] = pd.to_numeric(results_df['ra_sec'])
    results_df['dec_deg'] = pd.to_numeric(results_df['dec_deg'])
    results_df['dec_min'] = pd.to_numeric(results_df['dec_min'])
    results_df['dec_sec'] = pd.to_numeric(results_df['dec_sec'])
    results_df['v_mag'] = pd.to_numeric(results_df['v_mag'])
    results_df['ra_motion'] = pd.to_numeric(results_df['ra_motion'])
    results_df['dec_motion'] = pd.to_numeric(results_df['dec_motion'])
    
    return results_df

def runMPCRequests(dataframe, field_label):
    """Inputs-
        dataframe: pandas dataframe, pointing group dataframe
        field_label: str, identifier for field
    
    Outputs-
        results_df: pandas dataframe with MPC results for field
    """
    
    url     = 'https://www.minorplanetcenter.net/cgi-bin/mpcheck.cgi'
    headers = {}

    payload = {'year':'2018',
               'month':'01',
               'day': '16.67',
               'which':'obs',
               'ra':'',
               'decl':'',
               'TextArea':'',
               'radius':'90',
               'limit':'30.0',
               'oc':'500',
               'sort':'d',
               'mot':'h',
               'tmot':'s',
               'pdes':'u',
               'needed':'f',
               'ps':'n',
               'type':'1'
              }
    
    results_df = None

    ta = createObsTable(dataframe)
    payload['TextArea'] = ta
    print(ta)
    res = requests.post(url, data=payload, headers=headers)

    if results_df is None:
        results_df = parseResults(res)
        results_df['field'] = field_label
    else:
        label_results_df = parseResults(res)
        label_results_df['field'] = field_label
        results_df = results_df.append(label_results_df, ignore_index=True)
            
    return results_df

def matchPixels(objectDF,dataPath,verbose=False):
    """Match the MPC object RA and DEC to pixel coorinates in the DECam NEO Survey
    Inputs-
        objectDF: Dataframe from runMPCRequests.
            Can be cut on magnitude, etc. if desired
        dataPath: Path to the DECam NEO Survey warps.
        verbose: Verbosity flag for print output

    Outputs-
        objectDF: Updated Dataframe with object->pixel relationships.
    """
    
    nightVisits = np.unique(objectDF['visit_id'])
    for j,visit in enumerate(nightVisits):
        if verbose:
            print('Processing visit {}.'.format(visit))
        df_visit = objectDF.query('visit_id == %i' % visit)
        for idx, obj_row in df_visit.iterrows():
            ra = '%i:%i:%.1f' % (obj_row['ra_hour'], obj_row['ra_min'], obj_row['ra_sec'])
            dec = '%i:%i:%.1f' % (obj_row['dec_deg'], obj_row['dec_min'], obj_row['dec_sec'])
            c = SkyCoord(ra, dec, frame='icrs', unit=(u.hourangle, u.deg))

            for i in range(1, 63):
                if (i==2 or i==61):
                    continue
                if verbose:
                    print('Processing ccd {} of 62.'.format(i))
                try:
                    #print('%s/v%i/diffexp-%02i.fits' % (dataPath, visit, i))
                    fitsHeader = fits.getheader('%s/%02i/%i.fits' % (dataPath,i, visit),1)
                    w = WCS(fitsHeader)
                    x_pix, y_pix = c.to_pixel(w)
                    if (x_pix < 2010) and (x_pix > 0) and (y_pix < 4100) and (y_pix > 0):
                        if verbose:
                            print(obj_row['name'], ra, dec)
                            print(x_pix, y_pix)
                        objectDF['x_pixel'].iloc[idx] = x_pix
                        objectDF['y_pixel'].iloc[idx] = y_pix
                        objectDF['ccd'].iloc[idx] = i
                        break
                except NoConvergence:
                    continue
        print('Completed visit {} of {}'.format(j+1,len(nightVisits)))
    return(objectDF)
