import numpy as np
import pickle
from astropy.io.votable import parse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import pandas as pd

def load_raw_data(filepath='VOTMetadata/'):
    # This notebook is for managing the metadata for the Lori Allen Data Set

    # Initialize variable
    neo_metadata_df = None

    # Set the range of nights to ingest into the data frame
    for night_num in range(1,32):

        # Create the data frame
        metadata_df = pd.DataFrame()

        # Create a string for the give night's file name
        night_str = filepath+'night_%i.vot' % night_num

        # Ingest data into a table
        votable = parse(night_str)
        table = votable.get_first_table()

        # Turn the table into a data array
        data = table.array

        # Load file names for future use
        file_names = data['archive_file'].data

        # visit_id's need additional parsing so take them out now
        visit_id = data['dtacqnam'].data

        # Create fields in the data frame with given values
        metadata_df['visit_id'] = np.array([x[-14:-8] for x in visit_id])
        metadata_df['date_obs'] = data['date_obs'].data
        metadata_df['ra'] = data['ra'].data
        metadata_df['dec'] = data['dec'].data
        metadata_df['product'] = data['prodtype'].data
        metadata_df['filename'] = file_names
        metadata_df['survey_night'] = night_num

        # Now add the data frame for a given night to the total data frame
        if neo_metadata_df is None:
            neo_metadata_df = metadata_df
        else:
            neo_metadata_df = pd.concat([neo_metadata_df, metadata_df])

    # Only keep image data
    neo_metadata_df = neo_metadata_df[neo_metadata_df['product'] == b'image']
    neo_metadata_df = neo_metadata_df.reset_index(drop=True)
    return(neo_metadata_df)

def sort_pointings(neo_metadata_df,Ra_Tol=1e-2,Dec_Tol=2e-2,Min_Num_Visits=3):

    # Make a copy that we can manipulate
    df_copy = neo_metadata_df
    # Drop a few unnecessary columns
    df_copy = df_copy.drop(['product'],axis=1)

    # Set a maxinimum number of interations just in case
    max_iter = neo_metadata_df.axes[0][-1]
    i = 0

    # Initialize an array that will hold the dataframes for the sorted fields
    Pointing_Groups = []

    # This while loop interates over a given field, groups the data, and adds it to Pointing_Groups
    while (not df_copy.empty) and (i<max_iter):
        # Save the values for the current field
        current_ra = df_copy['ra'][0]
        current_dec = df_copy['dec'][0]
        current_id = df_copy['visit_id'][0]

        # Find all entries that are a part of this field.
        # Note that we have to use np.close because even within a field, RA and DEC vary somewhat
        ra_mask = np.isclose(current_ra,df_copy['ra'],atol=Ra_Tol)
        dec_mask = np.isclose(current_dec,df_copy['dec'],atol=Dec_Tol)
        duplicate_mask = (df_copy['visit_id'] == current_id)
        # Save the indicies for this field
        indexes = df_copy.index[ra_mask & dec_mask]

        # Create a new dataframe that only includes the entries of this field
        New_Field = df_copy.loc[indexes]
        New_Field.sort_values(['date_obs'])
        New_Field = New_Field.reset_index(drop=True)

        # Append the new field to the overall array
        Pointing_Groups.append(New_Field)
        df_copy = df_copy.drop(indexes)
        df_copy = df_copy.reset_index(drop=True)
        i+=1

    # Clean up the output
    for i,_ in enumerate(Pointing_Groups):
        # Convert the visit_id field to a numeric value
        Pointing_Groups[i]["visit_id"] = pd.to_numeric(Pointing_Groups[i]["visit_id"])
        # Find the index of duplicate visit_id values
        No_Duplicates = Pointing_Groups[i]["visit_id"].drop_duplicates()
        # Drop the duplicates
        Pointing_Groups[i] = Pointing_Groups[i].loc[No_Duplicates.index]
        # Reset indicies
        Pointing_Groups[i] = Pointing_Groups[i].reset_index(drop=True)

    # Now sort the array based on the length of the series
    Pointing_Groups = sorted(Pointing_Groups,key=len,reverse=True)

    # Strip out all values that have fewer visits than Min_Num_Visits
    Temp_Fields = []
    for field in Pointing_Groups:
        if len(field) > Min_Num_Visits:
            Temp_Fields.append(field)

    Pointing_Groups = Temp_Fields
    return(Pointing_Groups)

def link_instcal_files(Pointing_Groups,NEO_src,catalog_src,dest,script_name='link_files.sh'):
    """
    This function generates a bash script that creates directories and
    symlinks the correct file names for all pointing_groups.

    DO NOT PUT A TRAILING / IN YOUR FILE PATHS.
    The script adds them in where necessary

    Input
    ---------

    Pointing_Groups:
        An array of pointing group pandas tables loaded in from PickledPointings.pkl

    NEO_src : string
        File path to the folder containing all of the raw instcal files for the
        Lori Allen dataset

    catalog_src : string
        File path to the folder containing all of the reference catalogs needed for
        processing the Lori Allen dataset

    dest : string
        File path destination. All of the folders will be populated into "dest"

    script_name : string
        Optional file name of the script

    Output
    ---------

    none, function will write to a file, named by script_name

    """

    with open(script_name,'w') as link_files:
        link_files.write('# This file links data in a given field\n')
        link_files.write('cd '+dest+'\n')

    for i,Pointing in enumerate(Pointing_Groups):
        if Pointing['stellar_density'][0]>10000.:
            continue

        ra = str(Pointing['ra'][0])
        dec = str(Pointing['dec'][0])
        dir_name = dest+'/Pointing_Group_'+str(i).zfill(3)

        with open(script_name,'a') as link_files:
            link_files.write('\n# Link files for field: '+dir_name+'\n')
            link_files.write('mkdir '+dir_name+'\n')
            link_files.write('mkdir '+dir_name+'/ingest/\n')
            link_files.write('mkdir '+dir_name+'/ingest/instcal\n')
            link_files.write('mkdir '+dir_name+'/ingest/dqmask\n')
            link_files.write('mkdir '+dir_name+'/ingest/wtmap\n')
            link_files.write('mkdir '+dir_name+'/processed_data\n')
            link_files.write('echo lsst.obs.decam.DecamMapper > '+dir_name+'/processed_data/_mapper\n')
            link_files.write('ln -s '+catalog_src+' '+dir_name+'/processed_data/ref_cats\n')

            link_files.write('# Link the images files\n')

        for index,image in Pointing.iterrows():
            if image['survey_night'] is not 3:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+str(image['survey_night']) +
                      '/night_'+str(image['survey_night'])+'/'+image['filename'].decode('UTF-8') +
                      ' '+dir_name+'/ingest/instcal/\n')
            else:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+str(image['survey_night']) +
                      '/night_'+str(image['survey_night'])+'/night_'+str(image['survey_night'])+'/'+image['filename'].decode('UTF-8') +
                      ' '+dir_name+'/ingest/instcal/\n')

        with open(script_name,'a') as link_files:
            link_files.write('# Link the dqmask files\n')

        for index,image in Pointing.iterrows():
            dqmask = image['filename'].decode('UTF-8')[0:20]+'d'+image['filename'].decode('UTF-8')[21:]
            night = str(image['survey_night'])
            if image['survey_night'] is not 3:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+ night +
                      '/night_'+night+'/'+dqmask +
                      ' '+dir_name+'/ingest/dqmask/\n')
            else:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+night +
                      '/night_'+night+'/night_'+night+'/'+ dqmask +
                      ' '+dir_name+'/ingest/dqmask/\n')

        with open(script_name,'a') as link_files:
            link_files.write('# Link the wtmask files\n')

        for index,image in Pointing.iterrows():
            wtmap = image['filename'].decode('UTF-8')[0:20]+'w'+image['filename'].decode('UTF-8')[21:]
            night = str(image['survey_night'])
            if image['survey_night'] is not 3:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+ night +
                      '/night_'+night+'/'+ wtmap +
                      ' '+dir_name+'/ingest/wtmap/\n')
            else:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+night +
                      '/night_'+night+'/night_'+night+'/'+ wtmap +
                      ' '+dir_name+'/ingest/wtmap/\n')

def process_visits(Pointing_Groups,pg_location,pg_lims,script_name='process_visits.sh',
                   num_cores=20,source_stack=False,setup_loc=''):
    
    with open(script_name,'w') as f:
        
        if source_stack:
            f.write('source '+setup_loc+'\n')
            f.write('setup lsst_distrib\n')
        
        f.write('cd '+pg_location+'\n')
        pg_ids = np.linspace(pg_lims[0],pg_lims[1],pg_lims[1]-pg_lims[0]+1,dtype=int)
        for i in pg_ids:
            if Pointing_Groups[i]['stellar_density'][0]>10000:
                continue
            min_visit = str(np.min(Pointing_Groups[i]['visit_id']))
            max_visit = str(np.max(Pointing_Groups[i]['visit_id']))
            f.write('cd Pointing_Group_'+str(i).zfill(3)+'\n')
            f.write('ingestImagesDecam.py processed_data --filetype instcal --mode=link ingest/instcal/* >& ingest.log\n')
            f.write('processCcd.py processed_data/ --rerun rerun_processed_data --id visit='+min_visit+'..'+max_visit+' --longlog -C configProcessCcd_neo.py -j'+str(num_cores)+' >& processCcd.log\n')
            f.write('cd ..\n')
        
def pickle_pointings(Pointing_Groups,filename='PickledPointings.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(Pointing_Groups, f)

def plot_pointings(visit_id, ra, dec):
    arcsec2degree = 0.000277778
    pixscale = 0.2637 #arcsec/px

    height   = 2048 * pixscale * arcsec2degree
    width    = 4096 * pixscale * arcsec2degree
    angle    = [0]*len(ra)

    fig, ax = plt.subplots(figsize=(10, 7))

    rect_patches = []
    for i, r, d, a in zip(visit_id, ra, dec, angle):
        # this will likely kill the notebook
        #annot = ax.text(r-0.001, d-0.001, i, fontsize=12)
        #                 lower left corner        height  width  angle
        rect = Rectangle((r-height/2., d-width/2.), height, width, a)
        rect_patches.append(rect)
    rect_patches_collection = PatchCollection(rect_patches, alpha=0.1)
    ax.add_collection(rect_patches_collection)

    ax.scatter(ra, dec, color="red")
