import numpy as np
import pickle
from astropy.io.votable import parse
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

def pickle_pointings(Pointing_Groups,filename='PickledPointings.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(Pointing_Groups, f)
