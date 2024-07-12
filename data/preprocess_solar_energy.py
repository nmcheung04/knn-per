import os
import pandas as pd
import numpy as np
import itertools
from netCDF4 import Dataset
from os import listdir
from sklearn.preprocessing import StandardScaler

def pre_processing(data_dir = os.getcwd() + "/raw_data/train/", agg_dims = ['hours', 'ens_models'], crop_gefs_x_start = 2, crop_gefs_x_end = 6, crop_gefs_y_start = 3, crop_gefs_y_end = 12):
    #Looping through files holding GEFS data
    for n, f in enumerate(os.listdir(data_dir)):
        if not f.endswith('.nc'):
            continue
        # print(n,f)
        # Reading data
        nc_ = Dataset(f'{data_dir}{f}')

        # extracing GEFS variable from netcdf
        nc_gefs = list(nc_.variables.values())[-1]
        # print(nc_gefs)

        if agg_dims == 'hours':
            # taking mean of the measurements taken in different times
            # print(f'shape before processing {nc_gefs.shape}')
            nc_agg = np.mean(nc_gefs, axis = 2)
            # print("shape after taking mean of hours: ")
            # print(nc_agg.shape)

        elif agg_dims == 'ens_models':
            #taking mean of measurements taken in different times
            # print(f'shape before processing {nc_gefs.shape}')
            nc_agg = np.mean(nc_gefs, axis = 1)
            # print("shape after taking mean of hours: ")
            # print(nc_agg.shape)

        else:
            #taking mean of the measurements taken in different times
            nc_agg = np.mean(nc_gefs, axis = 2)
            # print("shape after taking mean of hours: ")
            # print(nc_agg.shape)
            nc_agg = np.mean(nc_agg, axis = 1)
            # print("shape after taking mean of ensembles: ")
            # print(nc_agg.shape)

        # print("=="*50)
        # print("AGGREGATING DONE!")
        # print("STARTING CROPPING...")

        #cropping GEFS based on lat-lon
        if agg_dims == 'hours':
            # print(f"shape before crop {nc_agg.shape}")
            nc_agg_cropped = nc_agg[:,:,crop_gefs_x_start:crop_gefs_x_end, crop_gefs_y_start:crop_gefs_y_end]
            # print("fshape after crop{nc_agg_cropped.shape}")

        elif agg_dims == 'ens_models':
            # print(f"shape before crop {nc_agg.shape}")
            nc_agg_cropped = nc_agg[:,:,crop_gefs_x_start:crop_gefs_x_end, crop_gefs_y_start:crop_gefs_y_end]
            # print(f"shape after crop {nc_agg_cropped.shape}")

        else:
            # print(f"shape before crop {nc_agg.shape}")
            nc_agg_cropped = nc_agg[:, crop_gefs_x_start:crop_gefs_x_end, crop_gefs_y_start:crop_gefs_y_end]
            # print(f"shape after crop {nc_agg_cropped.shape}")

        # print("=="*50)
        # print("CROPPING DONE!")
        # print("STARTING RESHAPING...")

        # reshaping into 2 dimensions and converting to dataframe
        if agg_dims == 'hours':
            # print("we have to form the shape dynamically too, rather than just giving the numbers")
            df_gefs_ = pd.DataFrame(nc_agg_cropped.reshape(np.prod(nc_agg_cropped.shape),1)) #5113 * 4 * 9 * 11
        elif agg_dims == 'ens_models':
            print(nc_agg_cropped.shape)
            df_gefs_ = pd.DataFrame(nc_agg_cropped.reshape(np.prod(nc_agg_cropped.shape)//5,5)) #5113 * 4 * 9
        else: 
            df_gefs_ = pd.DataFrame(nc_agg_cropped.reshape(np.prod(nc_agg_cropped.shape),1)) #5113 * 4 * 9

        # print("=="*50)
        # print("RESHAPING DONE!")
        # print("STARTING RENAMING...")

        # we will use the name of file to rename columns so we don't get confused as we keep appending files
        prefix = f.split("latlon")[0]
        # print(prefix)

        #existing columns of dataframe we created
        cols = list(df_gefs_.columns)
        # print(cols)

        # creating new names for columns
        newcols = [prefix+str(c) for c in cols]
        # print(newcols)

        #creaitng the dictionary to rename cols accordingly
        rename_cols = {cols[i]: newcols[i] for i in range(len(cols))}
        # print(rename_cols)

        # changing names on the dataset and inplacing
        df_gefs_ = df_gefs_.rename(rename_cols, axis = 1)

        if n == 0:
            latlon = list(itertools.product(nc_['lat'][:][:].data[crop_gefs_x_start:crop_gefs_x_end], nc_['lon'][:][:].data[crop_gefs_y_start:crop_gefs_y_end]))
            df_nc_time = pd.DataFrame(nc_['intValidTime'][:][:].data)
            date_latlon = list(itertools.product(df_nc_time[0].apply(lambda x: int(str(x)[:8])), latlon))
            df_gefs = pd.DataFrame(date_latlon, columns = ['date', 'coordinates'])
            df_gefs = pd.concat([df_gefs, df_gefs_], axis = 1)

        else:
            df_gefs = pd.concat([df_gefs, df_gefs_],axis = 1)

        # print(df_gefs.shape)
    return df_gefs

from haversine import haversine
def custom_haversine(coord1, coordDf):
    l = {}
    for coord2, n_coord2 in list(zip(coordDf.coord, coordDf.normalized_coord)):
        l[coord2] = haversine(coord1, n_coord2)
    return l

def get_min_distance_node(df_gefs):
    #Get noramlized GEFS coordinates
    df_gefs_loc = pd.DataFrame(df_gefs['coordinates'].unique(), columns = ['coord'])
    df_gefs_loc['normalized_coord'] = df_gefs_loc['coord'].apply(lambda x: (x[0], x[1] - 360))
    # Read mesonet metadata
    df_mes = pd.read_csv(os.getcwd() + "/raw_data/station_info.csv")
    #zip coordinates for later calculations
    df_mes['coord'] = list(zip(df_mes.nlat, df_mes.elon))
    #calculate haversine distances between the area we cropped and mesonets
    df_mes['new_distances'] = df_mes['coord'].apply(lambda x: custom_haversine(x, df_gefs_loc))
    #Find minimum distances GEFS for every mesonset
    df_mes['min_dist_node'] = df_mes['new_distances'].apply(lambda x: list(x.keys())[list(x.values()).index(min(x.values()))])
    return df_mes

# print(get_min_distance_node(df_gefs))

def get_one_ens(df_gefs, ens_model = 0):
    #taking only one ensemble model = number 0 
    #ens = ['date', 'coordinates']
    ens = [col for col in df_gefs.columns if col.split("_")[-1] == f"{ens_model}"]
    ens.insert(0, 'date')
    ens.insert(1, 'coordinates')
    #crop df_gefs based on the ens model we want to get
    df_gefs_ens = df_gefs[ens]
    return df_gefs_ens

# print(get_one_ens(df_gefs, ens_model = 0))

def combine_ens_gefs(df_mes, df_train_csv):
    #Joins labels (daily solar power production) with related GEFS point(closest GEFS in grid)
    #create mesonet list
    mesonets = list(df_mes['stid'])

    #melt training dataframe so mesonets are in rows instead of columns
    df_train = pd.melt(df_train_csv, id_vars = 'Date', value_vars = mesonets, var_name = 'stid', value_name = 'Daily_Production')

    #combine distance related features from metadata to training data
    df_train = pd.merge(df_train, df_mes[['stid', 'new_distances', 'min_dist_node']], on = 'stid', how = 'left')

    df_gefs = pre_processing(data_dir = os.getcwd() + "/raw_data/train/", agg_dims = ['hours', 'ens_models'], crop_gefs_x_start=2, crop_gefs_x_end=6, crop_gefs_y_start=3, crop_gefs_y_end=12)
    df_gefs_ens = get_one_ens(df_gefs)
    
    #create join columns
    df_train['join'] = df_train['Date'].astype(str) + df_train['min_dist_node'].astype(str)
    df_gefs_ens['join'] = df_gefs_ens['date'].astype(str) + df_gefs_ens['coordinates'].astype(str)

    #join two dataframes
    df_train_merged = pd.merge(df_train, df_gefs_ens[['tcolc_eatm_0', 'ulwrf_tatm_0', 'dlwrf_sfc_0','tmp_sfc_0', 'tcdc_eatm_0', 'dswrf_sfc_0', 'tmax_2m_0', 'tmin_2m_0','pwat_eatm_0', 'uswrf_sfc_0', 'spfh_2m_0', 'ulwrf_sfc_0', 'tmp_2m_0','apcp_sfc_0', 'pres_msl_0', 'join']], how='left', on='join')
    return df_train_merged

def split_df_by_station(df):
    # Get unique station IDs
    unique_stations = df['stid'].unique()
    
    # Create a directory to save the split files
    output_dir = "./raw_data/by_station"  # Adjust as needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over each unique station ID
    for station_id in unique_stations:
        # Select rows for the current station ID
        df_station = df[df['stid'] == station_id]
        
        # Construct output file path
        output_file = os.path.join(output_dir, f"{station_id}_data.csv")
        
        # Save the split data to CSV
        df_station.to_csv(output_file, index=False)
        print(f"Saved {len(df_station)} rows to {output_file}")

def load_data():
    data_dir = os.getcwd() + "/raw_data/train/"
    df_gefs = pre_processing(data_dir=data_dir, agg_dims=['hours', 'ens_models'],
                             crop_gefs_x_start=2, crop_gefs_x_end=6, crop_gefs_y_start=3, crop_gefs_y_end=12)
    df_mes = get_min_distance_node(df_gefs)
    df_train_csv = pd.read_csv(os.getcwd() + "/raw_data/train.csv")
    
    df_train_merged = combine_ens_gefs(df_mes, df_train_csv)
    return df_train_merged

def normalize_features(df_train_merged):
    columns_to_normalize = ['Daily_Production', 'tcolc_eatm_0', 'ulwrf_tatm_0', 'dlwrf_sfc_0', 'tmp_sfc_0', 
                            'tcdc_eatm_0', 'dswrf_sfc_0', 'tmax_2m_0', 'tmin_2m_0', 
                            'pwat_eatm_0', 'uswrf_sfc_0', 'spfh_2m_0', 'ulwrf_sfc_0', 
                            'tmp_2m_0', 'apcp_sfc_0', 'pres_msl_0']
    
    scaler = StandardScaler()
    df_train_merged[columns_to_normalize] = scaler.fit_transform(df_train_merged[columns_to_normalize])
    return df_train_merged
    
def main():
    df_train_merged = load_data()
    training_set = df_train_merged[['Daily_Production', 'stid', 'tcolc_eatm_0', 'ulwrf_tatm_0', 'dlwrf_sfc_0', 'tmp_sfc_0', \
       'tcdc_eatm_0', 'dswrf_sfc_0', 'tmax_2m_0', 'tmin_2m_0', 'pwat_eatm_0', \
       'uswrf_sfc_0', 'spfh_2m_0', 'ulwrf_sfc_0', 'tmp_2m_0', 'apcp_sfc_0', 'pres_msl_0']]
    training_set_normalized = normalize_features(training_set)
    split_df_by_station(training_set_normalized)

if __name__ == "__main__":
    main()