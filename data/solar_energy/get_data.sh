#!/bin/bash

# Constants
KAGGLE_COMPETITION="ams-2014-solar-energy-prediction-contest"
RAW_DATA_PATH="raw_data"
TARGET_PATH="all_data"
PROCESSED_DATA_PATH="$RAW_DATA_PATH/by_station"

# Function to download and unzip Kaggle competition data
download_kaggle_dataset() {
    echo "------------------------------"
    echo "Retrieving raw data from Kaggle"
    kaggle competitions download -c $KAGGLE_COMPETITION -p $RAW_DATA_PATH
    unzip "$RAW_DATA_PATH/$KAGGLE_COMPETITION.zip" -d $RAW_DATA_PATH
    rm "$RAW_DATA_PATH/$KAGGLE_COMPETITION.zip"
}

# Function to extract GEFS train and test data
extract_gefs_data() {
    echo "------------------------------"
    echo "Extracting GEFS train and test data"
    tar -xzvf "$RAW_DATA_PATH/gefs_train.tar.gz" -C $RAW_DATA_PATH
    tar -xzvf "$RAW_DATA_PATH/gefs_test.tar.gz" -C $RAW_DATA_PATH
}

# Check if all_data directory is missing or empty
if [ ! -d "$TARGET_PATH" ] || [ ! "$(ls -A $TARGET_PATH)" ]; then
    # Create raw_data directory if it doesn't exist
    if [ ! -d "$RAW_DATA_PATH" ]; then
        mkdir -p $RAW_DATA_PATH
    fi

    # Download Kaggle dataset if not already downloaded
    if [ ! -f "$RAW_DATA_PATH/train.csv" ]; then
        download_kaggle_dataset
    fi

    # Extract GEFS train and test data
    if [ ! -d "$RAW_DATA_PATH/gefs_train" ] || [ ! -d "$RAW_DATA_PATH/gefs_test" ]; then
        extract_gefs_data
    fi
fi

# Check if processed data directory exists
if [ ! -d "$PROCESSED_DATA_PATH" ]; then
    echo "Dividing by station"
    python preprocess_solar_energy.py
fi
