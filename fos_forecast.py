#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import requests
import argparse
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from sklearn.metrics import mean_squared_error,  r2_score
#from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
#import plotly.graph_objects as go
#import plotly.io as pio
#import matplotlib.cm as cm
#import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Optional
import openmeteo_requests
import requests_cache
from retry_requests import retry
from calculations import cyclic_fit,  calculate_relative_humidity, find_arima_order, add_albedo_values
from predictions import BlobClientLoader,KeyvaultSecretClient, EnvSecretClient, load_model,VWC_RF_predictions,Pastas_predictions,FoS_Predictions
from fetch_data import convert_time_for_frost_api,fetch_from_ngi_live, GetWeatherForecast,get_access_token



# In[2]:

def run(current_time: Optional[datetime]):

    # Enable loading environment variables from .env file
    load_dotenv()

    # Check that time zone is specified
    if current_time.tzinfo is None or current_time.tzinfo.utcoffset(current_time) is None:
        raise Exception("Need to specify 'aware' time stamp for current time")

    # Create azure blob client if configured
    if os.environ["USE_BLOB_STORAGE"].lower() == "true":
        print("Using blob storage")
        blob_client_loader = BlobClientLoader(os.environ["AZ_STORAGE_ACCOUNT_URL"], os.environ["AZ_INPUT_CONTAINER_NAME"])
    else:
        blob_client_loader = None

    # Load secrets. Use key vault if available
    if os.environ["USE_KEY_VAULT"].lower() == "true":
        print("Using keyvault")
        secret_client = KeyvaultSecretClient(f"https://{os.environ['AZ_KEY_VAULT_NAME']}.vault.azure.net")
    else:
        secret_client = EnvSecretClient()

    # Keep common current time on same format as data queries
    current_time_str = current_time.isoformat()

    # Calculate the start time as n_days days before the end time
    n_days=int(os.environ["DAYS_FOR_DATA_COLLECTION"]) #number of days for data collection from APIs
    start_time = current_time - timedelta(days=n_days)
    end_time = current_time

    n = int(os.environ["DAYS_FOR_CUMULATIVE_RAINFALL"]) #number of days for cumulative rainfall

    specified_duration = int(os.environ["HOURS_FOR_FORECASTS"]) #hours for forecasts
    model_name=os.environ["MODEL_NAME"] #choose RF or Pastas 
    Features_FoS=int(os.environ["FEATURES_FOS"])#Choose 10 OR 9


    # In[4]:
    df_1a = fetch_from_ngi_live(int(os.environ['NGILIVE_PROJECT_ID']), start_time, end_time, "PZ01", "Poretrykk", secret_client)
    df_1b = fetch_from_ngi_live(int(os.environ['NGILIVE_PROJECT_ID']), start_time, end_time, "DL1", "vwc", secret_client)
    df_1 = pd.merge(df_1a, df_1b, on="timestamp")
    #print(df_1)

    # Drop the first row
    df_1 = df_1.drop(0)

    # Reset index if needed
    df_1 = df_1.reset_index(drop=True)

    df_1.head()


    # In[7]:


    # Insert your own client ID here
    client_id = secret_client.get_secret(os.environ['FROST_API_CLIENT_ID_SECRET'])

    # Frost API reference time
    frost_reference_time = f'{convert_time_for_frost_api(start_time)}/{convert_time_for_frost_api(end_time)}'

    # Define endpoint and parameters
    endpoint = 'https://frost.met.no/observations/v0.jsonld'
    parameters = {
        'sources': 'SN4780', # Error response from Minnensund station, and therefore using Lufthavn station
        'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D)',
        'referencetime': frost_reference_time,
    }
    # Issue an HTTP GET request
    r2 = requests.get(endpoint, parameters, auth=(client_id,''))
    print(r2.ok)
    print(r2.reason)
    # Extract JSON data
    json = r2.json()

    #print(json)


    # In[8]:


    # Check if the request worked, print out any errors
    if r2.status_code == 200:
        data = json['data']
        #print('Data retrieved from frost.met.no!')
    else:
        print('Error! Returned status code %s' % r2.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])


    # In[9]:

    rows = []
    #print(data[2])
    # Iterate through the data and create dictionaries
    for i in range(len(data)):
        row_dict = {'referenceTime': data[i]['referenceTime'], 'sourceId': data[i]['sourceId']}

        # Handle each element in observations separately
        for obs in data[i]['observations']:
            observation_key = f'{obs["elementId"]}'
            row_dict.update({
                f'{observation_key}_value': obs['value'],
                f'{observation_key}_unit': obs['unit'],
                f'{observation_key}_timeOffset': obs['timeOffset']
            })
        rows.append(row_dict)

    # Convert the list of dictionaries to a DataFrame
    df2 = pd.DataFrame(rows)

    # Reset the index
    df2 = df2.reset_index()
    #print(df2)
    

    # Display the first few rows of the resulting DataFrame
    #df2.head()


    # In[10]:


    
    columns = ['sourceId','referenceTime','mean(air_temperature P1D)_value','mean(air_temperature P1D)_unit','mean(air_temperature P1D)_timeOffset','sum(precipitation_amount P1D)_value','sum(precipitation_amount P1D)_unit','sum(precipitation_amount P1D)_timeOffset']
    df3 = df2[columns].copy()
    # Convert the time value 
    df3['referenceTime'] = pd.to_datetime(df3['referenceTime'], utc=True)
    #df3.head()
    #print(df3.shape)
    #print(df3)


    # In[11]:


    #collect dew point temparature data from Lufthavn station
    parameters = {
        'sources': 'SN4780',
        'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D),mean(wind_speed P1D),mean(solar_irradiance PT1H),max(relative_humidity P1D),mean(dew_point_temperature P1D)',
        'referencetime': frost_reference_time,
    }
    # Issue an HTTP GET request
    r2 = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r2.json()
    #print(json)


    # In[12]:


    # Check if the request worked, print out any errors
    if r2.status_code == 200:
        data = json['data']
        #print('Data retrieved from frost.met.no!')
    #else:
        #print('Error! Returned status code %s' % r2.status_code)
        #print('Message: %s' % json['error']['message'])
        #print('Reason: %s' % json['error']['reason'])


    # In[13]:


    rows = []
    #print(data[2])
    # Iterate through the data and create dictionaries
    for i in range(len(data)):
        row_dict = {'referenceTime': data[i]['referenceTime'], 'sourceId': data[i]['sourceId']}

        # Handle each element in observations separately
        for obs in data[i]['observations']:
            observation_key = f'{obs["elementId"]}'
            row_dict.update({
                f'{observation_key}_value': obs['value'],
                f'{observation_key}_unit': obs['unit'],
                f'{observation_key}_timeOffset': obs['timeOffset']
            })
        rows.append(row_dict)

    # Convert the list of dictionaries to a DataFrame
    df4 = pd.DataFrame(rows)
    #print(df4)

    # Reset the index
    df4 = df4.reset_index()
    # Drop columns with only NaN values
    df4 = df4.dropna(axis=1, how='all')

    #print(df4)

    # Display the first few rows of the resulting DataFrame
    #df4.head()

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 60.322,
        "longitude": 11.245,
        "start_date": start_time.date(),
        "end_date": end_time.date(),
        "hourly": ["snow_depth"],
        "timezone": "UTC"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_snow_depth = hourly.Variables(0).ValuesAsNumpy()
    

    hourly_data = {"hour": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["snow_depth"] = hourly_snow_depth
    

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    #print(hourly_dataframe)
    # Extract date from 'hour' column
    hourly_dataframe['date'] = hourly_dataframe['hour'].dt.date

    # Calculate daily average snow depth 
    df5 = hourly_dataframe.groupby('date').agg({'snow_depth': 'mean', }).reset_index()

    # Rename columns
    df5.rename(columns={'snow_depth': 'snow_depth_avg'}, inplace=True)
    df5['date'] = pd.to_datetime(df5['date'])
    # Set a fixed time value
    fixed_time = pd.Timestamp('00:00:00')

    # Combine date with fixed time and localize to UTC
    df5['referenceTime'] = pd.to_datetime(df5['date'].dt.strftime('%Y-%m-%d') + ' ' + fixed_time.strftime('%H:%M:%S')).dt.tz_localize('UTC')


    #print(df5)


    # In[14]:


    # Convert 'timestamp' column in df to datetime with UTC timezone
    df_1['timestamp'] = pd.to_datetime(df_1['timestamp'], utc=True)

    # Convert 'referenceTime' column in df3 to the timezone of 'timestamp' column in df
    df3['referenceTime'] = df3['referenceTime'].dt.tz_convert(df_1['timestamp'].dt.tz)


    # In[15]:


    # Convert 'referenceTime' to datetime type
    df4['referenceTime'] = pd.to_datetime(df4['referenceTime'], utc=True)

    # Now use .dt accessor
    df4['referenceTime'] = df4['referenceTime'].dt.tz_convert(df_1['timestamp'].dt.tz)


    # In[16]:


    # Step 1: Convert 'referenceTime' to datetime type and extract date
    df4['referenceTime'] = pd.to_datetime(df4['referenceTime'], utc=True)
    df4['date'] = df4['referenceTime'].dt.date
       

    # Step 3: Merge daily_avg_dew_point with df3 based on 'date'
    df3['referenceTime'] = pd.to_datetime(df3['referenceTime'], utc=True)  # Make sure 'reference_time' is datetime type
    df3['date'] = df3['referenceTime'].dt.date
    # print(df3)
    # print(df4)
    # Merge df4 with df3 based on the 'reference_time' column
    df3 = pd.merge(df3, df4[['referenceTime', 'mean(dew_point_temperature P1D)_value', 'mean(wind_speed P1D)_value']], on='referenceTime', how='left')
    # Drop the 'date' column from df5
    df5.drop(columns=['date'], inplace=True) 


    # Print df3 to see the changes
    #print(df3)


    # In[17]:


    # Get the last value in the 'referenceTime' column
    last_reference_time = df3['referenceTime'].iloc[-1]
    time_difference = end_time - last_reference_time
    
    hours_difference = time_difference.total_seconds() / 3600
    #print(hours_difference)

    # Print the updated DataFrame
    #print(df3)
    if hours_difference <= 120:
        #If Minnensund station has data
        # Apply the function to calculate relative humidity and create a new column
        df3['relative_humidity'] = df3.apply(calculate_relative_humidity, axis=1)
    
        # Merge df5 with df3 based on the 'reference_time' column, and add the columns from df5 to df3
        df3 = pd.merge(df3, df5, on='referenceTime', how='left')
        # merging
        merged_df = pd.merge(df_1, df3, left_on='timestamp', right_on='referenceTime', how='outer')
    

    

    else:
        #If Minnensund station does not have data
        # Apply the function to calculate relative humidity and create a new column
        df4['relative_humidity'] = df4.apply(calculate_relative_humidity, axis=1)
        #print(df4)

        # Merge df5 with df4 based on the 'reference_time' column, and add the columns from df5 to df4
        df4 = pd.merge(df4, df5, on='referenceTime', how='left')
        # merging
        merged_df = pd.merge(df_1, df4, left_on='timestamp', right_on='referenceTime', how='outer')

    # print(df3)
    # print(df_1)
    # print(end_time)  


    # In[18]:


    

    # Drop the redundant 'referenceTime' column if needed
    merged_df = merged_df.drop('referenceTime', axis=1)

    # Display the resulting merged dataframe
    #merged_df.head()
    #print(merged_df.columns)
    #print(merged_df)


    # In[19]:


    #print(merged_df.shape)


    # In[20]:


    # # Check for columns with all NaN values
    columns_with_all_nan = merged_df.columns[merged_df.isna().all()]

    # # Drop columns with all NaN values
    merged_df = merged_df.drop(columns=columns_with_all_nan)

    # Print the updated shape
    #print(merged_df.shape)




    # In[22]:


    # Extract month and add as a separate column
    merged_df['month'] = merged_df['timestamp'].dt.month
    # Extract day of the year from the date
    merged_df['day_of_year'] = merged_df['timestamp'].dt.dayofyear
    # Load the fit parameters from the file

    if blob_client_loader:
        params_dir = "tmp_params"
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)
        blob_client_loader.load_blob(os.environ["FIT_PARAMS_PATH"], os.path.join(params_dir, os.environ["FIT_PARAMS_PATH"]))
        loaded_params = np.load(os.path.join(params_dir, os.environ["FIT_PARAMS_PATH"]))
    else:
        loaded_params = np.load(os.environ["FIT_PARAMS_PATH"])
    loaded_a, loaded_b, loaded_c, loaded_d = loaded_params['a'], loaded_params['b'], loaded_params['c'], loaded_params['d']

    # Generate values for the fitted curve using the loaded parameters
    merged_df['solar_radiation']  = cyclic_fit(merged_df['day_of_year'], loaded_a, loaded_b, loaded_c, loaded_d)

    # Use LAI values based on month column
    merged_df['LAI'] = merged_df['month'].apply(lambda x: 1.5 if 4 <= x <= 10 else 0)
    
    # Display the resulting merged dataframe
    #print(merged_df.columns)

    columns_to_convert = ['DL1_WC1', 'DL1_WC2', 'DL1_WC3', 'DL1_WC4', 'DL1_WC5', 'DL1_WC6', 'PZ01_D',
                       'mean(air_temperature P1D)_value', 'sum(precipitation_amount P1D)_value',
                       'mean(dew_point_temperature P1D)_value', 'relative_humidity', 'month', 'day_of_year',
                       'solar_radiation', 'LAI','snow_depth_avg','mean(wind_speed P1D)_value']

    # Convert specified columns to numeric
    merged_df[columns_to_convert] = merged_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    #print(merged_df)
    merged_df['date'] = pd.to_datetime(merged_df['date'], utc=True)
    # Create a complete date range
    date_range = pd.date_range(start=merged_df['timestamp'].min(), end=merged_df['timestamp'].max(), freq='D')

    # Create a DataFrame with the complete date range
    complete_df = pd.DataFrame({'timestamp': date_range})

    # Merge the original DataFrame with the complete date range DataFrame
    merged_df = pd.merge(complete_df, merged_df, on='timestamp', how='left')

    # Interpolate missing values in all columns
    merged_df = merged_df.interpolate()
    #print(merged_df)
    # # Save DataFrame to a pickle file
    # merged_df.to_pickle("merged_df.pkl")
    # # Read DataFrame from the pickle file
    # merged_df = pd.read_pickle("merged_df.pkl")    


    


    # In[23]:
        
     ## Get the weather forecast
    Time, Duration, RelativeHumidity, AirTemperature, Precipitation,wind_speed = GetWeatherForecast()
    ## Only use first 48 hours
    #Time = Time[0:72]
    Time = [datetime.strptime(EachTime, "%Y-%m-%dT%H:%M:%SZ") for EachTime in Time]

    ## Add zero time for instant values (or initial analysis) and get first 48 hours
    Duration.insert(0,0.0)
    #Duration  = Duration[0:72]

    ## Make a cumulative time list
    Duration=np.cumsum(Duration) ## *3600.0 to convert it to seconds

    ## Only use first 72 hours
    #AirTemperature   = AirTemperature[0:72] ## Instant
    #RelativeHumidity = RelativeHumidity[0:72] ## Instant
    #Precipitation    = Precipitation[0:72]
    # Create a dictionary with your data
    data1 = {
        'Time': pd.to_datetime(Time, utc=True),
        'Duration': Duration,
        'RelativeHumidity': RelativeHumidity,
        'AirTemperature': AirTemperature,
        'Precipitation': Precipitation,
        'wind_speed':wind_speed
    }
    #print(data1)
    # Filter the data based on the specified duration
    filtered_indices = [i for i, d in enumerate(Duration) if d <= specified_duration]

    data = {key: [data1[key][i] for i in filtered_indices] for key in data1}
    #print(data)
    # In[25]:
     #Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 60.322,
        "longitude": 11.245,
        "hourly": ["snow_depth"],
        "timezone": "UTC",
        "forecast_days": 3
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_snow_depth = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["snow_depth"] = hourly_snow_depth

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    #print(hourly_dataframe)

    # Rename the 'date' column to 'Time' in hourly_dataframe
    hourly_dataframe.rename(columns={'date': 'Time'}, inplace=True)
    # Rename other required columns
    hourly_dataframe.rename(columns={'snow_depth': 'Snow_depth'}, inplace=True)
    

    

     # In[25]:
    

    # Create a DataFrame from the dictionary
    forecast_df = pd.DataFrame(data)
    forecast_df.rename(columns={'wind_speed': 'Wind_speed'}, inplace=True)
    #print(forecast_df)
    # Convert 'Time' column to datetime format
    forecast_df['Time'] = pd.to_datetime(forecast_df['Time'], utc=True)
    # Add the 'snow_depth' column to 'forecast_df' based on the values in the 'Time' column
    forecast_df = pd.merge(forecast_df, hourly_dataframe[['Time', 'Snow_depth']], on='Time', how='left')
    #print(forecast_df)
    # Extract date from Time and create a new column 'date'
    forecast_df['date'] = (forecast_df['Time']).dt.date
    #print(type(forecast_df['Time']))

    # Drop the existing 'mid_time' column if it already exists
    if 'mid_time' in forecast_df.columns:
        forecast_df = forecast_df.drop(columns=['mid_time'])

    # Initialize 'mid_time' based on the first value in 'Time'
    forecast_df['mid_time'] = pd.to_datetime(forecast_df['Time'], utc=True) + pd.to_timedelta(12, unit='h')
    # Print 'mid_time' column
    #print(forecast_df['mid_time'])

    # Calculate the mid time for each 24-hour period
    for i in range(1, len(forecast_df)):
        time_difference = forecast_df['Time'].iloc[i] - forecast_df['mid_time'].iloc[i-1]
        if time_difference >= pd.to_timedelta(12, unit='h'):
            forecast_df.at[i, 'mid_time'] = forecast_df['mid_time'].iloc[i-1] + pd.to_timedelta(24, unit='h')
        else:
            forecast_df.at[i, 'mid_time'] = forecast_df['mid_time'].iloc[i-1]

    # Print 'mid_time' column
    #print(forecast_df['mid_time'])

    forecast_df = forecast_df.drop(columns=['date', 'Time'],axis=1)
    # Ensure 'mid_time' is in datetime format
    forecast_df['mid_time'] = pd.to_datetime(forecast_df['mid_time'], utc=True)

    # Calculating average for all variables
    forecast_df_mean_per_day = forecast_df.groupby(['mid_time']).mean().reset_index()

    # Calculating sum for Precipitation
    forecast_df_mean_per_day['Precipitation'] = forecast_df.groupby(['mid_time'])['Precipitation'].sum().reset_index()['Precipitation']

    # Add 'day_of_year' and 'month' columns to forecast_df_mean_per_day
    forecast_df_mean_per_day['day_of_year'] = forecast_df_mean_per_day['mid_time'].dt.dayofyear
    forecast_df_mean_per_day['month'] = forecast_df_mean_per_day['mid_time'].dt.month

    # Predict 'solar_radiation' using cyclic_fit
    forecast_df_mean_per_day['solar_radiation'] = cyclic_fit(forecast_df_mean_per_day['day_of_year'], loaded_a, loaded_b, loaded_c, loaded_d)

    # Calculate 'LAI'
    forecast_df_mean_per_day['LAI'] = forecast_df_mean_per_day['month'].apply(lambda x: 1.5 if 4 <= x <= 10 else 0)
    # Convert the index to datetime format
    forecast_df_mean_per_day['date'] = (forecast_df_mean_per_day['mid_time']).dt.date
    #forecast_df_mean_per_day = forecast_df_mean_per_day.drop(columns=['date', 'Time'])

    # Print or use the DataFrame as needed
    #print(forecast_df_mean_per_day)

    

        

    

    # In[26]:
    #print(merged_df['date'])
    merged_df.drop(['date'], axis=1, inplace=True)
    #print( merged_df.columns)

    # Define the mapping for column names
    column_mapping = {
        'timestamp': 'date',
        'mean(air_temperature P1D)_value': 'AirTemperature',
        'sum(precipitation_amount P1D)_value': 'Precipitation',
        'month':'month',
        'solar_radiation': 'solar_radiation',
        'LAI': 'LAI',
        'relative_humidity':'RelativeHumidity',
        'mean(wind_speed P1D)_value':'Wind_speed',
        'snow_depth_avg':'Snow_depth'
    }
    # Assign columns from merged_df to X with different names
    X = merged_df.rename(columns=column_mapping)[['date', 'AirTemperature', 'Precipitation', 'month',
                                                   'solar_radiation', 'LAI','RelativeHumidity','Wind_speed','Snow_depth']]
    #print(X)
    y = merged_df[['timestamp','DL1_WC1', 'DL1_WC2', 'DL1_WC3','DL1_WC4', 'DL1_WC5', 'DL1_WC6', 'PZ01_D']]
    #print(X.columns)
    # Convert the relevant columns to numeric before rounding
    y.iloc[:, 1:] = y.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').round(5)
    #print(y)
    #print(y)



    # In[27]:
        # Concatenate X and y for correlation analysis
    merged_data = pd.concat([X, y], axis=1)

    # Get the last 10 rows of the DataFrame
    df_observations_for_forecast = merged_data.copy()

    # Reset the index of the new DataFrame
    df_observations_for_forecast = df_observations_for_forecast.reset_index(drop=True)
    #print(df_observations_for_forecast)


    #print(df_observations_for_forecast)


    # In[28]:


    # Rename the 'date' column to 'mid_time' in df_observations_for_forecast
    df_observations_for_forecast.rename(columns={'date': 'mid_time'}, inplace=True)
    #print(df_observations_for_forecast)
    df_observations =df_observations_for_forecast[['mid_time', 'AirTemperature','Precipitation','month','solar_radiation','LAI','RelativeHumidity', 'Wind_speed','Snow_depth','DL1_WC1', 'DL1_WC2', 'DL1_WC3', 'DL1_WC4', 'DL1_WC5', 'DL1_WC6', 'PZ01_D']]

    # Ensure 'mid_time' columns have the same datetime format and timezone
    df_observations['mid_time'] = pd.to_datetime(df_observations['mid_time'], utc=True).dt.tz_localize(None)  # Remove timezone information
    forecast_df_mean_per_day['mid_time'] = pd.to_datetime(forecast_df_mean_per_day['mid_time'], utc=True).dt.tz_localize(None)  # Remove timezone information
    forecast_df_mean_per_day = forecast_df_mean_per_day.drop(columns=['Duration','day_of_year'],axis=1)

    # Concatenate dataframes sorted by 'mid_time' and merge common columns
    result_df = pd.concat([df_observations, forecast_df_mean_per_day], sort=False)
    result_df = result_df.groupby('mid_time').first().reset_index()


    # Sort by 'mid_time' and reset the index
    result_df = result_df.sort_values(by='mid_time').reset_index(drop=True)

    
    # Call the function to add albedo values to result_df
    result_df = add_albedo_values(result_df)
    # Print the resulting dataframe
    # print(result_df)
    # Save DataFrame to a pickle file
    #result_df.to_pickle("result_df.pkl")
 


    # In[30]:


    #result_df = result_df.dropna()
    
    # In[31]:

    # Read DataFrame from the pickle file
    #result_df = pd.read_pickle("result_df.pkl")  
    #print(result_df) 
    if (model_name=="RF"):
        loaded_model = load_model(os.environ["MODEL_PATH"], blob_client_loader)
        Dataframe_with_predictions =VWC_RF_predictions(result_df,loaded_model)
        predictions_df = pd.DataFrame(Dataframe_with_predictions, columns=['VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m', 'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m'])
        # Add 'mid_time' column from 'result_df' to 'predictions_df' and ignore index
        predictions_df['mid_time'] = result_df['mid_time'].values
    elif (model_name=="Pastas"):
        #print(result_df)
        #print(merged_df)
        Dataframe_with_predictions =Pastas_predictions(result_df,forecast_df_mean_per_day,specified_duration)
        Dataframe_with_predictions.dropna(subset=['VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m', 'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m'], inplace=True)
        predictions_df = pd.DataFrame(Dataframe_with_predictions, columns=['mid_time','VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m', 'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m'])
        # Create a mapping dictionary with 'mid_time' as key and corresponding 'mid_time' value
        mid_time_to_mid_time = dict(zip(result_df['mid_time'], result_df['mid_time']))
        # Replace the 'mid_time' column in 'predictions_df' with the corresponding values
        predictions_df['mid_time'] = predictions_df['mid_time'].map(mid_time_to_mid_time)


       

    # Reorder columns
    predictions_df = predictions_df[['mid_time', 'VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m',  'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m']]
    #print(predictions_df)
    #print(result_df)

    # Merge with 'result_df' based on 'mid_time' and ignore index
    final_result = pd.merge(result_df, predictions_df, on='mid_time')
    
    #print(final_result)
    
    final_result=FoS_Predictions(final_result,Features_FoS,blob_client_loader)




    # In[36]:


    # Define the mapping for column names
    column_mapping = {
        'date': 'mid_time',
        'DL1_WC1': 'VWC_0.1m',
        'DL1_WC2': 'VWC_0.5m',
        'DL1_WC3': 'VWC_1.0m',
        'DL1_WC4': 'VWC_2.0m',
        'DL1_WC5': 'VWC_4.0m',
        'DL1_WC6': 'VWC_6.0m',
        'PZ01_D': 'PP_6.0m'
    }

    # Assign columns from merged_df to X with different names
    df_observations_renamed = df_observations_for_forecast.rename(columns=column_mapping)[
        ['mid_time', 'VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m', 'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m']]
    #print('df_observations_for_forecast:',df_observations_for_forecast)
    #print('df_observations_renamed:',df_observations_renamed)



    # Convert 'mid_time' columns to the same format with timezone information (UTC)
    df_observations_renamed['mid_time'] = pd.to_datetime(df_observations_renamed['mid_time'], utc=True)
    final_result['mid_time'] = pd.to_datetime(final_result['mid_time'], utc=True)

    # Merge DataFrames on 'mid_time'
    #merged_df_to_plot = pd.merge(df_observations_renamed, final_result, on='mid_time', how='inner', suffixes=('_observed', '_predicted'))



    # In[37]:


    #print(merged_df_to_plot)


    # In[38]:


    # List of variables to plot
    #variables_to_plot = ['MEAS_0.1', 'MEAS_0.5', 'MEAS_1.0', 'MEAS_2.0', 'MEAS_4.0', 'MEAS_6.0', 'Poretrykk_6m']

    # Set up subplots
    #fig, axs = plt.subplots(len(variables_to_plot), 1, figsize=(10, 2 * len(variables_to_plot)), sharex=True)

    # Initialize lists to store metrics
    #mse_values = []
    #rmse_values = []
    #r2_values = []

    # Plot each variable
    #for i, variable in enumerate(variables_to_plot):
        # Convert observed and predicted columns to numeric
        #observed_values = pd.to_numeric(merged_df_to_plot[variable + '_observed'], errors='coerce')
        #predicted_values = pd.to_numeric(merged_df_to_plot[variable + '_predicted'], errors='coerce')

        # Calculate metrics
        #mse = mean_squared_error(observed_values, predicted_values)
        #rmse = np.sqrt(mse)
        #r2 = r2_score(observed_values, predicted_values)

        # Append values to lists
        #mse_values.append(mse)
        #rmse_values.append(rmse)
        #r2_values.append(r2)

        # Plot true observations
        #axs[i].plot(merged_df_to_plot['mid_time'], observed_values, label='True Observations', marker='o', linestyle='-', color='blue')

        # Plot predictions
        #axs[i].plot(merged_df_to_plot['mid_time'], predicted_values, label='Predictions', marker='x', linestyle='--', color='red')

        #axs[i].set_ylabel(variable)
        #axs[i].legend()

        # Set ticks and format for y-axis based on min and max values
        #min_value = min(observed_values.min(), predicted_values.min())
        #max_value = max(observed_values.max(), predicted_values.max())
        #axs[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        #axs[i].set_yticks([min_value, max_value])
        #axs[i].yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))  # Format with 2 digits after the decimal point

    # Set common X-axis label
    #axs[-1].set_xlabel('mid_time')

    # Show the plot
    #plt.tight_layout()
    #plt.show()

    # Print the calculated metrics
    #for variable, mse, rmse, r2 in zip(variables_to_plot, mse_values, rmse_values, r2_values):
        #print(f'{variable}: MSE = {mse}, RMSE = {rmse}, R^2 = {r2}')


    # In[39]:
    #final_result.to_pickle('final_result.pkl')
    #df_observations_renamed.to_pickle('fdf_observations_renamed.pkl')
# final_result = pd.read_pickle('final_result.pkl') 
# df_observations_renamed = pd.read_pickle('fdf_observations_renamed.pkl') 

    variables_to_plot = ['VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m', 'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m']
    # Extract relevant columns for scaling from 'df_observations_renamed'
    columns_to_scale = [col for col in df_observations_renamed.columns[1:] if col in variables_to_plot]

    # Normalize the data on a scale from 0 to 1 for columns present in 'variables_to_plot'
    scaler_observation = MinMaxScaler()
    scaled_data_observations = scaler_observation.fit_transform(df_observations_renamed[columns_to_scale])
    #print(df_observations_renamed)

    # Create a DataFrame with scaled observations
    scaled_df_observations = pd.DataFrame(scaled_data_observations, columns=df_observations_renamed.columns[1:])
    scaled_df_observations['mid_time'] = df_observations_renamed['mid_time']
    #print(scaled_df_observations)

    # Check the last three rows in 'final_result' for the same variable names
    last_three_rows = final_result.iloc[-3:, :]

    # Scale the data for the last three rows using the same scaler for columns present in 'variables_to_plot'
    scaled_data_last_three_rows = scaler_observation.transform(last_three_rows.loc[:, columns_to_scale])

    # Create a DataFrame with scaled values for the last three rows
    scaled_last_three_rows = pd.DataFrame(scaled_data_last_three_rows, columns=columns_to_scale, index=last_three_rows.index)



# Transpose the DataFrame, add 'Item' column, and rename columns

#     transposed_df = (
#     scaled_last_three_rows
#     .T
#     .reset_index()
#     .rename(columns={i: f'Day_{i+1}' for i in range(len(scaled_last_three_rows.columns))})
#     .rename(columns={'index': 'Item'})
#     )

#     print(transposed_df)
# # # Iterate over rows

    # import plotly.io as pio
    # import plotly.express as px
    # import plotly.graph_objects as go
    # import plotly.io as pio

# # for i in range(1, 4):  # 'Day_1', 'Day_2', 'Day_3' columns
# #     fig = px.bar_polar(
# #         transposed_df,
# #         r=f"Day_{i}",
# #         theta="Item",
# #         template="plotly_dark",
# #         color_discrete_sequence=['darkviolet'],  # Set the color to dark violet
# #     )
# #     fig.update_layout(
# #         title=f'Windrose - Day {i}',
# #         font=dict(family="Arial", size=20, color='white'),  # Set font to Arial, bold, white color
# #     )
# #     fig.show()




    # fig = go.Figure()

    # for i in range(1, 4):
    #     # Extract data for the current day
    #     current_day_data = transposed_df[['Item', f'Day_{i}']]

    #     # Define color for current iteration
    #     color = f'rgb({i * 50}, {255 - i * 50}, {i * 100})'

    #     # Create a radar plot without internal fill and connecting the first and last points
    #     fig.add_trace(go.Scatterpolar(
    #         r=current_day_data[f'Day_{i}'].tolist() + [current_day_data[f'Day_{i}'].iloc[0]],
    #         theta=current_day_data['Item'].tolist() + [current_day_data['Item'].iloc[0]],
    #         mode='lines+markers',  # Add markers for each point
    #         line=dict(color=color, width=8),  # Set line color and width
    #         marker=dict(color='black', size=16, line=dict(color='black', width=2)),  # Set marker properties
    #         name=f'Day_{i}'  # Set legend name
    #     ))

    # # Update layout
    # fig.update_layout(
    #     polar=dict(
    #         radialaxis=dict(
    #             visible=True,
    #             range=[0, 1.5],
    #             tickvals=[0, 0.5, 1, 1.5],  # Specify tick marks
    #             tickfont=dict(size=32, family='Arial', color='black'),  # Set tick font
    #             tickmode='array',
    #             ticktext=['<b>0</b>', '<b>0.5</b>', '<b>1</b>', '<b>1.5</b>'],  # Use HTML <b> tag for bold
    #             tickangle=0,
    #             tickprefix='',
    #             ticks='outside',
    #             ticklen=5,
    #             tickwidth=1,
    #             tickcolor='rgba(0,0,0,0)',
    #         )
    #     ),
    #     title='Windrose - All Days',
    #     font=dict(family='Arial', size=20, color='black'),  # Set font to Arial Bold
    #     showlegend=True  # Show legend
    # )

    # # Show the plot
    # fig.show()


# #Save the figure as an image
# file_name = f'radar.png'
# pio.write_image(fig, file_name, format='png', width=800, height=800, scale=2) 
# Add absolute values with suffix '_abs' to the DataFrame
    scaled_last_three_rows = scaled_last_three_rows.join(last_three_rows[columns_to_scale].add_suffix('_abs'))    

    scaled_last_three_rows['mid_time'] = last_three_rows['mid_time']
    scaled_last_three_rows['FoS_predictions'] = last_three_rows['FoS_predictions']
    scaled_last_three_rows['FoS_predictions_PR'] = last_three_rows['FoS_predictions_PR']
    # for index, row in scaled_last_three_rows.iterrows():
    #     fig = go.Figure()

    #     # Set the color scale from red to green
    #     color_scale = [
    #         {'range': [0, 1], 'color': 'darkred'},
    #         {'range': [1, 1.5], 'color': 'red'},
    #         {'range': [1.5, 2], 'color': 'orange'},
    #         {'range': [2, 2.5], 'color': 'yellow'},
    #         {'range': [2.5, 3], 'color': 'green'}
    #     ]

    #     # Add the gauge indicators
    #     fig.add_trace(go.Indicator(
    #         mode="gauge+number",
    #         value=3,
    #         domain={'x': [0, 1], 'y': [0, 1]},
    #         title={'text': f'Day {index + 1}'},
    #         gauge=dict(
    #             axis=dict(range=[0, 3]),
    #             bar=dict(color='white', thickness=0.85),
    #             bgcolor="white",
    #             borderwidth=2,
    #             bordercolor="white",
    #             steps=color_scale,
    #         )
    #     ))

    #     fig.add_trace(go.Indicator(
    #         mode="gauge+number",
    #         value=row['FoS_predictions_PR'],
    #         domain={'x': [0, 1], 'y': [0, 1]},
    #         title={'text': f'Day {index + 1}'},
    #         gauge=dict(
    #             axis=dict(range=[0, 3]),
    #             bar=dict(color='orange', thickness=0.85),
    #             bgcolor="rgba(0, 0, 0, 0)",
    #             borderwidth=2,
    #             bordercolor="rgba(0, 0, 0, 0)",
    #             steps=[],
    #         )
    #     ))
    #     fig.update_layout(
    #         title_text=f'FoS Predictions - Day {index + 1}',
    #         font=dict(family="Arial", size=20, color='black'),
    #         showlegend=False,
    #         plot_bgcolor="white"
    #     )
    #     fig.show()

# Print the scaled values along with 'mid_time' and 'FoS_predictions' columns
    print(scaled_last_three_rows)

    # Rename to valid names: feel free to change the names :)
    def col_rename(col: str):
        return col.strip().replace('.', '_')
    renames_scaled_last_three_rows = scaled_last_three_rows.rename(col_rename, axis='columns')

    # Convert time to posix time (values need to be numbers)
    renames_scaled_last_three_rows['mid_time'] = renames_scaled_last_three_rows['mid_time'].apply(lambda t: t.timestamp())

    access_token = get_access_token(
        os.environ["NGILIVE_INGEST_API_TOKEN_PROVIDER_URL"],
        secret_client.get_secret(os.environ['NGILIVE_INGEST_API_CLIENT_ID_SECRET']),
        secret_client.get_secret(os.environ['NGILIVE_INGEST_API_CLIENT_SECRET_SECRET'])
    )
    for idx, row in renames_scaled_last_three_rows.iterrows():
        # time_col = "mid_time"
        # payload = {"timestamp": row[time_col].isoformat(), "values": {col: row[col] for col in renames_scaled_last_three_rows.columns if col != time_col}}
        payload = {"timestamp": current_time_str, "values": {f"day{idx+1}_{col}": row[col] for col in renames_scaled_last_three_rows.columns}}
        print(f"Publishing: {payload}")
        r = requests.post(
            f"{os.environ['NGILIVE_INGEST_API_URL']}/measurement",
            json=payload,
            headers={"Authorization": f"Bearer {access_token}"}
        )
        if not r.ok:
            raise Exception(f"Request to ngi live ingest API failed with error code {r.status_code}: {r.reason}")



    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='SlopeStabilityDT', description='Predicts slope stability')
    parser.add_argument('-t', '--timestamp')
    args = parser.parse_args()
    if args.timestamp:
        run(datetime.fromisoformat(args.timestamp))
    else:
        run(datetime.now(timezone.utc))
