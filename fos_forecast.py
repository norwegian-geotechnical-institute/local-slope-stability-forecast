#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import requests
import argparse
import pandas as pd
from io import StringIO
import numpy as np
#import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from sklearn.metrics import mean_squared_error,  r2_score
import joblib
import math
#from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.keyvault.secrets import SecretClient
import pastas as ps
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



# In[2]:


# Function to fit solar radiation as a cyclic variable
def cyclic_fit(x, a, b, c, d):
    return a * np.sin(b * (x - c)) + d


# Define functions to calculate saturation vapor pressure, actual vapor pressure, and relative humidity
def saturation_vapor_pressure(temperature):
    return 6.11 * math.exp(7.5 * temperature / (237.7 + temperature))

def actual_vapor_pressure(dew_point):
    return 6.11 * math.exp(7.5 * dew_point / (237.7 + dew_point))

def calculate_relative_humidity(row):
    temperature = row['mean(air_temperature P1D)_value']
    dew_point = row['mean(dew_point_temperature P1D)_value']
    e_s = saturation_vapor_pressure(temperature)
    e = actual_vapor_pressure(dew_point)
    return (e / e_s) * 100

def find_arima_order(time_series):
    # Use pmdarima's auto_arima to find the optimal order
    model = auto_arima(time_series, suppress_warnings=True)
    order = model.get_params()['order']
    return order


class BlobClientLoader:
    def __init__(self, account_url: str, container_name: str):
        default_credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url, credential=default_credential)
        self._container_client = blob_service_client.get_container_client(container_name)

    def load_blob(self, src_blob: str, dst_file: str):
        with open(file=dst_file, mode="wb") as download_file:
            download_file.write(self._container_client.download_blob(src_blob).readall())


class KeyvaultSecretClient:
    def __init__(self, vault_url: str):
        credential = DefaultAzureCredential()
        self._client = SecretClient(vault_url=vault_url, credential=credential)

    def get_secret(self, name: str):
        return self._client.get_secret(name).value


class EnvSecretClient:
    def get_secret(self, name: str):
        return os.environ[name]


def load_model(filename: str, blob_client_loader: BlobClientLoader | None):
    if blob_client_loader:
        models_dir = "tmp_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        blob_client_loader.load_blob(filename, os.path.join(models_dir, filename))
        return joblib.load(os.path.join(models_dir, filename))
    else:
        return joblib.load(filename)
    
def add_albedo_values(df):
    # Define a mapping of month and snow depth to albedo values
    # reference: https://doi.org/10.1080/01431161.2017.1320442 
    albedo_values = {
        (1, True): 0.278,  
        (1, False): 0.335,
        (2, True): 0.293,
        (2, False): 0.378,
        (3, True): 0.233,
        (3, False): 0.416,
        (4, True): 0.213,
        (4, False): 0.375,
        (5, True): 0.152,
        (5, False): 0.254,
        (6, True): 0.138,
        (6, False): 0.144,
        (7, True): 0.138,
        (7, False): 0.135,
        (8, True): 0.134,
        (8, False): 0.115,
        (9, True): 0.127,
        (9, False): 0.147,
        (10, True): 0.147,
        (10, False): 0.224,
        (11, True): 0.168,
        (11, False): 0.273,
        (12, True): 0.21,
        (12, False): 0.368,
    }
    
    def get_snow_depth(row, df):
        # Check if Snow_depth value is NaN
        if pd.isnull(row['Snow_depth']):
            # print("row.name:", row.name)  # Print row name for debugging
            # print("Length of DataFrame:", len(df))
             # Reset index to ensure it's numeric and continuous
            df_reset = df.reset_index(drop=True)
            
            # Find the nearest non-NaN value above and below the current row
            nearest_above = df_reset.loc[:row.name, 'Snow_depth'].dropna().iloc[-1] if row.name != 0 else float('inf')
            nearest_below_series = df_reset.loc[row.name:, 'Snow_depth'].dropna()
            nearest_below = nearest_below_series.iloc[0] if not nearest_below_series.empty else float('inf')

            # Check if nearest_below is infinity (indicating end of the DataFrame)
            if nearest_below == float('inf'):
                # If at the end, reverse the DataFrame and perform the operation
                df_reverse = df_reset[::-1]
                nearest_below_reverse = df_reverse.loc[:len(df_reset) - row.name - 1, 'Snow_depth'].dropna().iloc[-1] if row.name != len(df_reset) - 1 else float('inf')
                nearest_above_reverse = df_reverse.loc[len(df_reset) - row.name - 1:, 'Snow_depth'].dropna().iloc[0] if row.name != 0 else float('inf')
                # Use the nearest value from the reversed DataFrame
                nearest_below = nearest_below_reverse if abs(nearest_below_reverse - (len(df_reset) - row.name - 1)) < abs(nearest_above_reverse - (len(df_reset) - row.name - 1)) else nearest_above_reverse

            # Use the nearest value to determine whether snow is present or not
            snow_depth = nearest_above if abs(nearest_above - row.name) < abs(nearest_below - row.name) else nearest_below
        else:
            snow_depth = row['Snow_depth']
        return snow_depth > 0
    
    # Add albedo column with mapped values based on month and snow depth
    df['albedo'] = df.apply(lambda row: albedo_values.get((row['month'], get_snow_depth(row, df)), None), axis=1)
    
    return df
    
def VWC_RF_predictions(merged_df,loaded_model):
    #print(merged_df)
   
    # Define input and output variables
    X_forecast = merged_df[[ 'Precipitation','AirTemperature','Wind_speed', 'LAI', 'solar_radiation','RelativeHumidity','Snow_depth','albedo']]
    # Interpolate NaN values using linear interpolation
    X_forecast = X_forecast.interpolate(method='linear', limit_direction='both')

    # fill remaining NaN values with the last available value:
    X_forecast.fillna(method='ffill', inplace=True)
    # Rename columns in X_forecast
    X_forecast = X_forecast.rename(columns={
        'Precipitation':'sum(precipitation_amount P1D)_value',
        'AirTemperature': 'mean(air_temperature P1D)_value',
        'Wind_speed': 'mean(wind_speed P1D)_value',
        'RelativeHumidity': 'relative_humidity',
        'Snow_depth': 'snow_depth'
         })
    #print(X_forecast)

    # Make forecasts for Vol moist content and pore pressure
    y_forecasts = loaded_model.predict(X_forecast)

        #print(y_forecasts_original_scale)
    return y_forecasts
# In[4]:

def Pastas_predictions(merged_df,forecast_df_mean_per_day,specified_duration):
    #print(merged_df.columns)

    merged_df['date'] = pd.to_datetime(merged_df['mid_time'], utc=True).dt.date
    #print(merged_df)
    # Check for duplicate values in the 'date' column
    duplicate_mask = merged_df.duplicated(subset='date', keep='last')

    # Keep the last row for each duplicate date and delete other rows
    merged_df = merged_df[~duplicate_mask].sort_values(by='mid_time').reset_index(drop=True)
    
    # Check for discontinuity in dates
    date_range = pd.date_range(start=merged_df['date'].min(), end=merged_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(merged_df['date'])]
    #print(missing_dates)
    
    # Add rows corresponding to missing dates and interpolate values
    for missing_date in missing_dates:
        if missing_date.date() in merged_df['date'].values:
            continue  # Skip existing dates
        else:
            #print(missing_date.date())  # Print the date without the timestamp
            missing_date = missing_date.date()

            # Find the closest dates before and after the missing date with a timedelta of 1 day
            date_before = (pd.to_datetime(missing_date) - pd.Timedelta(days=1)).date()
            date_after = (pd.to_datetime(missing_date) + pd.Timedelta(days=1)).date()
            #print(date_before,date_after)

            
            # Find the index of 'before' or increment the timedelta until it's found
            index_before = None
            delta = 1
            while index_before is None:
                try:
                    date_before_candidate = (pd.to_datetime(missing_date) - pd.Timedelta(days=delta)).date()
                    index_before = merged_df[merged_df['date'] == date_before_candidate].index[0]
                except IndexError:
                    delta += 1
            #print(delta)

            # Find the index of 'date_after' 
            index_after = index_before+1
            #print(index_before,index_after)
            # Interpolate values for each column using linear interpolation
            interpolated_values = {}
            for col in merged_df.columns:
                if col == 'mid_time':
                    interpolated_values[col] = pd.to_datetime(f"{missing_date} 00:00:00", utc=True)
                elif col == 'date':
                    interpolated_values[col] = missing_date
                else:
                    x = [index_before, index_after]
                    #print(x)
                    # Select rows by their integer position using .iloc[]
                    y = [merged_df.iloc[index_before][col], merged_df.iloc[index_after][col]]
                    #print(y)
                    interpolated_values[col] = np.interp((index_before + ((delta) / (index_after - index_before))), x, y)

            # Add a new row with the current index of 'date_after'
            merged_df = pd.concat([merged_df.loc[:index_after], pd.DataFrame([interpolated_values], index=[index_after]), merged_df.loc[index_after + 1:]]).sort_index()



    # Convert 'mid_time' to datetime in UTC
    merged_df['mid_time'] = pd.to_datetime(merged_df['mid_time'], utc=True)

    # Sort the DataFrame by 'mid_time'
    merged_df = merged_df.sort_values(by='mid_time').reset_index(drop=True)
    #print(merged_df.columns)
    #print(merged_df)
    days=(specified_duration/24)+1
    # Define the column names and corresponding series names
    columns_to_series = {
        'DL1_WC1': 'vwcdata1',
        'DL1_WC2': 'vwcdata2',
        'DL1_WC3': 'vwcdata3',
        'DL1_WC4': 'vwcdata4',
        'DL1_WC5': 'vwcdata5',
        'DL1_WC6': 'vwcdata6',
        'PZ01_D': 'ppdata1',
        'AirTemperature': 'temp',
        'Precipitation': 'precip',
        'RelativeHumidity': 'hum',
        'solar_radiation': 'sol',
        'LAI': 'lai',
        'Wind_speed':'ws',
        'Snow_depth':'sd',
        'albedo':'albedo'
    }
    
    # Create Series from columns in merged_df
    for column, series_name in columns_to_series.items():
        series = merged_df[['mid_time', column]].copy()  # Select both columns
        #print(series['mid_time'])
        series['mid_time'] = pd.to_datetime(series['mid_time'], utc=True).dt.date  # Extract only the date (in UTC)
        series.set_index('mid_time', inplace=True)  # Set 'mid_time' as the index
        series.index = pd.to_datetime(series.index)  # Ensure datetime format without time zone (pastas will fail if time zone aware)
        series.index.freq='D'
        globals()[series_name] = series[column]  
    #print(vwcdata1.index)

    # # #print(series)
    # import matplotlib.pyplot as plt
    # # Combine all series into a single DataFrame
    # df = pd.DataFrame({'P (mm)': precip, 'T (°C)': temp, 'SoR (kW/m²)': sol, 'LAI': lai, 'RH (%)': hum, 'WS (m/s)': ws, 'SD (m)': sd, 'Albedo': albedo})

    # # Set Arial Bold as the font globally
    # plt.rcParams['font.family'] = 'Arial'
    # plt.rcParams['font.weight'] = 'bold'

    # # Increase the length of y-axes by adjusting figsize and gridspec_kw
    # fig, axes = plt.subplots(nrows=len(df.columns), sharex=True, figsize=(10, 14), gridspec_kw={'height_ratios': [1.5] * len(df.columns)})

    # for i, col in enumerate(df.columns):
    #     axes[i].plot(df.index, df[col], label=col, color='darkblue')
    #     axes[i].set_ylabel(col, fontsize=14,fontweight='bold')  # Set font size for the y-axis label
    #     axes[i].tick_params(axis='both', labelsize=14)  # Set font size for tick labels

    # # Customize the plot
    # axes[-1].set_xlabel('Date', fontsize=14,fontweight='bold')  # Set font size for the x-axis label

    # # Remove the gap between subplots
    # plt.subplots_adjust(hspace=0)

    # # Save the plot with 600 dpi
    # plt.savefig('pastas_input_1.png', dpi=600)

    # # Show the plot
    # plt.show()

    # Define the column names and corresponding series names
    columns_to_series = {
        'VWC_0.1m': 'vwcdata1',
        'VWC_0.5m': 'vwcdata2',
        'VWC_1.0m': 'vwcdata3',
        'VWC_2.0m': 'vwcdata4',
        'VWC_4.0m': 'vwcdata5',
        'VWC_6.0m': 'vwcdata6',
        'PP_6.0m': 'ppdata1'
    }
    max_trials=5
    # Iterate over the series names
    for series_name, column in columns_to_series.items():
        trials = 0
        r2 = 0
        while trials < max_trials and r2 < 0.8:
            # Create a model object by passing it the observed series
            ml = ps.Model(globals()[column], name=series_name)

            # Add the rainfall data as an explanatory variable
            sm = ps.StressModel(precip, ps.Gamma(), name="rainfall", up=True, settings="prec")  # up= True
            ml.add_stressmodel(sm)

            # Add the temperature data as an explanatory variable
            sm2 = ps.StressModel(temp, ps.Gamma(), up=False, name="temperature", settings='evap')
            ml.add_stressmodel(sm2)

            # Add the solar data as an explanatory variable
            sm3 = ps.StressModel(sol, ps.Gamma(), up=False, name="solar_radiation", settings='evap')
            ml.add_stressmodel(sm3)

            # Add the LAI data as an explanatory variable
            sm5 = ps.StressModel(lai, ps.Gamma(), up=True, name="LAI", settings='prec')
            ml.add_stressmodel(sm5)

            # Add the relative humidity data as an explanatory variable
            sm6 = ps.StressModel(hum, ps.Gamma(),up=True, name="relative_humidity", settings='prec')
            ml.add_stressmodel(sm6)

            # Add the wind speed data as an explanatory variable
            sm7= ps.StressModel(ws, ps.Gamma(),name="Wind_speed", settings='evap')
            ml.add_stressmodel(sm7)

            if( column==f'vwcdata5'):
                # Add the snow_depth data as an explanatory variable
                sm8= ps.StressModel(sd, ps.Gamma(), name="Snow_depth", settings='evap')
                ml.add_stressmodel(sm8)
            else:
                # Add the snow_depth data as an explanatory variable
                sm8= ps.StressModel(sd, ps.Gamma(), name="Snow_depth", settings='prec')
                ml.add_stressmodel(sm8)

            # Add the albedo data as an explanatory variable
            sm9= ps.StressModel(albedo, ps.Gamma(),up=False, name="albedo", settings='evap')
            ml.add_stressmodel(sm9)

            # Get tmin and tmax from the index of the series
            tmin = globals()[column].index[0]
            #tmax_solve = globals()[column].index[int(0.99*len(globals()[column]))]
            tmax_solve = globals()[column].index[-1]
            #tmax_plot = globals()[column].index[-1] + pd.Timedelta(days=days)
            tmax_plot = globals()[column].index[-1] 
            #print(tmax_solve,tmax_plot)

            # Solve the model
            ml.solve(tmin=tmin, tmax=tmax_solve)
                
            # Plot the results
            #ml.plot(tmax=tmax_plot)
            # Get the values used for plotting
            
            y_observed = ml.observations()
            y_predicted = ml.simulate(tmax=tmax_plot)  # predicted values

            # Trim y_predicted to match the length of y_observed
            y_predicted_trimmed = y_predicted[:len(y_observed)]

            # Calculate R^2 score
            r2 = r2_score(y_observed, y_predicted_trimmed)
            print('r2', series_name,':',r2)
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_observed, y_predicted_trimmed))
            trials += 1

        # # Print the RMSE value
        # print(f"RMSE_{column}:", rmse)

        # # # Print the R^2 score
        # print(f"R^2 score_{column}:", r2)

        # # # Print or use the values as needed
        
        # # #print("Observed Values:", y_observed)
        # # #print("Predicted Values:", y_predicted)

        # from matplotlib import rcParams
        # import matplotlib.pyplot as plt
        # # Creating a DataFrame for plotting
        # df_plot = pd.DataFrame({'Observed': y_observed, 'Predicted': y_predicted})
        # # Set Arial Bold as the default font for the plot
        # rcParams['font.family'] = 'Arial'
        # rcParams['font.weight'] = 'bold'
        # rcParams['font.size'] = 18


        # # Plotting
        # plt.figure(figsize=(12, 8))
        # plt.plot(df_plot.index, df_plot['Observed'], label='Measured', linewidth=2,color='black')
        # plt.plot(df_plot.index, df_plot['Predicted'], label='Modelled', linestyle='dashed', linewidth=2.5,color='brown')

        # # Customize plot
        # plt.xlabel('Date', fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        # if( column==f'ppdata1'):
        #     plt.ylabel('PWP (kPa)', fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        # else:
        #      plt.ylabel('VWC (%)', fontdict={'family': 'Arial', 'size': 24, 'weight': 'bold'})
        # #plt.xticks(rotation=45, ha='right')
        # plt.xticks(df_plot.index[::90])  # Show every 90th date
        # plt.tick_params(axis='both', which='both', direction='in', labelsize=22)

        # # Add legend
        # plt.legend(fontsize=24)
        # # Save the plot with 600 dpi
        # plt.savefig(f'pastas_{column}.png', dpi=600)

        # # # Show the plot
        # plt.show()
        y_predicted.index = pd.to_datetime(y_predicted.index, utc=True).date
        #print(forecast_df_mean_per_day)

        # Merge the DataFrames based on the 'mid_time' column
        forecast_df_mean_per_day = pd.merge(forecast_df_mean_per_day, pd.DataFrame({f"{series_name}": y_predicted}), left_on='date', right_index=True, how='left')
        #print(forecast_df_mean_per_day)
    return forecast_df_mean_per_day

# In[6]:

def FoS_Predictions(final_result,Features_FoS,blob_client_loader):
    
    
    columns_to_predict = [ 'VWC_0.1m', 'VWC_0.5m', 'VWC_1.0m', 'VWC_2.0m', 'VWC_4.0m', 'VWC_6.0m', 'PP_6.0m', 'AirTemperature', 'LAI']
    # Extract relevant columns from 'final_result'    
    data_to_predict = final_result[columns_to_predict]

    best_regressor_rf = load_model(os.environ["REGRESSOR_FOS_9_PATH"], blob_client_loader)

    # Load the scaler
    scaler_rf = load_model(os.environ["SCALER_FOS_9_PATH"], blob_client_loader)
    
    # Scale the data using the loaded scaler
    scaled_data_to_predict_rf = scaler_rf.transform(data_to_predict) 
    # Make predictions using the loaded RF regressor
    fos_predictions_rf = best_regressor_rf.predict(scaled_data_to_predict_rf)  


    best_regressor_pr = load_model(os.environ["REGRESSOR_FOS_PR_PATH"], blob_client_loader)
    # Load the scaler
    scaler_pr = load_model(os.environ["SCALER_FOS_PR_PATH"], blob_client_loader) 
    converter_pr= load_model(os.environ["POLY_CONVERT_FOS_PR_PATH"], blob_client_loader)  
    
    # Scale the data using the loaded scaler
    scaled_data_to_predict_pr = scaler_pr.transform(data_to_predict)
    #convert data for PR
    converted_data_to_predict_pr=converter_pr.transform(scaled_data_to_predict_pr)    
    # Make predictions using the loaded RF regressor
    fos_predictions_pr = best_regressor_pr.predict(converted_data_to_predict_pr)
    
    # Add 'fos_predictions' as new columns to 'final_result'
    final_result['FoS_predictions'] = fos_predictions_rf
    final_result['FoS_predictions_PR'] = fos_predictions_pr

    # Save the result to a CSV file
    #last_mid_time = final_result['mid_time'].iloc[-1].strftime('%Y-%m-%dT%H-%M-%S')
   
    #file_name = f"FoS_prediction_{last_mid_time}.csv"
    #final_result.to_csv(file_name, index=False)

    # Print the resulting dataframe
    #print(final_result)

    return final_result


def get_access_token(token_provider_url: str, client_id: str, client_secret: str):
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }
    r = requests.post(token_provider_url, data=data)
    if not r.ok:
        raise Exception(f"Token request failed with error code {r.status_code}: {r.reason}")
    response_data = r.json()
    return response_data["access_token"]


# Frost API wants times in UTC, but otherwise iso-8601
# Ref.: https://frost.met.no/concepts2.html#timespecifications
def convert_time_for_frost_api(timestamp: datetime):
    return timestamp.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def fetch_from_ngi_live(project_id, start_time, end_time, logger_name, sensor_type, secret_client):
    payload = {
        "sensor": {
            "project": project_id,
            "loggers": [logger_name],
            "sensor_type": sensor_type
        },
        "sample": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "status_code": 0,
            "resample": {
                "method": "pre_mean",
                "interval": "1 day"
            }
        },
        "process": {
        }
    }
    access_token = get_access_token(
        os.environ["NGILIVE_API_TOKEN_PROVIDER_URL"],
        secret_client.get_secret(os.environ['NGILIVE_API_CLIENT_ID_SECRET']),
        secret_client.get_secret(os.environ['NGILIVE_API_CLIENT_SECRET_SECRET'])
    )
    r = requests.post(
        f"{os.environ['NGILIVE_API_URL']}/datapoints/loggers",
        json=payload,
        headers={"Authorization": f"Bearer {access_token}"}
    )
    if not r.ok:
        raise Exception(f"Request to ngi live API failed with error code {r.status_code}: {r.reason}")

    csvStringIO = StringIO(r.text)
    return pd.read_csv(csvStringIO, sep=",")

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

    #print(df_1)
    #df_1.head()
    # Check the first row for NaN values in each column
    nan_check = df_1.iloc[0].isna()

    # Update headers where the first value is not NaN
    df_1.columns = [value if not nan_check[col] else col for col, value in zip(df_1.columns, df_1.iloc[0])]

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
        'sources': 'SN11450',
        'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D),mean(wind_speed P1D),mean(solar_irradiance PT1H),max(max(wind_speed PT1H) P1D),max(relative_humidity P1D),dew_point_temperature',
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
    #else:
        #print('Error! Returned status code %s' % r2.status_code)
        #print('Message: %s' % json['error']['message'])
        #print('Reason: %s' % json['error']['reason'])


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
    # Merge df4 with df3 based on the 'reference_time' column
    df3 = pd.merge(df3, df4[['referenceTime', 'mean(dew_point_temperature P1D)_value', 'mean(wind_speed P1D)_value']], on='referenceTime', how='left')


    # Print df3 to see the changes
    #print(df3)


    # In[17]:


    # Apply the function to calculate relative humidity and create a new column
    df3['relative_humidity'] = df3.apply(calculate_relative_humidity, axis=1)

    # Print the updated DataFrame
    #print(df3)

    # Drop the 'date' column from df5
    df5.drop(columns=['date'], inplace=True) 
    # Merge df5 with df3 based on the 'reference_time' column, and add the columns from df5 to df3
    df3 = pd.merge(df3, df5, on='referenceTime', how='left')

    # print(df3)
    # print(df_1)
    # print(end_time)  


    # In[18]:


    # merging
    merged_df = pd.merge(df_1, df3, left_on='timestamp', right_on='referenceTime', how='outer')

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

    # Assuming 'month' column is already created
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
     
    


    # In[24]:
    """
    Created on Sunday February 5 14:30:00 2023
    
    @author: EAO
    """

    def GetWeatherForecast():
        """
        This function accesses the weather forecast data from the Norwegian Meteorological Institute, and returns the required data to the user.
        Written by: Emir Ahmet Oguz, February 5 14:30:00 2023

        Returns
        -------
        Time : list
            List of times in the forecast data as string.
        Duration : list
            List of duration of the forrecast at aach time as float.
        Relative humidity: list
            List of relative humidity in the forecast data as float.
        AirTemperature : list
            List of air temperature in the forecast data as float.
        Precipitation : list
            List of precipitation in the forecast data as float.

        """

        ## Libraries
        ## https://docs.python.org/3/library/urllib.request.html#module-urllib.request
        ## https://docs.python.org/3/library/json.html
        import urllib, json
        from urllib import request, error

        '''
        ## For the following section, "https://docs.python.org/3/howto/urllib2.html" is utilized.
        '''
        ## Define url including the coordinates of the desired location
        ## Municipality of Eidsvoll, Norway (60°19′23.376", 11°14′44.646")
        # Latitude  = 60.322
        # Longitude = 11.245    ## https://www.latlong.net/  & https://www.google.com/maps/search/church+near+Eidsvoll+Municipality/@60.3236357,11.2462548,542m/data=!3m1!1e3
        # Altitude  = 170       ## https://en-gb.topographic-map.com/map-m45k/Norway/?center=60.32248%2C11.24637&zoom=16&base=6&popup=60.3223%2C11.24617
        url = "https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=60.322&lon=11.245&altitude=170"
        ## https://www.whatismybrowser.com/detect/what-is-my-user-agent/
        User_Agent = os.environ['MET_API_USER_AGENT']

        ## Define data to send to the server
        Values = {'name': 'NGI',
                  'location': 'Eidsvoll,',
                  'language': 'Python' }
        Headers = {'User-Agent': User_Agent}

        ## Generate requert
        Data = urllib.parse.urlencode(Values)
        Data = Data.encode('ascii')
        #req = request.Request(url, Data, Headers)
        headers = {'user-agent': User_Agent}
        r6 = requests.get(url, headers=headers)
        if r6.ok:
             # Extract JSON data
            Data = r6.json()
        else:
            print (r6.reason)
            return
        #print(Data)


        ## Open and read url
        # with urllib.request.urlopen(req) as response:
        #    The_page = response.read()
        '''
        '''
        ## Take the page and returns the json object
        # Data = json.loads(The_page)
        #print(Data)

        ## Allocate lists for required information: Time, air temperature (celcius), precipitation (mm) and duration (hour)
        Time             = []
        Duration         = []
        RelativeHumidity = []
        AirTemperature   = []
        Precipitation    = []
        wind_speed       = []

        ## Read required values from the data
        for Pred in Data['properties']['timeseries']:

            ## Time current prediction
            Time.append(Pred['time'])
            AirTemperature.append(Pred['data']['instant']['details']['air_temperature'])
            RelativeHumidity.append(Pred['data']['instant']['details']['relative_humidity'])
            wind_speed.append(Pred['data']['instant']['details']['wind_speed'])


            ## Next 1 hour precipitation prediction
            if('next_1_hours' in Pred['data']):
                Precipitation.append(Pred['data']['next_1_hours']['details']['precipitation_amount'])
                Duration.append(1.0)
            ## Next 6 hour precipitation prediction (take if if there is no 1-hour prediction)
            elif('next_6_hours' in Pred['data']):
                 Precipitation.append(Pred['data']['next_6_hours']['details']['precipitation_amount'])
                 Duration.append(6.0)
            ## Next 12 hour precipitation prediction (take it if there is no 1 or 6 hour prediction)
            elif('next_12_hours' in Pred['data']):
                 Precipitation.append(Pred['data']['next_12_hours']['details']['precipitation_amount'])
                 Duration.append(12.0)

        ## Return information
        return Time, Duration, RelativeHumidity, AirTemperature, Precipitation, wind_speed

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

# # for i in range(1, 4):  # Assuming 'Day_1', 'Day_2', 'Day_3' columns
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
