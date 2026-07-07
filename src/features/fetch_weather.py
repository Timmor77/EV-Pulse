"""Weather data fetching module using Open-Meteo API.

This module fetches historical weather data for the Pasadena, CA area
(Caltech/JPL location) and upsamples it to match the 15-minute resolution
of the charging data time series.
"""

import logging
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FILE = Path("data/processed/weather_data.parquet")

# Coordinates for Caltech/JPL (Pasadena, CA)
LATITUDE = 34.1377
LONGITUDE = -118.1253

# Period defined by your audit (with safety margin)
START_DATE = "2018-09-01"
END_DATE = "2020-03-01"  # Small margin to properly finish February


def fetch_weather_data():
    # Setup the Open-Meteo client with cache and retry on error
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    logger.info(f"Fetching weather data for Pasadena ({START_DATE} to {END_DATE})...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ["temperature_2m", "precipitation", "direct_radiation"],
        "timezone": "UTC",  # Keep UTC to match ACN data timezone
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]  # Single location

    # Process hourly data
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_radiation = hourly.Variables(2).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    hourly_data["temperature"] = hourly_temperature_2m
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["solar_radiation"] = hourly_radiation

    weather_df = pd.DataFrame(data=hourly_data)

    return weather_df


def process_and_upsample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform hourly weather data to 15-min intervals to match ACN data.
    Use linear interpolation for temperature (changes gradually)
    and forward fill for precipitation (if raining at 2pm, still raining at 2:15pm).
    """
    logger.info("Upsampling weather data from 1h to 15min...")

    # Set index
    df = df.set_index("date")

    # Resample to 15 min
    # Temperature and Radiation: Linear interpolation (smooth)
    df_interp = df[["temperature", "solar_radiation"]].resample("15min").interpolate(method="linear")

    # Precipitation: Forward Fill (or divide by 4 if cumulative,
    # but OpenMeteo often gives intensity mm/h. Keep ffill for simplicity as "it's raining" indicator)
    df_pad = df[["precipitation"]].resample("15min").ffill()

    # Merge
    final_df = pd.concat([df_interp, df_pad], axis=1).reset_index()

    # Rename date column for future merge
    final_df = final_df.rename(columns={"date": "datetime"})

    return final_df


def main():
    try:
        df_hourly = fetch_weather_data()
        df_15min = process_and_upsample(df_hourly)

        logger.info(f"Saving weather data to {OUTPUT_FILE}...")
        df_15min.to_parquet(OUTPUT_FILE, index=False)

        print("--- 🌦️ Weather Data Sample ---")
        print(df_15min.head())
        print(df_15min.describe())

    except Exception as e:
        logger.error(f"Error fetching weather: {e}")


if __name__ == "__main__":
    main()
