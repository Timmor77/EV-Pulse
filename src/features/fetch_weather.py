import logging
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_FILE = Path("data/processed/weather_data.parquet")

# Coordinates for Caltech/JPL (Pasadena, CA)
LATITUDE = 34.1377
LONGITUDE = -118.1253

# Période définie par ton audit (avec marge de sécurité)
START_DATE = "2018-09-01"
END_DATE = "2020-03-01"  # On prend une petite marge pour finir proprement février


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
        "timezone": "UTC",  # Important : On reste en UTC comme les données ACN
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
    Transforme les données Météo horaires en 15-min pour matcher ACN.
    On utilise l'interpolation linéaire pour la température (ça ne change pas brusquement)
    et le 'forward fill' pour la pluie (s'il pleut à 14h, on considère qu'il pleut à 14h15).
    """
    logger.info("Upsampling weather data from 1H to 15T...")

    # Set index
    df = df.set_index("date")

    # Resample to 15 min
    # Température et Radiation : Interpolation linéaire (douce)
    df_interp = df[["temperature", "solar_radiation"]].resample("15T").interpolate(method="linear")

    # Précipitation : Forward Fill (ou diviser par 4 si c'est du cumul,
    # mais OpenMeteo donne souvent l'intensité mm/h. Gardons ffill pour simplifier l'indicateur "il pleut")
    df_pad = df[["precipitation"]].resample("15T").ffill()

    # Merge
    final_df = pd.concat([df_interp, df_pad], axis=1).reset_index()

    # Renommer la colonne date pour le merge futur
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
