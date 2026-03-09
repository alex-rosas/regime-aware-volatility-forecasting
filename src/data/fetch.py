# src/data/fetch.py
import os
import requests
import pandas as pd
from fredapi import Fred
from pathlib import Path
from dotenv import load_dotenv
from datetime import date

load_dotenv()

TODAY = date.today().strftime("%Y-%m-%d")

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")
FRED_API_KEY = os.getenv("FRED_API_KEY")


def fetch_mxn_usd(start:str="2000-01-01", end:str=TODAY) -> pd.DataFrame:
    # Banxico SIE REST API
    # Series: SF43718 (tipo de cambio FIX)
    series = "SF43718"
    
    # URL pattern:
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series}/datos/{start}/{end}"
    
    # Header: {"Bmx-Token": BANXICO_TOKEN}
    headers = {"Bmx-Token": BANXICO_TOKEN}

    # The response is JSON — explore its structure to find where the data lives
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    raw = response.json()

    # Parse dates and values into a DataFrame
    data = pd.DataFrame(raw["bmx"]["series"][0]["datos"])
    data["fecha"] = pd.to_datetime(data["fecha"], format="%d/%m/%Y")
    data["dato"] = pd.to_numeric(data["dato"], errors="coerce")
    data = data[["fecha", "dato"]].rename(
        columns={"fecha": "Date", "dato": "MXN_USD"}
        )
    
    # Set the date as index
    data = data.set_index("Date")
    
    # Save to data/raw/mxn_usd.csv
    data.to_csv(RAW_DIR / "mxn_usd.csv")

    # Return the DataFrame
    return data


def fetch_ipc(start: str = "2000-01-01", end: str = TODAY) -> pd.DataFrame:
    # Yahoo Finance direct HTTP request — yfinance has a bug with ^MXX ticker
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EMXX"
    params = {
        "interval": "1d",
        "range": "25y"
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    raw = response.json()

    # Parse timestamps and closing prices from JSON response
    timestamps = raw["chart"]["result"][0]["timestamp"]
    closes = raw["chart"]["result"][0]["indicators"]["quote"][0]["close"]

    # Build DataFrame
    df = pd.DataFrame({"IPC": closes},
                      index=pd.to_datetime(timestamps, unit="s"))
    df.index.name = "Date"
    df.index = df.index.normalize()  # remove time component

    # Filter by start/end dates for consistency with other functions
    df = df[df.index >= start]
    df = df[df.index <= end]
    df = df.ffill()  # forward-fill missing values (e.g. weekends) for validation purposes

    # Save to data/raw/ipc.csv
    df.to_csv(RAW_DIR / "ipc.csv")
    return df


def fetch_macro_indicators(start:str="2000-01-01", end:str=TODAY) -> pd.DataFrame:
    # FRED API via fredapi
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    # Series: VIXCLS, DFF, T10Y2Y
    series_ids = ["VIXCLS", "DFF", "T10Y2Y"]

    data = pd.DataFrame({ 
        serie_id: fred.get_series(
            serie_id, observation_start=start, observation_end=end
            )
        for serie_id in series_ids
    })
    # Save to data/raw/macro.csv
    data.to_csv(RAW_DIR / "macro.csv")
    # Return the DataFrame
    return data


if __name__ == "__main__":
    print("Fetching MXN/USD...")
    mxn = fetch_mxn_usd()
    print(f"  shape: {mxn.shape}, date range: {mxn.index[0]} to {mxn.index[-1]}")

    print("Fetching IPC...")
    ipc = fetch_ipc()
    print(f"  shape: {ipc.shape}, date range: {ipc.index[0]} to {ipc.index[-1]}")

    print("Fetching macro indicators...")
    macro = fetch_macro_indicators()
    print(f"  shape: {macro.shape}, date range: {macro.index[0]} to {macro.index[-1]}")

    print("Done. Check data/raw/ for CSV files.")