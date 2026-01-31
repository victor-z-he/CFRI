#!/usr/bin/env python3
"""
Download real-time data from NOAA and USGS APIs for CFRI analysis.

This script fetches:
- Hourly water level data from NOAA CO-OPS API
- Daily mean discharge data from USGS Water Services API

Stations:
- Wilmington: NOAA 8658120 + USGS 02105769 (Cape Fear River)
- Washington: NOAA 8652587 + USGS 02084000 (Tar River)

Usage:
    python scripts/download_data.py --site wilmington
    python scripts/download_data.py --site washington
    python scripts/download_data.py --site all --start-year 1995 --end-year 2024
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from loguru import logger

# Site configurations
SITE_CONFIGS = {
    'wilmington': {
        'name': 'Wilmington-Cape Fear',
        'noaa_station': '8658120',
        'noaa_name': 'Wilmington, NC',
        'usgs_station': '02105769',
        'usgs_name': 'Cape Fear River at Lock #1 near Kelly, NC',
        'datum': 'MHHW',
        'flood_threshold_m': 0.56,
    },
    'washington': {
        'name': 'Washington-Tar River',
        'noaa_station': '8652587',
        'noaa_name': 'Washington, NC',
        'usgs_station': '02084000',
        'usgs_name': 'Tar River at Greenville, NC',
        'datum': 'MHHW',
        'flood_threshold_m': 0.70,
    }
}


def fetch_noaa_hourly(
    station_id: str,
    start_year: int,
    end_year: int,
    datum: str = 'MHHW',
    units: str = 'metric'
) -> pd.DataFrame:
    """
    Fetch hourly water level data from NOAA CO-OPS API.

    Parameters
    ----------
    station_id : str
        NOAA CO-OPS station ID (e.g., '8658120')
    start_year : int
        First year to fetch
    end_year : int
        Last year to fetch
    datum : str
        Vertical datum ('MHHW', 'MSL', 'NAVD', etc.)
    units : str
        'metric' or 'english'

    Returns
    -------
    pd.DataFrame
        Hourly water level data with datetime index
    """
    base_url = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'

    all_data = []

    for year in range(start_year, end_year + 1):
        logger.info(f"  Fetching NOAA {station_id}: {year}...")

        # NOAA API has a limit, so we fetch month by month for reliability
        for month in range(1, 13):
            # Calculate start and end dates for this month
            if month == 12:
                begin_date = f"{year}{month:02d}01"
                end_date = f"{year}{month:02d}31"
            else:
                begin_date = f"{year}{month:02d}01"
                # Get last day of month
                if month in [1, 3, 5, 7, 8, 10]:
                    end_date = f"{year}{month:02d}31"
                elif month in [4, 6, 9, 11]:
                    end_date = f"{year}{month:02d}30"
                else:  # February
                    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                        end_date = f"{year}{month:02d}29"
                    else:
                        end_date = f"{year}{month:02d}28"

            params = {
                'product': 'hourly_height',
                'station': station_id,
                'datum': datum,
                'units': units,
                'time_zone': 'gmt',
                'format': 'json',
                'begin_date': begin_date,
                'end_date': end_date,
                'application': 'python_cfri'
            }

            try:
                response = requests.get(base_url, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()

                if 'error' in data:
                    logger.warning(f"    {year}-{month:02d}: API error - {data['error'].get('message', 'Unknown')}")
                    continue

                if 'data' not in data or not data['data']:
                    continue

                # Parse records
                for record in data['data']:
                    try:
                        dt = pd.to_datetime(record['t'], format='%Y-%m-%d %H:%M')
                        val = float(record['v']) if record['v'] else None
                        all_data.append({'datetime': dt, 'water_level': val})
                    except (ValueError, KeyError):
                        continue

            except requests.exceptions.RequestException as e:
                logger.warning(f"    {year}-{month:02d}: Request failed - {e}")
                continue

            # Rate limiting
            time.sleep(0.1)

        logger.info(f"    {year}: {len([d for d in all_data if d['datetime'].year == year])} records")

    if not all_data:
        logger.error(f"No data retrieved for NOAA station {station_id}")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.drop_duplicates(subset=['datetime'])
    df = df.set_index('datetime').sort_index()

    logger.info(f"  Total NOAA records: {len(df)}")

    return df


def fetch_usgs_hourly(
    station_id: str,
    start_year: int,
    end_year: int,
    parameter_code: str = '00060'
) -> pd.DataFrame:
    """
    Fetch instantaneous (hourly) discharge data from USGS Water Services API.

    Parameters
    ----------
    station_id : str
        USGS station ID (e.g., '02105769')
    start_year : int
        First year to fetch
    end_year : int
        Last year to fetch
    parameter_code : str
        USGS parameter code ('00060' = discharge)

    Returns
    -------
    pd.DataFrame
        Hourly discharge data with datetime index
    """
    # USGS instantaneous values service
    base_url = 'https://waterservices.usgs.gov/nwis/iv/'

    all_data = []

    # Fetch in chunks (USGS has size limits)
    chunk_years = 5

    for start in range(start_year, end_year + 1, chunk_years):
        end = min(start + chunk_years - 1, end_year)

        logger.info(f"  Fetching USGS {station_id}: {start}-{end}...")

        start_date = f"{start}-01-01"
        end_date = f"{end}-12-31"

        params = {
            'format': 'json',
            'sites': station_id,
            'startDT': start_date,
            'endDT': end_date,
            'parameterCd': parameter_code,
            'siteStatus': 'all'
        }

        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()

            # Parse USGS JSON format
            if 'value' not in data or 'timeSeries' not in data['value']:
                logger.warning(f"    No time series data found")
                continue

            for ts in data['value']['timeSeries']:
                if 'values' not in ts or not ts['values']:
                    continue

                for value_set in ts['values']:
                    if 'value' not in value_set:
                        continue

                    for record in value_set['value']:
                        try:
                            dt = pd.to_datetime(record['dateTime'])
                            val = float(record['value']) if record['value'] != '-999999' else None
                            all_data.append({'datetime': dt, 'discharge_cfs': val})
                        except (ValueError, KeyError):
                            continue

            logger.info(f"    {start}-{end}: {len([d for d in all_data if start <= d['datetime'].year <= end])} records")

        except requests.exceptions.RequestException as e:
            logger.warning(f"    Request failed - {e}")
            # Try daily values as fallback
            logger.info(f"    Trying daily values instead...")
            df_daily = fetch_usgs_daily(station_id, start, end, parameter_code)
            if not df_daily.empty:
                for idx, row in df_daily.iterrows():
                    all_data.append({'datetime': idx, 'discharge_cfs': row['discharge_cfs']})

        time.sleep(0.5)

    if not all_data:
        logger.warning(f"No instantaneous data, trying daily values...")
        return fetch_usgs_daily(station_id, start_year, end_year, parameter_code)

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df = df.drop_duplicates(subset=['datetime'])
    df = df.set_index('datetime').sort_index()

    # Resample to hourly
    df = df.resample('h').mean()

    logger.info(f"  Total USGS records: {len(df)}")

    return df


def fetch_usgs_daily(
    station_id: str,
    start_year: int,
    end_year: int,
    parameter_code: str = '00060'
) -> pd.DataFrame:
    """
    Fetch daily mean discharge data from USGS Water Services API.

    This is a fallback when instantaneous values aren't available.
    """
    base_url = 'https://waterservices.usgs.gov/nwis/dv/'

    logger.info(f"  Fetching USGS daily values {station_id}: {start_year}-{end_year}...")

    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    params = {
        'format': 'json',
        'sites': station_id,
        'startDT': start_date,
        'endDT': end_date,
        'parameterCd': parameter_code,
        'statCd': '00003',  # Mean daily
        'siteStatus': 'all'
    }

    try:
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        all_data = []

        if 'value' not in data or 'timeSeries' not in data['value']:
            logger.error(f"No data found for USGS station {station_id}")
            return pd.DataFrame()

        for ts in data['value']['timeSeries']:
            if 'values' not in ts:
                continue

            for value_set in ts['values']:
                if 'value' not in value_set:
                    continue

                for record in value_set['value']:
                    try:
                        dt = pd.to_datetime(record['dateTime'])
                        val = float(record['value']) if record['value'] != '-999999' else None
                        all_data.append({'datetime': dt, 'discharge_cfs': val})
                    except (ValueError, KeyError):
                        continue

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.drop_duplicates(subset=['datetime'])
        df = df.set_index('datetime').sort_index()

        logger.info(f"  Total USGS daily records: {len(df)}")

        # Expand daily to hourly (forward fill)
        df = df.resample('h').ffill()

        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch USGS data: {e}")
        return pd.DataFrame()


def download_site_data(
    site_name: str,
    start_year: int,
    end_year: int,
    output_dir: Path
) -> Tuple[Path, Path]:
    """
    Download all data for a site.

    Parameters
    ----------
    site_name : str
        Site name ('wilmington' or 'washington')
    start_year : int
        First year to fetch
    end_year : int
        Last year to fetch
    output_dir : Path
        Output directory for CSV files

    Returns
    -------
    tuple
        Paths to (river_csv, tide_csv)
    """
    if site_name not in SITE_CONFIGS:
        raise ValueError(f"Unknown site: {site_name}. Valid: {list(SITE_CONFIGS.keys())}")

    config = SITE_CONFIGS[site_name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading data for {config['name']}")
    logger.info(f"Period: {start_year} - {end_year}")
    logger.info(f"{'='*60}")

    # Download NOAA water level data
    logger.info(f"\n--- NOAA Water Level: {config['noaa_name']} ({config['noaa_station']}) ---")
    wl_df = fetch_noaa_hourly(
        config['noaa_station'],
        start_year,
        end_year,
        datum=config['datum']
    )

    if wl_df.empty:
        logger.error("Failed to download water level data")
        return None, None

    # Save water level
    wl_path = output_dir / f"{site_name}_water_level.csv"
    wl_df.to_csv(wl_path)
    logger.info(f"Saved: {wl_path} ({len(wl_df)} records)")

    # Download USGS discharge data
    logger.info(f"\n--- USGS Discharge: {config['usgs_name']} ({config['usgs_station']}) ---")
    q_df = fetch_usgs_hourly(
        config['usgs_station'],
        start_year,
        end_year
    )

    if q_df.empty:
        logger.error("Failed to download discharge data")
        return wl_path, None

    # Convert cfs to cms (cubic meters per second)
    q_df['discharge_cms'] = q_df['discharge_cfs'] * 0.0283168

    # Save discharge
    q_path = output_dir / f"{site_name}_discharge.csv"
    q_df.to_csv(q_path)
    logger.info(f"Saved: {q_path} ({len(q_df)} records)")

    # Print summary
    logger.info(f"\n--- Summary for {config['name']} ---")
    logger.info(f"Water Level: {wl_df.index.min()} to {wl_df.index.max()}")
    logger.info(f"  Valid records: {wl_df['water_level'].notna().sum()}")
    logger.info(f"  Missing: {wl_df['water_level'].isna().sum()} ({100*wl_df['water_level'].isna().mean():.1f}%)")
    logger.info(f"Discharge: {q_df.index.min()} to {q_df.index.max()}")
    logger.info(f"  Valid records: {q_df['discharge_cfs'].notna().sum()}")
    logger.info(f"  Missing: {q_df['discharge_cfs'].isna().sum()} ({100*q_df['discharge_cfs'].isna().mean():.1f}%)")

    return q_path, wl_path


def update_config_with_real_data(
    site_name: str,
    q_path: Path,
    wl_path: Path,
    config_path: Path
) -> None:
    """
    Update YAML config file with real data file paths.
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if site_name in config['sites']:
        config['sites'][site_name]['data']['river']['file'] = str(q_path.relative_to(config_path.parent.parent))
        config['sites'][site_name]['data']['tide']['file'] = str(wl_path.relative_to(config_path.parent.parent))

        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Updated config: {config_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Download NOAA and USGS data for CFRI analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_data.py --site wilmington
    python download_data.py --site washington --start-year 2000 --end-year 2023
    python download_data.py --site all --output data/
        """
    )

    parser.add_argument(
        '--site', '-s',
        type=str,
        required=True,
        choices=['wilmington', 'washington', 'all'],
        help='Site to download data for'
    )

    parser.add_argument(
        '--start-year',
        type=int,
        default=1995,
        help='First year to download (default: 1995)'
    )

    parser.add_argument(
        '--end-year',
        type=int,
        default=datetime.now().year,
        help='Last year to download (default: current year)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data',
        help='Output directory (default: data/)'
    )

    parser.add_argument(
        '--update-config',
        action='store_true',
        help='Update YAML config with data file paths'
    )

    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Determine sites to download
    if args.site == 'all':
        sites = list(SITE_CONFIGS.keys())
    else:
        sites = [args.site]

    output_dir = Path(args.output)

    # Download data for each site
    for site in sites:
        site_output = output_dir / site
        q_path, wl_path = download_site_data(
            site,
            args.start_year,
            args.end_year,
            site_output
        )

        if q_path and wl_path and args.update_config:
            config_path = Path(__file__).parent.parent / 'configs' / 'ncsef.yaml'
            if config_path.exists():
                update_config_with_real_data(site, q_path, wl_path, config_path)

    logger.info("\n" + "="*60)
    logger.info("DATA DOWNLOAD COMPLETE")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("\nTo run CFRI analysis with this data:")
    logger.info(f"  python scripts/run_pipeline.py --config configs/ncsef.yaml --site {sites[0]}")


if __name__ == '__main__':
    main()
