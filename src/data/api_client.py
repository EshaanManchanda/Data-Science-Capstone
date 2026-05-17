"""
SpaceX Data Collection Module
Collects data from SpaceX API
"""

import requests
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class SpaceXAPIClient:
    """Client for SpaceX API data collection"""

    BASE_URL = "https://api.spacexdata.com/v4"

    def __init__(self):
        self.session = requests.Session()
        self.launches_cache = None

    def get_all_launches(self) -> pd.DataFrame:
        """Fetch all SpaceX launches"""
        if self.launches_cache is not None:
            return self.launches_cache

        logger.info("Fetching all launches from SpaceX API...")
        response = self.session.get(f"{self.BASE_URL}/launches/query", json={
            "query": {},
            "options": {
                "populate": ["rocket", "launchpad", "payloads", "cores"],
                "limit": 100
            }
        })

        if response.status_code == 200:
            data = response.json()
            launches = []
            for launch in data['docs']:
                launch_data = {
                    'flight_number': launch.get('flight_number'),
                    'mission_name': launch.get('name'),
                    'launch_date': launch.get('date_utc'),
                    'launch_site': launch.get('launchpad', {}).get('name') if launch.get('launchpad') else None,
                    'rocket': launch.get('rocket', {}).get('name') if launch.get('rocket') else None,
                    'success': launch.get('success'),
                    'upcoming': launch.get('upcoming'),
                    'details': launch.get('details')
                }
                launches.append(launch_data)

            self.launches_cache = pd.DataFrame(launches)
            logger.info(f"Fetched {len(self.launches_cache)} launches")
            return self.launches_cache
        else:
            logger.error(f"API request failed: {response.status_code}")
            return pd.DataFrame()

    def get_launch_by_id(self, flight_number: int) -> Dict:
        """Get specific launch by flight number"""
        response = self.session.get(f"{self.BASE_URL}/launches/{flight_number}")
        if response.status_code == 200:
            return response.json()
        return {}

    def get_rockets(self) -> pd.DataFrame:
        """Get all rocket information"""
        response = self.session.get(f"{self.BASE_URL}/rockets/query", json={
            "query": {},
            "options": {"limit": 10}
        })
        if response.status_code == 200:
            data = response.json()
            rockets = []
            for rocket in data['docs']:
                rockets.append({
                    'id': rocket.get('id'),
                    'name': rocket.get('name'),
                    'type': rocket.get('type'),
                    'active': rocket.get('active'),
                    'stages': rocket.get('stages'),
                    'boosters': rocket.get('boosters'),
                    'cost_per_launch': rocket.get('cost_per_launch'),
                    'success_rate': rocket.get('success_rate_pct')
                })
            return pd.DataFrame(rockets)
        return pd.DataFrame()

    def get_launchpads(self) -> pd.DataFrame:
        """Get all launch pad information"""
        response = self.session.get(f"{self.BASE_URL}/launchpads/query", json={
            "query": {},
            "options": {"limit": 10}
        })
        if response.status_code == 200:
            data = response.json()
            pads = []
            for pad in data['docs']:
                pads.append({
                    'id': pad.get('id'),
                    'name': pad.get('name'),
                    'full_name': pad.get('full_name'),
                    'locality': pad.get('locality'),
                    'region': pad.get('region'),
                    'latitude': pad.get('latitude'),
                    'longitude': pad.get('longitude'),
                    'status': pad.get('status')
                })
            return pd.DataFrame(pads)
        return pd.DataFrame()


def collect_spacex_data(output_path: str = "data/raw/spacex_api_data.csv") -> pd.DataFrame:
    """Main function to collect all SpaceX data"""
    client = SpaceXAPIClient()

    launches = client.get_all_launches()
    rockets = client.get_rockets()
    launchpads = client.get_launchpads()

    if not launches.empty:
        launches.to_csv(output_path, index=False)
        logger.info(f"Saved data to {output_path}")

    return launches


if __name__ == "__main__":
    print("Collecting SpaceX data...")
    df = collect_spacex_data()
    print(f"Collected {len(df)} launches")