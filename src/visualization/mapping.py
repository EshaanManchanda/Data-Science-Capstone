"""
SpaceX Launch Site Mapping Module
Interactive maps using Folium
"""

import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, MousePosition
from folium.features import DivIcon
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class SpaceXMapper:
    """Create interactive maps for SpaceX launch sites"""

    LAUNCH_SITE_COORDS = {
        'CCAFS LC-40': (28.56230, -80.57728),
        'CCAFS SLC-40': (28.56230, -80.57728),
        'KSC LC-39A': (28.57320, -80.64690),
        'KSC LC-39A': (28.57320, -80.64690),
        'VAFB SLC-4E': (34.74220, -120.57280),
        'VAFB SLC-4E': (34.74220, -120.57280),
    }

    def __init__(self, center: Tuple[float, float] = (29.559684888503615, -95.0830971930759)):
        self.center = center
        self.map = None

    def create_base_map(self, zoom: int = 5) -> folium.Map:
        """Create base map centered on NASA Johnson Space Center"""
        self.map = folium.Map(location=self.center, zoom_start=zoom)
        return self.map

    def add_launch_sites(self, df: pd.DataFrame) -> folium.Map:
        """Add launch site markers to map"""
        if self.map is None:
            self.create_base_map()

        unique_sites = df['Launch Site'].unique() if 'Launch Site' in df.columns else []

        for site in unique_sites:
            coords = self.LAUNCH_SITE_COORDS.get(site)
            if coords:
                folium.Circle(
                    location=coords,
                    radius=1000,
                    color='#d35400',
                    fill=True,
                    popup=site
                ).add_to(self.map)

                folium.Marker(
                    location=coords,
                    icon=DivIcon(
                        icon_size=(20, 20),
                        icon_anchor=(0, 0),
                        html=f'<div style="font-size: 12px; color: #d35400;"><b>{site}</b></div>'
                    )
                ).add_to(self.map)

        logger.info(f"Added {len(unique_sites)} launch sites to map")
        return self.map

    def add_launch_markers(self, df: pd.DataFrame) -> folium.Map:
        """Add launch outcome markers (green=success, red=failure)"""
        if self.map is None:
            self.create_base_map()

        if 'Launch Site' not in df.columns or 'class' not in df.columns:
            logger.warning("Required columns not found")
            return self.map

        marker_cluster = MarkerCluster().add_to(self.map)

        df['marker_color'] = df['class'].apply(lambda x: 'green' if x == 1 else 'red')

        for _, row in df.iterrows():
            site = row.get('Launch Site')
            coords = self.LAUNCH_SITE_COORDS.get(site) if site else None

            if coords:
                folium.Marker(
                    location=coords,
                    icon=folium.Icon(
                        color='white',
                        icon_color=row['marker_color']
                    ),
                    popup=f"Flight: {row.get('Flight Number', 'N/A')}<br>Outcome: {'Success' if row.get('class') == 1 else 'Failure'}"
                ).add_to(marker_cluster)

        logger.info(f"Added {len(df)} launch markers")
        return self.map

    def add_mouse_position(self) -> folium.Map:
        """Add mouse position plugin to get coordinates"""
        if self.map is None:
            self.create_base_map()

        formatter = "function(num) {return L.Util.formatNum(num, 5);};"
        mouse_position = MousePosition(
            position='topright',
            separator=' Long: ',
            empty_string='NaN',
            lng_first=False,
            num_digits=20,
            prefix='Lat:',
            lat_formatter=formatter,
            lng_formatter=formatter
        )
        mouse_position.add_to(self.map)
        return self.map

    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates using Haversine formula"""
        from math import sin, cos, sqrt, atan2, radians

        R = 6373.0

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def add_distance_marker(self, launch_site: str, point_name: str,
                           point_coords: Tuple[float, float]) -> folium.Map:
        """Add distance marker between launch site and a point"""
        if self.map is None:
            self.create_base_map()

        site_coords = self.LAUNCH_SITE_COORDS.get(launch_site)
        if not site_coords:
            logger.warning(f"Coordinates not found for {launch_site}")
            return self.map

        distance = self.calculate_distance(
            site_coords[0], site_coords[1],
            point_coords[0], point_coords[1]
        )

        folium.Marker(
            location=point_coords,
            icon=DivIcon(
                icon_size=(20, 20),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 12px;"><b>{point_name}</b><br>{distance:.2f} km</div>'
            )
        ).add_to(self.map)

        folium.PolyLine(
            locations=[site_coords, point_coords],
            weight=2,
            color='blue'
        ).add_to(self.map)

        logger.info(f"Added {point_name} at {distance:.2f} km from {launch_site}")
        return self.map

    def add_proximity_markers(self, launch_site: str) -> folium.Map:
        """Add common proximity markers (coastline, city, highway, railway)"""
        site_coords = self.LAUNCH_SITE_COORDS.get(launch_site)
        if not site_coords:
            return self.map

        proximities = {
            'Coastline': (site_coords[0] + 0.001, site_coords[1] + 0.008),
            'City (Cocoa)': (28.3200, -80.6077),
            'Highway': (site_coords[0] - 0.0003, site_coords[1] + 0.0095),
            'Railway': (site_coords[0] + 0.0098, site_coords[1] - 0.0075)
        }

        for name, coords in proximities.items():
            self.add_distance_marker(launch_site, name, coords)

        return self.map

    def save_map(self, filename: str = "spacex_launch_map.html"):
        """Save map to HTML file"""
        if self.map:
            self.map.save(filename)
            logger.info(f"Map saved to {filename}")

    def get_map(self) -> folium.Map:
        """Return the map object"""
        return self.map


def create_launch_map(df: pd.DataFrame, output: str = "spacex_map.html") -> folium.Map:
    """Main function to create launch site map"""
    mapper = SpaceXMapper()
    mapper.create_base_map()
    mapper.add_launch_sites(df)
    mapper.add_launch_markers(df)
    mapper.add_mouse_position()

    if 'Launch Site' in df.columns and len(df) > 0:
        mapper.add_proximity_markers(df['Launch Site'].iloc[0])

    mapper.save_map(output)
    return mapper.get_map()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        create_launch_map(df)
    else:
        print("Usage: python mapping.py <data_file.csv>")