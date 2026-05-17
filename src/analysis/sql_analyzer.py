"""
SpaceX SQL Analysis Module
SQL queries and analysis for SpaceX data
"""

import pandas as pd
import sqlite3
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class SpaceXSQLAnalyzer:
    """SQL-based analysis for SpaceX data"""

    def __init__(self, db_name: str = "spacex.db"):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        """Create database connection"""
        self.conn = sqlite3.connect(self.db_name)
        logger.info(f"Connected to {self.db_name}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def load_data_to_sql(self, df: pd.DataFrame, table_name: str = "SPACEXTABLE"):
        """Load pandas dataframe to SQL table"""
        if self.conn is None:
            self.connect()

        df.to_sql(table_name, self.conn, if_exists='replace', index=False)
        logger.info(f"Loaded {len(df)} rows to {table_name}")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results"""
        if self.conn is None:
            self.connect()

        result = pd.read_sql(query, self.conn)
        return result

    def get_unique_launch_sites(self) -> List[str]:
        """Get unique launch sites"""
        query = "SELECT DISTINCT Launch_Site FROM SPACEXTABLE"
        result = self.execute_query(query)
        return result['Launch_Site'].tolist()

    def get_launches_by_site(self, site_prefix: str = "CCA%", limit: int = 5) -> pd.DataFrame:
        """Get launches by site prefix"""
        query = f"""
        SELECT * FROM SPACEXTABLE
        WHERE Launch_Site LIKE '{site_prefix}'
        LIMIT {limit}
        """
        return self.execute_query(query)

    def get_total_payload_mass(self, customer: str = "NASA (CRS)") -> float:
        """Get total payload mass for a customer"""
        query = f"""
        SELECT SUM(PAYLOAD_MASS__KG_) as Total_Mass
        FROM SPACEXTABLE
        WHERE Customer = '{customer}'
        """
        result = self.execute_query(query)
        return result['Total_Mass'].iloc[0] if not result.empty else 0

    def get_avg_payload_mass(self, booster_version: str) -> float:
        """Get average payload mass for a booster version"""
        query = f"""
        SELECT AVG(PAYLOAD_MASS__KG_) as Avg_Mass
        FROM SPACEXTABLE
        WHERE Booster_Version = '{booster_version}'
        """
        result = self.execute_query(query)
        return result['Avg_Mass'].iloc[0] if not result.empty else 0

    def get_first_successful_ground_landing(self) -> str:
        """Get first successful ground pad landing date"""
        query = """
        SELECT MIN(Date) as First_Success
        FROM SPACEXTABLE
        WHERE Landing_Outcome = 'Success (ground pad)'
        """
        result = self.execute_query(query)
        return result['First_Success'].iloc[0] if not result.empty else None

    def get_boosters_with_success_drone_ship(self, min_payload: int = 4000, max_payload: int = 6000) -> List[str]:
        """Get boosters with successful drone ship landing and specific payload range"""
        query = f"""
        SELECT DISTINCT Booster_Version
        FROM SPACEXTABLE
        WHERE Landing_Outcome = 'Success (drone ship)'
        AND PAYLOAD_MASS__KG_ > {min_payload}
        AND PAYLOAD_MASS__KG_ < {max_payload}
        """
        result = self.execute_query(query)
        return result['Booster_Version'].tolist() if not result.empty else []

    def get_mission_outcome_counts(self) -> pd.DataFrame:
        """Get count of each mission outcome"""
        # Check available columns first
        try:
            cols_query = "PRAGMA table_info(SPACEXTABLE)"
            cols_result = self.execute_query(cols_query)
            columns = cols_result['name'].tolist() if not cols_result.empty else []
        except:
            columns = []
        
        # Use Class column if Mission_Outcome doesn't exist
        if 'Mission_Outcome' in columns:
            query = """
            SELECT Mission_Outcome, COUNT(*) as Count
            FROM SPACEXTABLE
            GROUP BY Mission_Outcome
            """
            return self.execute_query(query)
        elif 'Class' in columns:
            query = """
            SELECT Class, COUNT(*) as Count
            FROM SPACEXTABLE
            GROUP BY Class
            """
            result = self.execute_query(query)
            result['Mission_Outcome'] = result['Class'].apply(lambda x: 'Success' if x == 1 else 'Failure')
            return result[['Mission_Outcome', 'Count']]
        else:
            return pd.DataFrame()

    def get_max_payload_boosters(self) -> List[str]:
        """Get boosters that carried maximum payload mass"""
        query = """
        SELECT DISTINCT Booster_Version
        FROM SPACEXTABLE
        WHERE PAYLOAD_MASS__KG_ = (SELECT MAX(PAYLOAD_MASS__KG_) FROM SPACEXTABLE)
        """
        result = self.execute_query(query)
        return result['Booster_Version'].tolist() if not result.empty else []

    def get_2015_failure_drone_ship(self) -> pd.DataFrame:
        """Get failure outcomes on drone ship in 2015"""
        query = """
        SELECT substr(Date, 6, 2) as Month, Landing_Outcome, Booster_Version, Launch_Site
        FROM SPACEXTABLE
        WHERE Landing_Outcome = 'Failure (drone ship)'
        AND substr(Date, 1, 4) = '2015'
        """
        return self.execute_query(query)

    def get_landing_outcome_counts_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get landing outcome counts in date range"""
        query = f"""
        SELECT Landing_Outcome, COUNT(*) as Count
        FROM SPACEXTABLE
        WHERE Date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY Landing_Outcome
        ORDER BY Count DESC
        """
        return self.execute_query(query)

    def run_all_queries(self) -> Dict[str, Any]:
        """Run all standard analysis queries"""
        results = {}

        results['unique_launch_sites'] = self.get_unique_launch_sites()
        results['cca_launches'] = self.get_launches_by_site().to_dict()

        try:
            results['nasa_crs_payload'] = self.get_total_payload_mass()
        except:
            results['nasa_crs_payload'] = 0

        results['mission_outcomes'] = self.get_mission_outcome_counts().to_dict()

        return results


def analyze_with_sql(df: pd.DataFrame) -> Dict[str, Any]:
    """Main function to run SQL-based analysis"""
    analyzer = SpaceXSQLAnalyzer()

    analyzer.load_data_to_sql(df)

    results = analyzer.run_all_queries()

    analyzer.close()
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        df = pd.read_csv(sys.argv[1])
        results = analyze_with_sql(df)
        print(results)
    else:
        print("Usage: python sql_analysis.py <data_file.csv>")