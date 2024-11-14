import pandas as pd
import geopandas as gpd
from pathlib import Path
import shutil
from typing import Set, List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
import time


class RiverNetworkProcessor:
    """
    A class to process river networks, find upstream connections, and manage related files.
    """
    COUNTRY_PATHS = {
        'US': '/Users/yubinbaaniya/Library/CloudStorage/Box-Box/USA',
        'CA': '/Users/yubinbaaniya/Library/CloudStorage/Box-Box/Canada',
        'BR': '/Users/yubinbaaniya/Library/CloudStorage/Box-Box/Brazil',
        'FR': '/Users/yubinbaaniya/Library/CloudStorage/Box-Box/France',    #change the path for france
        'EN': '/Users/yubinbaaniya/Library/CloudStorage/Box-Box/England'    # change the path for England
    }

    def __init__(self,
                 network_file: str,
                 streams_file: str,
                 stations_file: str,
                 target_folder: str,
                 log_file: str = 'river_processing.log'):
        """
        Initialize the processor with necessary file paths.

        Args:
            network_file: Path to the network CSV file (all_704_reach_id.csv)
            streams_file: Path to the streams GeoPackage file
            stations_file: Path to the stations CSV file
            target_folder: Where to copy the found files
            log_file: Where to save processing logs
        """
        # Set up logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize file paths
        self.network_file = network_file
        self.streams_file = streams_file
        self.stations_file = stations_file
        self.target_folder = Path(target_folder)

        # Create target folder if it doesn't exist
        self.target_folder.mkdir(parents=True, exist_ok=True)

        # Load and cache the network data
        self.logger.info("Loading network data...")
        self._load_network_data()

    def _load_network_data(self):
        """Load and prepare the network data for processing."""
        try:
            # Check file extension and load accordingly
            if self.network_file.endswith('.parquet'):
                self.network_df = pd.read_parquet(
                    self.network_file,
                    columns=['LINKNO', 'DSLINKNO']
                )
            elif self.network_file.endswith('.csv'):
                self.network_df = pd.read_csv(
                    self.network_file,
                    usecols=['LINKNO', 'DSLINKNO']
                )
            else:
                raise ValueError("Unsupported file format. Use either .csv or .parquet")

            self.logger.info(f"Loaded network data with {len(self.network_df)} rows")

            # Create an upstream lookup dictionary for faster processing
            self.upstream_lookup = self.network_df.groupby('DSLINKNO')['LINKNO'].agg(list).to_dict()

        except Exception as e:
            self.logger.error(f"Error loading network data: {str(e)}")
            raise

    def find_upstream_rivers(self, river_ids: List[int]) -> Dict[int, Set[int]]:
        """
        Find all upstream rivers for multiple river IDs efficiently.

        Args:
            river_ids: List of river IDs to process

        Returns:
            Dictionary mapping each input river ID to its set of upstream river IDs
        """
        results = {}
        start_time = time.time()

        for river_id in river_ids:
            upstream_rivers = set()
            to_process = {river_id}

            while to_process:
                current_river = to_process.pop()
                # Use the lookup dictionary instead of filtering DataFrame
                upstream_matches = self.upstream_lookup.get(current_river, [])

                for upstream_river in upstream_matches:
                    if upstream_river not in upstream_rivers:
                        upstream_rivers.add(upstream_river)
                        to_process.add(upstream_river)

            results[river_id] = upstream_rivers

        processing_time = time.time() - start_time
        self.logger.info(f"Found upstream rivers for {len(river_ids)} IDs in {processing_time:.2f} seconds")

        return results

    def match_stations(self, upstream_rivers: Dict[int, Set[int]]) -> pd.DataFrame:
        """
        Match upstream rivers with station information.

        Args:
            upstream_rivers: Dictionary of river IDs and their upstream sets

        Returns:
            DataFrame with matched station information
        """
        try:
            # Load station data
            stations_df = pd.read_csv(self.stations_file)

            # Create a DataFrame from the upstream rivers
            all_upstream = [(orig_id, up_id)
                            for orig_id, up_set in upstream_rivers.items()
                            for up_id in up_set]

            upstream_df = pd.DataFrame(all_upstream, columns=['original_river_id', 'upstream_rivid'])

            # Merge with station information
            merged_df = pd.merge(
                upstream_df,
                stations_df,
                left_on='upstream_rivid',
                right_on='COMID_v2',
                how='inner'
            )

            # Filter for required conditions
            filtered_df = merged_df[merged_df['Q'] == 'YES'].copy()

            return filtered_df

        except Exception as e:
            self.logger.error(f"Error matching stations: {str(e)}")
            raise

    def copy_station_files(self, matched_stations: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Copy station files to target directory based on country.

        Args:
            matched_stations: DataFrame with matched station information

        Returns:
            Dictionary with success and failure lists
        """
        results = {
            'success': [],
            'missing': []
        }

        for _, row in matched_stations.iterrows():
            country = row['country']
            station_id = f"{row['samplingFeatureCode']}_Q.csv"

            # Get source directory based on country
            source_dir = self.COUNTRY_PATHS.get(country)
            if not source_dir:
                self.logger.warning(f"Unknown country code: {country} for station {station_id}")
                results['missing'].append(f"{station_id} (Unknown country: {country})")
                continue

            source_path = Path(source_dir) / station_id
            target_path = self.target_folder / station_id

            try:
                if source_path.exists():
                    shutil.copy(source_path, target_path)
                    results['success'].append(station_id)
                    self.logger.info(f"Copied {station_id} successfully")
                else:
                    results['missing'].append(station_id)
                    self.logger.warning(f"File not found: {station_id}")
            except Exception as e:
                self.logger.error(f"Error copying {station_id}: {str(e)}")
                results['missing'].append(f"{station_id} (Error: {str(e)})")

        return results

    def process_river_ids(self, river_ids: List[int]) -> Dict:
        """
        Main processing method to handle multiple river IDs.

        Args:
            river_ids: List of river IDs to process

        Returns:
            Dictionary with processing results
        """
        try:
            # Find upstream rivers
            self.logger.info(f"Processing {len(river_ids)} river IDs")
            upstream_rivers = self.find_upstream_rivers(river_ids)

            # Match with stations
            matched_stations = self.match_stations(upstream_rivers)
            self.logger.info(f"Found {len(matched_stations)} matching stations")

            # Copy files
            file_results = self.copy_station_files(matched_stations)

            return {
                'upstream_count': {rid: len(ups) for rid, ups in upstream_rivers.items()},
                'matched_stations': len(matched_stations),
                'files_copied': len(file_results['success']),
                'files_missing': file_results['missing']
            }

        except Exception as e:
            self.logger.error(f"Error in main processing: {str(e)}")
            raise


# Main execution
if __name__ == "__main__":
    # Initialize the processor with your specific file paths
    processor = RiverNetworkProcessor(
        network_file='/Users/yubinbaaniya/Library/CloudStorage/Box-Box/master thesis and what not/Geoglows AWS files except VPU/v2-master-table.parquet',
        streams_file='/Users/yubinbaaniya/Library/CloudStorage/Box-Box/VPU/streams_714.gpkg',
        stations_file='/Users/yubinbaaniya/Library/CloudStorage/Box-Box/Jorge dessertation/2ND ITERATION CSV FILES/all_station climate_strmOrder_DSarea.csv',
        target_folder='/Users/yubinbaaniya/Documents/subdaily'
    )

    # Define the river IDs you want to process
    river_ids = [760754696]  # Add more IDs to this list if needed

    # Process the river IDs
    results = processor.process_river_ids(river_ids)

    # Print results
    print("\nProcessing Results:")
    print("==================")
    print(f"Processed {len(river_ids)} river IDs")
    print(f"\nUpstream rivers found for each ID:")
    for rid, count in results['upstream_count'].items():
        print(f"  River ID {rid}: {count} upstream rivers")
    print(f"\nMatched stations: {results['matched_stations']}")
    print(f"Files successfully copied: {results['files_copied']}")
    print(f"Files missing: {len(results['files_missing'])}")

    if results['files_missing']:
        print("\nMissing files:")
        for missing in results['files_missing']:
            print(f"  - {missing}")