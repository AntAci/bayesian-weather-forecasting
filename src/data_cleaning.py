"""
Data Cleaning Utility for Weather Forecasting Dataset

This module provides functions to clean and process raw weather data:
-Convert units (÷10 for temperatures only, PRCP left unchanged)
-Create tidy dataframe with standardized columns
-Handle data quality attributes (QFLAG)
-Output processed data to the processed directory
"""

import pandas as pd
import numpy as np
from pathlib import Path

class WeatherDataCleaner:
    """Class for cleaning and processing weather data."""
    
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        """
        Initialize the data cleaner.
        
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, filename):
        """
        Load raw weather data from CSV file.
        """
        file_path = self.raw_data_dir / filename
        
        try:
            df = pd.read_csv(file_path, parse_dates=['DATE'])
            return df
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            raise
        except Exception as e:
            print(f"Error loading file {filename}: {str(e)}")
            raise
    
    def _coerce_numeric(self, df):
        """Coerce weather columns to numeric, handling string values."""
        for col in ['TMAX', 'TMIN', 'PRCP']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _infer_prcp_units(self, s: pd.Series) -> str:
        """Infer PRCP units: 'mm' or 'tenths_mm'."""
        x = pd.to_numeric(s.dropna().sample(min(500, s.dropna().shape[0])), errors='coerce')
        if x.empty:
            return 'unknown'
        # If mostly integers and typical magnitudes up to few hundred -> likely tenths
        frac = (x % 1 != 0).mean()
        median = x.median()
        # Heuristics: your CSV shows floats like 0.02; tenths-mm would be integers 0..1000+
        if frac < 0.05 and median >= 1.0:
            return 'tenths_mm'
        return 'mm'
    
    def convert_units(self, df):
        """
        Convert weather data units by dividing by 10 if needed.
        Auto-detects if temperatures are in tenths of degrees or already in degrees.
        """
        df_converted = df.copy()
        
        # Heuristic: if temps look like tenths (e.g., 73, 102), convert; otherwise leave as-is
        def needs_div10(series):
            s = pd.to_numeric(series, errors='coerce')
            q95 = s.quantile(0.95)
            return q95 is not None and q95 > 80  # tenths-of-°C often have 90–400
        
        for col in ['TMAX', 'TMIN']:
            if col in df_converted.columns and needs_div10(df_converted[col]):
                mask = pd.to_numeric(df_converted[col], errors='coerce').notna()
                df_converted.loc[mask, col] = pd.to_numeric(df_converted.loc[mask, col]) / 10.0
        
        return df_converted
    
    def create_tidy_dataframe(self, df, include_attributes=True):
        """
        Create a tidy dataframe with standardized columns.
        """
        # Define the core columns
        core_columns = ['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE', 'TMAX', 'TMIN', 'PRCP']
        
        # Check columns 
        available_columns = [col for col in core_columns if col in df.columns]
        missing_columns = [col for col in core_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        # Create the tidy dataframe
        tidy_df = df[available_columns].copy()
        
        # Rename columns to include units
        column_mapping = {
            'TMAX': 'TMAX (°C)',
            'TMIN': 'TMIN (°C)',
            'PRCP': 'PRCP (mm)',
            'LATITUDE': 'LATITUDE',
            'LONGITUDE': 'LONGITUDE',
            'ELEVATION': 'ELEVATION'
        }
        tidy_df = tidy_df.rename(columns=column_mapping)
        
        # Parse flags correctly (MFLAG, QFLAG, SFLAG by position)
        if include_attributes:
            def split_attrs(attr):
                if pd.isna(attr): 
                    return ('','','')
                p = (attr + ',,').split(',')[:3]  # [MFLAG,QFLAG,SFLAG]
                return tuple(x.strip() for x in p)
            
            for base in ['TMAX','TMIN','PRCP']:
                col = f'{base}_ATTRIBUTES'
                if col in df.columns:
                    m,q,s = zip(*df[col].map(split_attrs))
                    tidy_df[f'{base}_MFLAG'] = m
                    tidy_df[f'{base}_QFLAG'] = q
                    tidy_df[f'{base}_SFLAG'] = s
        
        # Convert DATE to datetime
        if 'DATE' in tidy_df.columns:
            tidy_df['DATE'] = pd.to_datetime(tidy_df['DATE'])
        
        # Drop rows with missing STATION
        tidy_df = tidy_df.dropna(subset=['STATION'])
        
        # Sort by station and date
        tidy_df = tidy_df.sort_values(['STATION', 'DATE']).reset_index(drop=True)
        return tidy_df
    
    def filter_quality_data(self, df, quality_threshold=''):
        """
        Filter data based on quality flags.
        """
        if not quality_threshold:
            return df
        
        # Get quality flag columns
        qflag_columns = [col for col in df.columns if col.endswith('_QFLAG')]
        
        if not qflag_columns:
            print("Warning: No quality flag columns found for filtering")
            return df
        
        # Create filter mask
        filter_mask = pd.Series([True] * len(df))
        
        for qflag_col in qflag_columns:
            col_mask = (df[qflag_col] == '') | (df[qflag_col] == quality_threshold)
            filter_mask = filter_mask & col_mask
        
        filtered_df = df[filter_mask].copy()
        return filtered_df
    
    def advanced_processing(self, df):
        """
        Apply advanced processing: filter QFLAG, handle traces, forward-fill, and create splits.
        """
        # Filter: keep only rows with blank QFLAG (good quality data)
        qcols = [c for c in df.columns if c.endswith('_QFLAG')]
        mask = np.ones(len(df), bool)
        for c in qcols: 
            mask &= (df[c] == '')
        df_filtered = df[mask].copy()
        
        # Handle traces: set PRCP=0.0 when MFLAG=='T', keep is_trace=True
        if 'PRCP_MFLAG' in df_filtered.columns:
            df_filtered['is_trace'] = (df_filtered['PRCP_MFLAG'] == 'T')
            # Only set to 0.0 if >0 (avoid double-toggling)
            m = df_filtered['is_trace'] & (df_filtered['PRCP (mm)'] > 0)
            df_filtered.loc[m, 'PRCP (mm)'] = 0.0
        else:
            # Fallback: detect very small positive values as traces
            small_positive = (df_filtered['PRCP (mm)'] > 0) & (df_filtered['PRCP (mm)'] <= 0.05)
            df_filtered['is_trace'] = small_positive
            df_filtered.loc[df_filtered['is_trace'], 'PRCP (mm)'] = 0.0
        
        # Per-station gap handling; never forward-fill precipitation
        def ffill_small_gaps(g):
            g = g.sort_values('DATE').set_index('DATE').asfreq('D')  # create calendar days
            for col in ['TMAX (°C)','TMIN (°C)']:
                if col in g: 
                    g[col] = g[col].ffill(limit=2)  # only tiny gaps
            # PRCP (mm): do not ffill
            return g.reset_index()
        
        df_filtered = df_filtered.groupby('STATION', group_keys=False).apply(ffill_small_gaps)
        
        # Station-aware train/val/test splits using absolute date ranges
        def assign_split(g):
            g = g.sort_values('DATE').copy()
            split_train_end = pd.Timestamp('2012-12-31')
            split_val_end = pd.Timestamp('2018-12-31')
            
            g['split'] = 'train'
            g.loc[g['DATE'] > split_train_end, 'split'] = 'val'
            g.loc[g['DATE'] > split_val_end, 'split'] = 'test'
            
            return g
        
        df_filtered = df_filtered.groupby('STATION', group_keys=False).apply(assign_split)
        
        # Coverage gates: drop stations/splits failing 85% coverage rule
        weather_cols = ['TMAX (°C)', 'TMIN (°C)', 'PRCP (mm)']
        good_mask = np.ones(len(df_filtered), dtype=bool)
        
        for station_id, group in df_filtered.groupby('STATION'):
            station_name = group['NAME'].iloc[0]
            for split in ['train', 'val', 'test']:
                split_data = group[group['split'] == split]
                if len(split_data) == 0:
                    continue
                
                ok = True
                for col in weather_cols:
                    if col in split_data.columns:
                        coverage = split_data[col].notna().mean()
                        if coverage < 0.85:
                            print(f"[DROP] {station_name} {split} fails coverage for {col}: {coverage:.1%}")
                            ok = False
                
                if not ok:
                    idx = split_data.index
                    good_mask[idx] = False
        
        df_filtered = df_filtered.loc[good_mask]
        
        # Sort by station and date
        df_filtered = df_filtered.sort_values(['STATION','DATE'])
        
        # Drop empty flag columns (MFLAG and QFLAG are empty after filtering)
        empty_flag_cols = [col for col in df_filtered.columns if col.endswith(('_MFLAG', '_QFLAG'))]
        if empty_flag_cols:
            df_filtered = df_filtered.drop(columns=empty_flag_cols)
        
        return df_filtered
    
    def _assert_plausible(self, df):
        """Assert plausible value ranges for UK weather data."""
        tcols = [c for c in ['TMAX (°C)', 'TMIN (°C)'] if c in df.columns]
        for c in tcols:
            if df[c].notna().any():
                assert df[c].min() > -60 and df[c].max() < 60, f"{c} out of plausible UK bounds"
        if 'PRCP (mm)' in df.columns and df['PRCP (mm)'].notna().any():
            prcp_valid = df['PRCP (mm)'].dropna()
            assert (prcp_valid >= 0).all(), f"PRCP negative values present: {prcp_valid[prcp_valid < 0].tolist()}"
            assert prcp_valid.max() < 200, "PRCP exceeds reasonable UK daily maximum"
    
    def save_metadata(self, df):
        """
        Save metadata about stations and data coverage.
        """
        metadata = []
        
        for station_id, group in df.groupby('STATION'):
            station_name = group['NAME'].iloc[0]
            lat = group['LATITUDE'].iloc[0]
            lon = group['LONGITUDE'].iloc[0]
            elev = group['ELEVATION'].iloc[0]
            
            min_date = group['DATE'].min()
            max_date = group['DATE'].max()
            total_records = len(group)
            
            # Calculate coverage for key variables
            weather_cols = ['TMAX (°C)', 'TMIN (°C)', 'PRCP (mm)']
            coverage = {}
            for col in weather_cols:
                if col in group.columns:
                    coverage[col] = group[col].notna().mean()
            
            # Split counts and date ranges
            split_counts = group['split'].value_counts().to_dict()
            split_dates = {}
            split_coverage = {}
            
            for split in ['train', 'val', 'test']:
                split_data = group[group['split'] == split]
                if len(split_data) > 0:
                    split_dates[f'{split}_start'] = split_data['DATE'].min()
                    split_dates[f'{split}_end'] = split_data['DATE'].max()
                    split_dates[f'{split}_records'] = len(split_data)
                    
                    # Per-split coverage
                    for col in weather_cols:
                        if col in split_data.columns:
                            split_coverage[f'{split}_{col.replace(" (°C)", "").replace(" (mm)", "")}_coverage'] = split_data[col].notna().mean()
                else:
                    split_dates[f'{split}_start'] = None
                    split_dates[f'{split}_end'] = None
                    split_dates[f'{split}_records'] = 0
            
            metadata.append({
                'station_id': station_id,
                'station_name': station_name,
                'latitude': lat,
                'longitude': lon,
                'elevation': elev,
                'first_date': min_date,
                'last_date': max_date,
                'total_records': total_records,
                'tmax_coverage': coverage.get('TMAX (°C)', 0),
                'tmin_coverage': coverage.get('TMIN (°C)', 0),
                'prcp_coverage': coverage.get('PRCP (mm)', 0),
                **split_dates,
                **split_coverage
            })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_path = self.processed_data_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        return metadata_df
    
    def save_station_files(self, df):
        """
        Save individual CSV files for each station.
        """
        # Create stations directory
        stations_dir = self.processed_data_dir / "stations"
        stations_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by station and process each group
        for sid, g in df.groupby('STATION'):
            if g.empty: 
                continue
            name = g['NAME'].dropna().iloc[0] if g['NAME'].notna().any() else sid
            
            # Drop STATION and NAME columns (redundant for individual files)
            # Also drop split column as it's the same for all records in a station file
            # Drop STATION, NAME columns as they're not needed for individual station analysis
            # Keep flags for potential re-filtering
            columns_to_drop = ['STATION', 'NAME']
            if 'split' in g.columns:
                columns_to_drop.append('split')
            
            station_data = g.drop(columns=[c for c in columns_to_drop if c in g.columns])
            
            # Create filename based on station name
            if 'OXFORD' in name.upper():
                filename = 'oxford.csv'
            elif 'HAMPSTEAD' in name.upper():
                filename = 'hampstead.csv'
            elif 'BOSCOMBE' in name.upper():
                filename = 'boscombe.csv'
            elif 'WISLEY' in name.upper():
                filename = 'wisley.csv'
            else:
                # Fallback to station ID
                filename = f'{sid.lower()}.csv'
                print(f"Warning: Using station ID for filename: {filename}")
            
            # Save station data
            output_path = stations_dir / filename
            station_data.to_csv(output_path, index=False)
    
    def save_processed_data(self, df, output_filename):
        """
        Save processed data to CSV file.
        """
        output_path = self.processed_data_dir / output_filename
        
        try:
            df.to_csv(output_path, index=False)
            # Also save as Parquet for faster IO (if available)
            try:
                parquet_path = output_path.with_suffix('.parquet')
                df.to_parquet(parquet_path, index=False)
            except ImportError:
                pass
        except Exception as e:
            print(f"Error saving file {output_filename}: {str(e)}")
            raise
    
    def process_weather_file(self, 
                           input_filename, 
                           output_filename=None,
                           include_attributes=True,
                           quality_threshold='',
                           filter_quality=False):
        """
        Complete processing pipeline for a weather data file.
        """
        # Set default output filename
        if output_filename is None:
            output_filename = f"processed_{input_filename}"
        
        # Load raw data
        df = self.load_raw_data(input_filename)
        
        # Coerce numeric columns
        df = self._coerce_numeric(df)
        
        # Detect PRCP units and convert if needed
        prcp_units = 'mm'
        if 'PRCP' in df.columns:
            prcp_units = self._infer_prcp_units(df['PRCP'])
        
        # Convert units
        df_converted = self.convert_units(df)
        
        # Convert PRCP if needed
        if prcp_units == 'tenths_mm':
            df_converted['PRCP'] = df_converted['PRCP'] / 10.0
        
        # Create tidy dataframe
        tidy_df = self.create_tidy_dataframe(df_converted, include_attributes)
        
        # Apply quality filtering
        if filter_quality:
            tidy_df = self.filter_quality_data(tidy_df, quality_threshold)
        
        # Apply advanced processing
        tidy_df = self.advanced_processing(tidy_df)
        
        # Assert plausible values
        self._assert_plausible(tidy_df)
        
        # Save processed data
        self.save_processed_data(tidy_df, output_filename)
        
        # Save individual station files
        self.save_station_files(tidy_df)
        
        # Save metadata
        self.save_metadata(tidy_df)
        
        # Print summary statistics
        self._print_summary(tidy_df)
        return tidy_df
    
    def _print_summary(self, df):
        """Print summary statistics of the processed data."""
        print(f"Processed {len(df):,} records from {df['STATION'].nunique()} stations")
        print(f"Date range: {df['DATE'].min().strftime('%Y-%m-%d')} to {df['DATE'].max().strftime('%Y-%m-%d')}")
        
        # Station breakdown
        station_counts = df['STATION'].value_counts()
        for station, count in station_counts.items():
            station_name = df[df['STATION'] == station]['NAME'].iloc[0]
            print(f"{station_name}: {count:,} records")


def main():
    """Main function to demonstrate usage."""
    cleaner = WeatherDataCleaner()
    
    try:
        processed_df = cleaner.process_weather_file(
            input_filename="4127830.csv",
            output_filename="processed_weather_data.csv",
            include_attributes=True,
            filter_quality=False 
        )
        print("Processing complete!")
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
