import pandas as pd
import numpy as np


def load_data(filepath):
    """Load earthquake dataset from CSV"""
    print(f"Loading data from {filepath}...")
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} records")
    return data


def select_features(data):
    """Extract core earthquake features"""
    print("\nSelecting key features...")
    data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
    print(f"Selected features: {data.columns.tolist()}")
    return data


def convert_to_timestamp(data):
    """Convert datetime strings to Unix timestamps"""
    print("\nConverting dates to timestamps...")
    
    timestamp_list = []
    errors = 0
    
    for d, t in zip(data['Date'], data['Time']):
        try:
            dt_string = f"{d} {t}"
            dt = pd.to_datetime(dt_string, format='%m/%d/%Y %H:%M:%S', errors='coerce')
            
            if pd.isna(dt):
                timestamp_list.append(None)
                errors += 1
            else:
                timestamp_list.append(dt.timestamp())
                
        except (ValueError, TypeError, OverflowError):
            timestamp_list.append(None)
            errors += 1
    
    data['Timestamp'] = timestamp_list
    initial_count = len(data)
    final_data = data.drop(['Date', 'Time'], axis=1)
    final_data = final_data.dropna(subset=['Timestamp'])
    
    print("Converted timestamps successfully")
    print(f"Removed {errors} invalid records")
    print(f"Final dataset: {len(final_data)} records (from {initial_count})")
    
    return final_data


def engineer_features(data):
    """
    Create temporal, geographic, and interaction features
    
    Temporal: year, month, day_of_year, hour, cyclical encodings
    Geographic: distance from equator, hemisphere, seismic zones
    Interaction: lat*lon, squared terms, distances from hotspots
    """
    print("\nEngineering additional features...")
    
    data['datetime'] = pd.to_datetime(data['Timestamp'], unit='s')
    
    # Temporal features
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day_of_year'] = data['datetime'].dt.dayofyear
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    
    # Cyclical encoding for periodic features
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    
    # Geographic features
    data['distance_from_equator'] = np.abs(data['Latitude'])
    data['hemisphere'] = (data['Latitude'] >= 0).astype(int)
    data['lat_lon_interaction'] = data['Latitude'] * data['Longitude']
    data['lat_squared'] = data['Latitude'] ** 2
    data['lon_squared'] = data['Longitude'] ** 2
    
    # Seismic zone indicators
    data['ring_of_fire'] = (
        ((data['Latitude'] >= -60) & (data['Latitude'] <= 70) &
         ((data['Longitude'] >= 100) & (data['Longitude'] <= 180) |
          (data['Longitude'] >= -180) & (data['Longitude'] <= -60)))
    ).astype(int)
    
    data['pacific_zone'] = (
        (data['Longitude'] >= 100) & (data['Longitude'] <= 180) |
        (data['Longitude'] >= -180) & (data['Longitude'] <= -100)
    ).astype(int)
    
    data['mediterranean_zone'] = (
        (data['Latitude'] >= 25) & (data['Latitude'] <= 50) &
        (data['Longitude'] >= -10) & (data['Longitude'] <= 45)
    ).astype(int)
    
    # Distance from major seismic hotspots
    data['dist_japan'] = np.sqrt((data['Latitude'] - 35.6)**2 + (data['Longitude'] - 139.7)**2)
    data['dist_california'] = np.sqrt((data['Latitude'] - 36.7)**2 + (data['Longitude'] + 121.6)**2)
    data['dist_chile'] = np.sqrt((data['Latitude'] + 33.4)**2 + (data['Longitude'] + 70.6)**2)
    
    data = data.drop('datetime', axis=1)
    
    feature_count = len([col for col in data.columns if col not in ['Magnitude', 'Depth']])
    print(f"Created {feature_count} features (from original 3)")
    
    return data


def prepare_train_test_split(data, test_size=0.2, random_state=42):
    """Split data into training and test sets"""
    print(f"\nPreparing train/test split (test_size={test_size})...")
    
    from sklearn.model_selection import train_test_split
    
    feature_cols = [col for col in data.columns if col not in ['Magnitude', 'Depth']]
    X = data[feature_cols]
    y = data[['Magnitude', 'Depth']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(filepath, test_size=0.2, random_state=42, use_feature_engineering=True):
    """
    Complete preprocessing pipeline
    
    Returns: X_train, X_test, y_train, y_test, original_data (for visualization)
    """
    print("="*50)
    print("STARTING ENHANCED PREPROCESSING PIPELINE")
    print("="*50)
    
    data = load_data(filepath)
    data = select_features(data)
    data = convert_to_timestamp(data)
    
    # Store original coordinates for map visualization
    original_data = data[['Latitude', 'Longitude']].copy()
    
    if use_feature_engineering:
        data = engineer_features(data)
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(data, test_size, random_state)
    
    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print("="*50)
    
    return X_train, X_test, y_train, y_test, original_data