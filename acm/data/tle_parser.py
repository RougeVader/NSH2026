import numpy as np
from sgp4.api import WGS72, Satrec, jday
from datetime import datetime
import os

def parse_tle_file(filepath: str):
    """
    Reads a TLE file and parses each TLE entry into a dictionary.
    Each entry will have 'name', 'line1', and 'line2'.
    Assumes a 3-line format (name, line1, line2).
    """
    tles = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    print(f"Total lines in {filepath}: {len(lines)}")
    # Iterate in chunks of 3 (name, line1, line2)
    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            name_line = lines[i].strip() # Still strip name line, but not TLE lines
            line1 = lines[i+1].rstrip()  # rstrip() to remove only trailing newline, preserve trailing spaces
            line2 = lines[i+2].rstrip()  # rstrip() to remove only trailing newline, preserve trailing spaces

            # Basic validation for TLE lines (check for '1 ' and '2 ' at start and length 69)
            if (line1.startswith('1 ') and len(line1) == 69 and
                line2.startswith('2 ') and len(line2) == 69):
                tles.append({
                    'name': name_line,
                    'line1': line1,
                    'line2': line2
                })
                print(f"  --> Valid TLE for '{name_line}' added.")
            else:
                print(f"  --> Skipping malformed TLE block for '{name_line}'.")
        else:
            print(f"Skipping incomplete TLE block starting at line {i+1} in {filepath}.")
    return tles

def tles_to_state_vectors(tle_data, timestamp: float):
    """
    Converts a list of parsed TLE data into ECI state vectors (position and velocity).
    timestamp: Unix timestamp (seconds since epoch) at which to calculate the state.
    """
    debris_objects = {}
    
    # Convert Unix timestamp to Julian date
    dt_utc = datetime.utcfromtimestamp(timestamp)
    jd, fr = jday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond / 1e6)

    # Use a counter to ensure unique names for debris objects with identical names
    name_counts = {}

    for tle_entry in tle_data:
        original_name = tle_entry['name']
        name_counts[original_name] = name_counts.get(original_name, 0) + 1
        # Use NORAD ID if possible, otherwise use original_name-count
        norad_id = tle_entry['line1'][2:7].strip() # NORAD CATNR is characters 3-7 of line 1
        unique_id = f"DEB-{norad_id}-{name_counts[original_name]}" if name_counts[original_name] > 1 else f"DEB-{norad_id}"

        try:
            satellite = Satrec.twoline2rv(tle_entry['line1'], tle_entry['line2'])
            
            # Propagate to the given timestamp
            error, r, v = satellite.sgp4(jd, fr)

            if error == 0:
                # r (position) and v (velocity) are in km and km/s in TEME frame
                # For our purposes, TEME is close enough to ECI for initial population.
                debris_objects[unique_id] = np.array([r[0], r[1], r[2], v[0], v[1], v[2]])
            else:
                print(f"SGP4 error for {unique_id}: {error}")
        except Exception as e:
            print(f"Error processing TLE for {unique_id}: {e}")
            
    return debris_objects

def load_and_parse_debris_tles(directory: str, timestamp: float):
    """
    Loads all TLE files from a directory, parses them, and returns ECI state vectors.
    """
    all_tle_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and ('debris' in filename.lower() or 'tle' in filename.lower()):
            filepath = os.path.join(directory, filename)
            print(f"Parsing TLE file: {filepath}")
            all_tle_data.extend(parse_tle_file(filepath))
            
    return tles_to_state_vectors(all_tle_data, timestamp)
