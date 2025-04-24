from pathlib import Path
import pandas as pd
import re

def parse_image_filenames(input_folder):
    """
    Parses structured microscopy .dv filenames from a given folder and returns a DataFrame
    with extracted metadata.
    
    Parameters:
        input_folder (str or Path): Path to the folder containing .dv files
    
    Returns:
        pd.DataFrame:
            DataFrame containing extracted metadata from filenames
    """
    input_path = Path(input_folder)
    filenames = list(input_path.glob("*.dv"))

    # Regex pattern to extract components
    pattern = re.compile(
        r'(?P<condition>[a-zA-Z0-9]+)_'
        r'(?P<donor>BEC\d+)_'
        r'(?P<time>\d+h)_'
        r'(?P<date>\d{2}\.\d{2}\.\d{2})'
        r'(?:\.(?P<replicate>\d+))?_'
        r'(?P<mode1>[A-Z0-9]+)_'
        r'(?P<mode2>[A-Z0-9]+)\.dv$'
    )

    records = []
    for file in filenames:
        match = pattern.match(file.name)
        if match:
            data = match.groupdict()
            data["filename"] = file.name
            data["filepath"] = str(file.resolve()) # Full path for loading
            records.append(data)

    # Create DataFrame
    df = pd.DataFrame(records)

    if not df.empty:
        df['date'] = pd.to_datetime(df['date'], format="%d.%m.%y")

    return df
