"""
Script to download and prepare the PlantVillage dataset.
"""
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, destination: Path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8KB blocks
    
    with open(destination, 'wb') as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

def main():
    # Create data directory
    data_dir = Path("data/plantvillage")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download dataset (using a smaller subset for faster training)
    dataset_url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    zip_path = data_dir / "plantvillage.zip"
    
    if not zip_path.exists():
        print("Downloading PlantVillage dataset...")
        download_file(dataset_url, zip_path)
    
    # Extract dataset
    extract_dir = data_dir / "raw"
    if not extract_dir.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    print("Dataset downloaded and extracted successfully!")
    print(f"Dataset location: {extract_dir.absolute()}")

if __name__ == "__main__":
    main()
