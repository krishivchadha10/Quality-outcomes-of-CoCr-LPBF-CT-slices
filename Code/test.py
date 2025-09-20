#!/usr/bin/env python3
import os
import pandas as pd

def main():
    # Adjust this if your project layout differs
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_root   = os.path.join(project_dir, 'CoCrOutput')
    csv_path    = os.path.join(data_root, 'build_logs.csv')
    images_dir  = os.path.join(data_root, 'images')
    masks_dir   = os.path.join(data_root, 'masks')

    # 1. List folders under images/ and masks/
    print("Image folders found:")
    try:
        image_samples = sorted(os.listdir(images_dir))
        print(" ", image_samples)
    except FileNotFoundError:
        print(f"  ERROR: '{images_dir}' not found")

    print("\nMask folders found:")
    try:
        mask_samples = sorted(os.listdir(masks_dir))
        print(" ", mask_samples)
    except FileNotFoundError:
        print(f"  ERROR: '{masks_dir}' not found")

    # 2. Load CSV and print sample_id column
    print("\nCSV sample_id values:")
    try:
        df = pd.read_csv(csv_path, dtype={'sample_id': str})
        df['sample_id'] = df['sample_id'].str.strip()
        print(" ", df['sample_id'].tolist())
    except FileNotFoundError:
        print(f"  ERROR: '{csv_path}' not found")

    # 3. Show the intersection
    if 'image_samples' in locals() and 'mask_samples' in locals() and 'df' in locals():
        available = set(image_samples) & set(mask_samples)
        csv_ids   = set(df['sample_id'])
        print("\nCommon folders in images/ and masks/:")
        print(" ", sorted(available))
        print("CSV vs folders intersection:")
        print(" ", sorted(csv_ids & available))
        print("\nCSV entries not in folders:")
        print(" ", sorted(csv_ids - available))
        print("Folder names not in CSV:")
        print(" ", sorted(available - csv_ids))

if __name__ == '__main__':
    main()
