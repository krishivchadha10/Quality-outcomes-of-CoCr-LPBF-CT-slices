#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread

def main():
    # Determine script directory and CSV path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BUILD_CSV = os.path.join(SCRIPT_DIR, 'CoCrOutput', 'build_logs.csv')

    # Load build parameters and filter samples 2–6
    build_df = pd.read_csv(BUILD_CSV)
    build_df['sample_id'] = build_df['sample_id'].astype(str)
    sample_ids = ['2', '3', '4', '5', '6']
    df = build_df[build_df['sample_id'].isin(sample_ids)]
    print(f"Filtered to samples: {', '.join(sample_ids)} -> {len(df)} rows")

    # Parameters to plot
    params = ['power_W', 'scan_speed_mm_s', 'hatch_spacing_mm', 'layer_thick_mm', 'porosity_pct']

    # 1. Bar plots in one figure: samples on x-axis, measurements on y-axis
    fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 5))
    for ax, p in zip(axes, params):
        ax.bar(df['sample_id'], df[p], edgecolor='black')
        ax.set_xlabel('Sample ID')
        ax.set_ylabel(p)
        ax.set_title(f'{p} by Sample')
    fig.suptitle('Parameters for Samples 2–6')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 2. Scatter plots: porosity vs each other parameter
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    axes2 = axes2.flatten()
    for ax, p in zip(axes2, params[:-1]):  # exclude porosity_pct itself
        ax.scatter(df[p], df['porosity_pct'], edgecolors='black')
        ax.set_xlabel(p)
        ax.set_ylabel('porosity_pct')
        ax.set_title(f'Porosity vs {p}')
    fig2.suptitle('Porosity Relationships for Samples 2–6')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3. Display a random image-mask pair
    sample = df.sample(1).iloc[0]['sample_id']
    img_path = os.path.join(SCRIPT_DIR, 'CoCrOutput', 'images', sample, '0001.tif')
    mask_path = os.path.join(SCRIPT_DIR, 'CoCrOutput', 'masks',  sample, '0001.tif')
    img = imread(img_path)
    mask = imread(mask_path)

    fig3, axes3 = plt.subplots(1, 2, figsize=(8, 4))
    axes3[0].imshow(img,  cmap='gray')
    axes3[0].set_title(f'Image {sample}')
    axes3[1].imshow(mask, cmap='gray')
    axes3[1].set_title(f'Mask {sample}')
    fig3.suptitle('Sample Image vs Mask')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 4. Folder consistency check
    images_dir = os.path.join(SCRIPT_DIR, 'CoCrOutput', 'images')
    masks_dir  = os.path.join(SCRIPT_DIR, 'CoCrOutput', 'masks')
    imgs = set(os.listdir(images_dir))
    msks = set(os.listdir(masks_dir))

    print(f"-- Folder consistency checks for samples {', '.join(sample_ids)} --")
    print("Images only:", imgs - msks)
    print("Masks only: ", msks - imgs)
    print("Missing CSV samples:", set(sample_ids) - imgs)

if __name__ == '__main__':
    main()
