import os

def count_slices(image_root):
    sample_counts = {}
    for sample_id in sorted(os.listdir(image_root)):
        sample_path = os.path.join(image_root, sample_id)
        if os.path.isdir(sample_path):
            slice_files = [f for f in os.listdir(sample_path) if f.lower().endswith('.tif')]
            sample_counts[sample_id] = len(slice_files)
    return sample_counts

# Usage
base_dir = os.path.dirname(__file__)
image_root = os.path.join(base_dir, 'CoCrOutput', 'images')
counts = count_slices(image_root)

# Print nicely
for sample, count in counts.items():
    print(f"{sample}: {count} slices")
