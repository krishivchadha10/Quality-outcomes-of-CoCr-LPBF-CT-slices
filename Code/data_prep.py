import os, glob, zipfile, tempfile, shutil, csv
from skimage import io

# ─── UPDATE THESE PATHS ───────────────────────────────────────────────────────
DOWNLOAD_DIR = r"C:\Users\krish\OneDrive\Desktop\ML Internship\ML_LPBF_Project\CoCrData"
DATASET_ROOT = r"C:\Users\krish\OneDrive\Desktop\ML Internship\ML_LPBF_Project\Code\CoCrOutput"
# ─────────────────────────────────────────────────────────────────────────────

def extract_tiffs_from_zip(zippath, out_dir):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zippath, 'r') as zf:
            zf.extractall(tmp)
        files = sorted(glob.glob(os.path.join(tmp, '**', '*.tif'), recursive=True))
        for i, src in enumerate(files, 1):
            dst = os.path.join(out_dir, f"{i:04d}.tif")
            shutil.copy(src, dst)
    return len(files)

def prepare_dataset():
    img_root = os.path.join(DATASET_ROOT, 'images')
    msk_root = os.path.join(DATASET_ROOT, 'masks')
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(msk_root, exist_ok=True)

    raw_map, seg_map = {}, {}
    for zp in glob.glob(os.path.join(DOWNLOAD_DIR, '*.zip')):
        base = os.path.basename(zp).rsplit('.',1)[0].lower()
        if 'raw' in base:
            sample = base.replace('raw','')
            raw_map[sample] = zp
        elif 'segmented' in base:
            sample = base.replace('segmented','')
            seg_map[sample] = zp

    samples = sorted(set(raw_map) & set(seg_map))
    if not samples:
        print("❌ No matching raw/segmented zip pairs found.")
        return

    for s in samples:
        print(f"→ Processing sample '{s}'")
        img_dir = os.path.join(img_root, s)
        msk_dir = os.path.join(msk_root, s)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        n_img = extract_tiffs_from_zip(raw_map[s], img_dir)
        n_msk = extract_tiffs_from_zip(seg_map[s], msk_dir)
        print(f"   • {n_img} images, {n_msk} masks")

def verify_and_write_csv():
    out_csv = os.path.join(DATASET_ROOT, 'build_logs.csv')
    img_root = os.path.join(DATASET_ROOT, 'images')
    msk_root = os.path.join(DATASET_ROOT, 'masks')

    rows = []
    for s in sorted(os.listdir(img_root)):
        imgs = sorted(os.listdir(os.path.join(img_root, s)))
        msks = sorted(os.listdir(os.path.join(msk_root, s)))
        assert len(imgs)==len(msks), f"Mismatch in {s}"

        im = io.imread(os.path.join(img_root, s, imgs[0]))
        print(f"{s}: {len(imgs)} slices; dtype={im.dtype}; range={im.min()}–{im.max()}")

        # placeholder values
        rows.append([s, 195, 0.02, 800, 0.10])

    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id','power_W','layer_thick_mm','scan_speed_mm_s','hatch_spacing_mm'])
        w.writerows(rows)
    print(f"🗒️ Metadata written to {out_csv}")

if __name__=='__main__':
    prepare_dataset()
    verify_and_write_csv()
    print("✅ Data prep complete!")
