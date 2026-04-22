import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import find_image, find_mask
from src.feature_area import get_lesion_area
from src.feature_shape import get_lesion_dimensions, get_perimeter, get_compactness
from src.feature_color import get_color_features


# Project root folder
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def extract_features_for_image(img_id):
    """
    Extract all features for one image.

    Parameters:
        img_id (str): image filename, e.g. "img_001.png"

    Returns:
        dict or None: feature dictionary if successful, otherwise None
    """
    img_path = find_image(img_id)
    mask_path = find_mask(img_id)

    if img_path is None or mask_path is None:
        print(f"Skipping {img_id}: image or mask not found")
        return None

    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"Skipping {img_id}: could not read image or mask")
        return None

    features = {"img_id": img_id}

    # Shape features
    features["area"] = get_lesion_area(mask, img)
    height, width = get_lesion_dimensions(mask, img)
    features["height"] = height
    features["width"] = width
    features["perimeter"] = get_perimeter(mask, img)
    features["compactness"] = get_compactness(mask, img)

    # Color features
    color_features = get_color_features(img, mask)
    features.update(color_features)

    return features


def main():
    metadata_path = DATA_DIR / "metadata.csv"
    output_path = DATA_DIR / "features.csv"

    print("Loading metadata...")
    df = pd.read_csv(metadata_path)
    print(f"Total images in metadata: {len(df)}")

    results = []
    failed = 0

    print("Extracting features...")
    for img_id in tqdm(df["img_id"]):
        features = extract_features_for_image(img_id)
        if features is not None:
            results.append(features)
        else:
            failed += 1

    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {failed}")

    features_df = pd.DataFrame(results)

    # Merge labels if available
    merge_cols = ["img_id"]
    if "diagnostic" in df.columns:
        merge_cols.append("diagnostic")
    if "patient_id" in df.columns:
        merge_cols.append("patient_id")

    features_df = features_df.merge(df[merge_cols], on="img_id", how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"Features saved to: {output_path}")
    print("Columns in features.csv:")
    print(list(features_df.columns))


if __name__ == "__main__":
    main()


