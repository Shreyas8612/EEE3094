import numpy as np
import os
import cv2
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from PIL import Image
from Vessel_Isolation import process_fundus_image

warnings.filterwarnings('ignore')


def load_image(file_path):
    try:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.gif':
            return np.array(Image.open(file_path))
        elif file_ext == '.png':
            return np.array(Image.open(file_path))
        elif file_ext in ['.jpg', '.jpeg']:
            img = cv2.imread(file_path)
            if img is not None and len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            # For other image formats (.tif)
            return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


def calculate_metrics(pred, gt, mask=None):
    # Ensure binary images
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    # Apply mask if provided
    if mask is not None:
        pred = pred * mask.astype(np.uint8)
        gt = gt * mask.astype(np.uint8)

    # Calculate TP, FP, TN, FN
    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    TN = np.sum((pred == 0) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    # Handle division by zero
    eps = 1e-100

    # Calculate metrics (only accuracy, sensitivity, specificity)
    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def process_single_image(args):
    img_path, gt_path = args

    # Time the processing
    start_time = time.time()
    results = process_fundus_image(img_path)
    processing_time = time.time() - start_time

    # Get a binary segmentation result
    segmentation = results['masked_erode_image'].astype(np.uint8)

    # Load and process ground truth image
    if isinstance(gt_path, list):
        gt_path = gt_path[0]  # Use the first ground truth if multiple is provided

    gt = load_image(gt_path)
    if gt is None:
        return None

    # Ensure ground truth is binary
    gt = (gt > 0).astype(np.uint8)

    # Resize ground truth to match segmentation if needed
    if gt.shape != segmentation.shape:
        gt = cv2.resize(gt, (segmentation.shape[1], segmentation.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    # Extract fundus mask from the results if available
    mask = None
    if 'fundus_mask' in results:
        mask = results['fundus_mask']

    # Calculate metrics
    metrics = calculate_metrics(segmentation, gt, mask)
    metrics['processing_time'] = processing_time

    return metrics


def process_dataset(image_paths, gt_paths, num_workers=None):
    # Track total processing time
    dataset_start_time = time.time()

    # Prepare arguments for parallel processing
    process_args = []
    for i, img_path in enumerate(image_paths):
        process_args.append((img_path, gt_paths[i]))

    # Process images in parallel
    all_metrics = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_image = {executor.submit(process_single_image, arg): arg for arg in process_args}
        for future in as_completed(future_to_image):
            result = future.result()
            if result:
                all_metrics.append(result)

    # Calculate dataset-level statistics
    if not all_metrics:
        print("Warning: No valid results were collected for the dataset")
        return None, 0, 0

    # Compute averages
    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_sensitivity = np.mean([m['sensitivity'] for m in all_metrics])
    avg_specificity = np.mean([m['specificity'] for m in all_metrics])
    avg_processing_time = np.mean([m['processing_time'] for m in all_metrics])

    # Calculate std deviations
    std_accuracy = np.std([m['accuracy'] for m in all_metrics])
    std_sensitivity = np.std([m['sensitivity'] for m in all_metrics])
    std_specificity = np.std([m['specificity'] for m in all_metrics])
    std_processing_time = np.std([m['processing_time'] for m in all_metrics])

    # Calculate total processing time
    dataset_processing_time = time.time() - dataset_start_time

    result_metrics = {
        'accuracy_mean': avg_accuracy,
        'accuracy_std': std_accuracy,
        'sensitivity_mean': avg_sensitivity,
        'sensitivity_std': std_sensitivity,
        'specificity_mean': avg_specificity,
        'specificity_std': std_specificity,
        'processing_time_mean': avg_processing_time,
        'processing_time_std': std_processing_time,
        'total_processing_time': dataset_processing_time,
        'num_images': len(all_metrics)
    }

    return result_metrics, dataset_processing_time, avg_processing_time


def load_drive_dataset(base_path):
    valid_image_paths = []
    gt_paths = []

    # Process Subset 1
    subset1_dir = os.path.join(base_path, 'Subset 1')
    subset1_img_dir = os.path.join(subset1_dir, 'images')
    subset1_manual2_dir = os.path.join(subset1_dir, '2nd_manual')

    # Get all images in Subset 1
    subset1_images = sorted(glob.glob(os.path.join(subset1_img_dir, '*.tif')))

    print(f"    Found {len(subset1_images)} images in DRIVE Subset 1")

    # Match each image with its ground truths based-on-number prefix
    for img_path in subset1_images:
        img_name = os.path.basename(img_path)
        # Extract prefix
        prefix = img_name.split('_')[0]

        # Construct ground truth paths with the appropriate naming pattern
        manual2_path = os.path.join(subset1_manual2_dir, f"{prefix}_manual2.gif")
        if os.path.exists(manual2_path):
            valid_image_paths.append(img_path)
            gt_paths.append([manual2_path])
        else:
            print(f"    Warning: Missing ground truth for {img_name} in Subset 1")
            print(f"      Missing: {manual2_path}")


    # Process Subset 2 (with one manual segmentation)
    subset2_dir = os.path.join(base_path, 'Subset 2')
    subset2_img_dir = os.path.join(subset2_dir, 'images')
    subset2_manual1_dir = os.path.join(subset2_dir, '1st_manual')

    # Get all images in Subset 2
    subset2_images = sorted(glob.glob(os.path.join(subset2_img_dir, '*.tif')))

    print(f"    Found {len(subset2_images)} images in DRIVE Subset 2")

    # Match each image with its ground truth
    for img_path in subset2_images:
        img_name = os.path.basename(img_path)
        prefix = img_name.split('_')[0]

        # Construct ground truth path with the appropriate naming pattern
        manual1_path = os.path.join(subset2_manual1_dir, f"{prefix}_manual1.gif")

        # Check if a ground truth file exists
        if os.path.exists(manual1_path):
            valid_image_paths.append(img_path)
            gt_paths.append([manual1_path])
        else:
            print(f"    Warning: Missing ground truth for {img_name} in Subset 2")
            print(f"      Missing: {manual1_path}")

    print(f"    Total valid DRIVE images with ground truth: {len(valid_image_paths)}")
    return valid_image_paths, gt_paths


def generate_readme(dataset_name, metrics):
    if not metrics:
        return "No valid metrics were calculated."

    # Create README content
    readme = "# DRIVE Performance\n\n"
    readme += f"## {dataset_name} Dataset Performance Metrics\n\n"

    # Add processing time information
    readme += "### Processing Time\n\n"
    readme += f"- **Total Processing Time**: {metrics['total_processing_time']:.2f} seconds\n"
    readme += f"- **Average Processing Time per Image**: {metrics['processing_time_mean']:.2f} ± {metrics['processing_time_std']:.2f} seconds\n"
    readme += f"- **Number of Images Processed**: {metrics['num_images']}\n\n"

    # Add performance metrics
    readme += "### Performance Metrics\n\n"
    readme += "| Metric | Mean | Standard Deviation |\n"
    readme += "|--------|------|-------------------|\n"
    readme += f"| Accuracy | {metrics['accuracy_mean']:.4f} | {metrics['accuracy_std']:.4f} |\n"
    readme += f"| Sensitivity | {metrics['sensitivity_mean']:.4f} | {metrics['sensitivity_std']:.4f} |\n"
    readme += f"| Specificity | {metrics['specificity_mean']:.4f} | {metrics['specificity_std']:.4f} |\n"

    return readme


def main():
    # Set path to STARE dataset
    stare_path = 'Fundus Databases/DRIVE'

    # Track total processing time
    overall_start_time = time.time()

    # Process STARE dataset
    if os.path.exists(stare_path):
        # Load dataset
        image_paths, gt_paths = load_drive_dataset(stare_path)

        if image_paths:
            print(f"Processing STARE dataset ({len(image_paths)} images)...")

            # Process dataset
            metrics, total_time, avg_time = process_dataset(
                image_paths,
                gt_paths,
                num_workers=None  # Adjust based on your CPU cores
            )

            print(f"Completed processing STARE dataset.")
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Average time per image: {avg_time:.2f} seconds")

            # Generate and save README
            readme_content = generate_readme('STARE', metrics)
            with open('DRIVE Performance.md', 'w') as f:
                f.write(readme_content)

            print("README file 'DRIVE Performance.md' has been generated.")
        else:
            print("Error: Could not find matching images and ground truths in STARE dataset.")
    else:
        print(f"Warning: STARE dataset not found at {stare_path}")

    # Calculate overall processing time
    overall_processing_time = time.time() - overall_start_time
    print(f"Overall processing time: {overall_processing_time:.2f} seconds")


if __name__ == "__main__":
    main()