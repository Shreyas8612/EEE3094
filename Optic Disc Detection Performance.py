import os
import cv2
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops
from skimage import color
from datetime import datetime


class OpticDiscDetector:
    def __init__(self):
        pass

    def threshold_level(self, image):
        # Convert to uint8 and flatten
        image_uint8 = (image * 255).astype(np.uint8)
        hist = cv2.calcHist([image_uint8], [0], None, [256], [0, 256])
        hist = hist.flatten()
        bin_centers = np.arange(256)

        # Initialize
        cumulative_sum = np.cumsum(hist)
        t = np.zeros(100)  # Allocate array for thresholds

        # Initial threshold
        t[0] = np.sum(bin_centers * hist) / cumulative_sum[-1]
        t[0] = np.round(t[0])

        # Calculate mean below threshold and mean above threshold
        i = 0
        idx = int(t[i])

        if idx >= 256:
            idx = 255
        if idx <= 0:
            idx = 1

        cumulative_sum_below = np.sum(hist[:idx])
        if cumulative_sum_below > 0:
            mbt = np.sum(bin_centers[:idx] * hist[:idx]) / cumulative_sum_below
        else:
            mbt = 0

        cumulative_sum_above = np.sum(hist[idx:])
        if cumulative_sum_above > 0:
            mat = np.sum(bin_centers[idx:] * hist[idx:]) / cumulative_sum_above
        else:
            mat = 0

        # Next threshold
        i = 1
        t[i] = np.round((mat + mbt) / 2)

        # Iterate until convergence
        while abs(t[i] - t[i - 1]) >= 1 and i < 98:
            idx = int(t[i])
            if idx >= 256:
                idx = 255
            if idx <= 0:
                idx = 1

            cumulative_sum_below = np.sum(hist[:idx])
            if cumulative_sum_below > 0:
                mbt = np.sum(bin_centers[:idx] * hist[:idx]) / cumulative_sum_below
            else:
                mbt = 0

            cumulative_sum_above = np.sum(hist[idx:])
            if cumulative_sum_above > 0:
                mat = np.sum(bin_centers[idx:] * hist[idx:]) / cumulative_sum_above
            else:
                mat = 0

            i += 1
            t[i] = np.round((mat + mbt) / 2)

        threshold = t[i]
        level = threshold / 255.0

        return level

    def anisotropic_diffusion(self, img, niter=1, kappa=50, gamma=0.1):
        # Convert to float32 for numerical stability
        diffused = img.astype(np.float32)

        for _ in range(niter):
            # Compute finite differences (gradients) in the four directions
            gradN = np.roll(diffused, 1, axis=0) - diffused
            gradS = np.roll(diffused, -1, axis=0) - diffused
            gradE = np.roll(diffused, -1, axis=1) - diffused
            gradW = np.roll(diffused, 1, axis=1) - diffused

            # Perona–Malik conduction coefficients
            cN = np.exp(-(gradN / kappa) ** 2)
            cS = np.exp(-(gradS / kappa) ** 2)
            cE = np.exp(-(gradE / kappa) ** 2)
            cW = np.exp(-(gradW / kappa) ** 2)

            # Update the image by discrete PDE
            diffused += gamma * (
                    cN * gradN + cS * gradS +
                    cE * gradE + cW * gradW
            )

        return diffused

    def preprocess_image(self, image):
        # Downscale to 128x128
        resized = cv2.resize(image, (128, 128))

        # Convert to float in range [0,1]
        converted_image = resized.astype(np.float32) / np.max(resized)

        # Convert BGR to RGB for proper LAB conversion
        converted_image_rgb = cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB)

        # Convert to LAB color space
        lab_image = color.rgb2lab(converted_image_rgb)

        # Extract only the L channel
        gray_image = lab_image[:, :, 0]  # L-channel only

        # Normalize to [0,1]
        gray = (gray_image - np.min(gray_image)) / (np.max(gray_image) - np.min(gray_image))

        # Apply median filtering with 5x5 kernel
        median_filtered = median_filter(gray, size=5)

        # Calculate standard deviation
        sd = np.std(median_filtered)

        # Apply intensity transformation based on standard deviation (Gamma Correction)
        bias = 7.25
        transformed = np.power(median_filtered, 1 / (bias * sd))
        transformed = transformed / np.max(transformed)

        # Background subtraction
        # Estimate background using median filter with 15x15 kernel
        background = median_filter(transformed, size=15)
        subtracted = transformed - background

        # Median filter again with 7x7 kernel
        median_filtered_again = median_filter(subtracted, size=7)

        # Apply anisotropic diffusion
        diffused = self.anisotropic_diffusion(
            median_filtered_again,
            niter=2,
            kappa=30,
            gamma=0.1
        )

        return diffused

    def gravitational_edge_detection(self, img):
        # Get image dimensions
        height, width = img.shape

        # Initialize the edge map
        edge_map = np.zeros((height, width), dtype=np.float64)

        # For each pixel in the image
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Get the 8-neighborhood of the pixel
                neighborhood = img[i - 1:i + 2, j - 1:j + 2]

                # Calculate average intensity in the neighborhood
                g_avg = np.mean(neighborhood)

                # Calculate standard deviation in the neighborhood
                sigma = np.std(neighborhood)

                # Calculate Gravitational Constant
                C = 1 / (1 + np.exp(sigma * (img[i, j] - g_avg)))

                # Initialize force components
                Fx = 0
                Fy = 0

                # For each neighbor in the 8-neighborhood
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        # Skip the center pixel itself
                        if k == i and l == j:
                            continue

                        # Calculate vector magnitude
                        r_magnitude = np.sqrt((k - i) ** 2 + (l - j) ** 2)

                        # Calculate force components
                        f_x = C * img[k, l] * (k - i) / (r_magnitude ** 3)
                        f_y = C * img[k, l] * (l - j) / (r_magnitude ** 3)

                        # Add to the total force components
                        Fx += f_x
                        Fy += f_y

                # Calculate the size of the force
                F_magnitude = np.sqrt(Fx ** 2 + Fy ** 2)

                # Assign the magnitude to the edge map
                edge_map[i, j] = F_magnitude

        # Normalize the edge map to [0, 255]
        edge_map = (edge_map - np.min(edge_map)) / (np.max(edge_map) - np.min(edge_map)) * 255

        return edge_map.astype(np.uint8)

    def post_process_edge_map(self, edge_map):
        # Get image dimensions
        height, width = edge_map.shape
        center_y, center_x = height // 2, width // 2

        # Create a mask for filtering
        mask = np.ones_like(edge_map, dtype=bool)

        # Remove pixels outside a radius of 52 pixels from the center
        for i in range(height):
            for j in range(width):
                if (i - center_y) ** 2 + (j - center_x) ** 2 > 52 ** 2:
                    mask[i, j] = False

        # Remove pixels inside a radius of 6 pixels from the center
        for i in range(height):
            for j in range(width):
                if (i - center_y) ** 2 + (j - center_x) ** 2 < 6 ** 2:
                    mask[i, j] = False

        # Remove rectangular regions from top and bottom (30 pixels height)
        pixel_rect = 30
        mask[:pixel_rect, :] = False
        mask[-pixel_rect:, :] = False

        # Remove a vertical strip of 5 pixels width from the center
        mask[:, center_x - 2:center_x + 3] = False

        # Apply the mask to the edge map
        filtered_edge_map = edge_map.copy()
        filtered_edge_map[~mask] = 0

        # Thresholding
        threshold_bias = 0.05
        level = self.threshold_level(filtered_edge_map / 255.0)
        binary_edge_map = (filtered_edge_map > (level + threshold_bias) * 255).astype(np.uint8) * 255

        return binary_edge_map

    def candidate_selection(self, binary_edge_map):
        D = 15.7  # Predetermined threshold

        # Label connected components
        labeled = label(binary_edge_map)

        # Get properties of the labeled regions
        regions = regionprops(labeled)

        # Sort regions by area (largest first)
        regions.sort(key=lambda x: x.area, reverse=True)

        # Iterate through regions to find the candidate
        candidate_found = False
        selected_region_idx = -1
        optic_disc_location = None

        for i, region in enumerate(regions):
            # Get coordinates of the pixels in the region
            coords = region.coords

            # Calculate centroid
            centroid = region.centroid

            # Calculate Euclidean distance from centroid to each pixel in the region
            total_distance = 0
            for coord in coords:
                distance = np.sqrt((coord[0] - centroid[0]) ** 2 + (coord[1] - centroid[1]) ** 2)
                total_distance += distance

            # Average distance of region pixels from its centroid
            mean_distance = total_distance / len(coords)

            # If mean distance is less than a threshold, select this region
            if mean_distance < D:
                candidate_found = True
                optic_disc_location = (int(centroid[1]), int(centroid[0]))
                break

        # If no suitable region is found, return the centroid of the largest region
        if not candidate_found and regions:
            centroid = regions[0].centroid
            optic_disc_location = (int(centroid[1]), int(centroid[0]))
        elif not regions:
            optic_disc_location = (64, 64)  # Center of the image

        return optic_disc_location

    def detect_optic_disc(self, image_path):
        # Start timing
        start_time = time.time()

        # Load the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Failed to read image: {image_path}")
            return None, 0

        original_height, original_width = image.shape[:2]

        # Preprocess the image
        preprocessed = self.preprocess_image(image)

        # Apply gravitational edge detection
        edge_map = self.gravitational_edge_detection(preprocessed)

        # Post-process the edge map
        binary_edge_map = self.post_process_edge_map(edge_map)

        # Select the candidate optic disc location
        optic_disc_location_128 = self.candidate_selection(binary_edge_map)

        # Scale the location back to the original image dimensions
        optic_disc_location = (
            int(optic_disc_location_128[0] * original_width / 128),
            int(optic_disc_location_128[1] * original_height / 128)
        )

        # End timing
        elapsed_time = time.time() - start_time

        return optic_disc_location, elapsed_time


class OpticDiscEvaluator:
    def __init__(self, detector, stare_gt_path, hrf_gt_path):
        self.detector = detector
        self.stare_gt = self.load_stare_gt(stare_gt_path)
        self.hrf_gt, self.hrf_diameters = self.load_hrf_gt(hrf_gt_path)

    def load_stare_gt(self, path):
        gt_data = {}
        df = pd.read_csv(path)

        for _, row in df.iterrows():
            img_name = row['image']
            y, x = int(row['x']), int(row['y'])

            # Skip entries with -1, -1 (no optic disc)
            if x == -1 and y == -1:
                continue

            gt_data[img_name] = (x, y)

        return gt_data

    def load_hrf_gt(self, path):
        gt_data = {}
        diameters = {}

        df = pd.read_csv(path)

        for _, row in df.iterrows():
            img_name = row['image']
            x, y = int(row['Pap. Center x']), int(row['Pap. Center y'])
            diameter = float(row['disk diameter'])

            gt_data[img_name] = (x, y)
            diameters[img_name] = diameter

        return gt_data, diameters

    def calculate_distance(self, pred, gt):
        return np.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)

    def evaluate_accuracy(self, dataset_path, dataset_type, output_dir):
        correct_count = 0
        total_count = 0
        total_time = 0
        failed_count = 0

        # Lists to store ground truth and prediction coordinates for scatter plot
        gt_x, gt_y = [], []
        pred_x, pred_y = [], []
        image_names = []
        distances = []
        thresholds = []
        correctness = []

        # Process each ground truth entry
        if dataset_type == 'STARE':
            gt_data = self.stare_gt
            file_extension = '.ppm'
            images_subdir = ''
        else:  # HRF
            gt_data = self.hrf_gt
            file_extension = '.JPG'
            images_subdir = 'images'

        print(f"\n=== Processing {dataset_type} Dataset ===")
        print(f"Found {len(gt_data)} ground truth entries")

        # Process each ground truth entry
        for img_name, gt_location in tqdm(gt_data.items(), desc=f"Processing {dataset_type} dataset"):
            # Construct the full image path
            if dataset_type == 'STARE':
                base_name = img_name.split('.')[0]
                img_path = os.path.join(dataset_path, f"{base_name}{file_extension}")
            else:
                img_path_upper = os.path.join(dataset_path, images_subdir, f"{img_name}.JPG")
                img_path_lower = os.path.join(dataset_path, images_subdir, f"{img_name}.jpg")

                # Check which path exists
                if os.path.exists(img_path_upper):
                    img_path = img_path_upper
                elif os.path.exists(img_path_lower):
                    img_path = img_path_lower
                else:
                    print(f"Warning: Image not found for {img_name}")
                    failed_count += 1
                    continue

            # First check if the file exists
            if not os.path.exists(img_path):
                print(f"File does not exist: {img_path}")
                failed_count += 1
                continue

            # Detect optic disc
            try:
                pred_location, elapsed_time = self.detector.detect_optic_disc(img_path)

                if pred_location is None:
                    print(f"Failed to process: {img_path}")
                    failed_count += 1
                    continue
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                failed_count += 1
                continue

            # Get image dimensions to calculate a diagonal-based threshold
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                failed_count += 1
                continue

            # The Rest of the evaluation logic remains the same
            h, w = image.shape[:2]
            img_diagonal = np.sqrt(h ** 2 + w ** 2)
            threshold = 0.08 * img_diagonal  # 8% of image diagonal

            # Calculate distance between prediction and ground truth
            distance = self.calculate_distance(pred_location, gt_location)

            # Check if the prediction is correct
            is_correct = distance <= threshold

            # Update counters
            total_count += 1
            if is_correct:
                correct_count += 1

            # Update time tracking
            total_time += elapsed_time

            # Store coordinates for scatter plot
            gt_x.append(gt_location[0])
            gt_y.append(gt_location[1])
            pred_x.append(pred_location[0])
            pred_y.append(pred_location[1])
            image_names.append(img_name)
            distances.append(distance)
            thresholds.append(threshold)
            correctness.append("Correct" if is_correct else "Incorrect")

        # Print summary
        print(f"\nSummary for {dataset_type}:")
        print(f"Total ground truth entries: {len(gt_data)}")
        print(f"Successfully processed: {total_count}")
        print(f"Failed to process: {failed_count}")

        # Calculate accuracy and average time
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        avg_time = total_time / total_count if total_count > 0 else 0

        # Create a detailed results table for README
        detailed_results = []
        for i in range(len(image_names)):
            detailed_results.append({
                'Image': image_names[i],
                'GT_X': gt_x[i],
                'GT_Y': gt_y[i],
                'Pred_X': pred_x[i],
                'Pred_Y': pred_y[i],
                'Distance': distances[i],
                'Threshold': thresholds[i],
                'Correct': correctness[i]
            })

        return accuracy, avg_time, detailed_results

    def save_results(self, dataset_type, accuracy, avg_time, detailed_results, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # Create README file
        readme_path = os.path.join(output_dir, f'{dataset_type} OD Detection Performance.md')

        with open(readme_path, 'w') as f:
            # Write header
            f.write(f"# Optic Disc Detection Performance - {dataset_type} Dataset\n\n")
            f.write(f"## Evaluation Summary\n\n")
            f.write(f"- **Threshold:** 8% of image diagonal\n")
            f.write(f"- **Accuracy:** {accuracy:.2f}%\n")
            f.write(f"- **Average processing time per image:** {avg_time:.4f} seconds\n\n")

            # Write detailed results
            f.write(f"## Detailed Results\n\n")
            f.write("| Image | Ground Truth (X,Y) | Prediction (X,Y) | Distance | Threshold | Result |\n")
            f.write("|-------|-------------------|------------------|----------|-----------|--------|\n")

            for result in detailed_results:
                f.write(
                    f"| {result['Image']} | ({result['GT_X']},{result['GT_Y']}) | ({result['Pred_X']},{result['Pred_Y']}) | {result['Distance']:.2f} | {result['Threshold']:.2f} | {result['Correct']} |\n")

        print(f"README file saved to {readme_path}")


def main():
    # Define fixed paths (no command-line arguments)
    stare_path = 'Fundus Databases/STARE Optic Disc Detection'
    hrf_path = 'Fundus Databases/HRF'
    stare_gt = 'Fundus Databases/STARE Optic Disc Detection/GT_OPTIC_DISC.csv'
    hrf_gt = 'Fundus Databases/HRF/optic_disk_centers.csv'
    output_dir = 'EEE3094'

    # Initialize the detector and evaluator
    detector = OpticDiscDetector()
    evaluator = OpticDiscEvaluator(detector, stare_gt, hrf_gt)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # STARE dataset evaluation
    if os.path.exists(stare_path):
        print("\n=== Evaluating STARE Dataset ===")

        # Check if PPM files exist in the directory
        ppm_files = [f for f in os.listdir(stare_path) if f.endswith('.ppm')]
        print(f"Found {len(ppm_files)} PPM files in {stare_path}")
        if len(ppm_files) > 0:
            print(f"Example files: {ppm_files[:5]}")

        accuracy, avg_time, detailed_results = evaluator.evaluate_accuracy(
            stare_path, 'STARE', output_dir
        )

        print(f"STARE Dataset - Threshold: 8% of image diagonal")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average processing time per image: {avg_time:.4f} seconds")

        # Save results as README
        evaluator.save_results('STARE', accuracy, avg_time, detailed_results, output_dir)
    else:
        print(f"STARE dataset path '{stare_path}' not found")

    # HRF dataset evaluation
    if os.path.exists(hrf_path):
        print("\n=== Evaluating HRF Dataset ===")
        accuracy, avg_time, detailed_results = evaluator.evaluate_accuracy(
            hrf_path, 'HRF', output_dir
        )

        print(f"HRF Dataset - Threshold: 8% of image diagonal")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Average processing time per image: {avg_time:.4f} seconds")

        # Save results as README
        evaluator.save_results('HRF', accuracy, avg_time, detailed_results, output_dir)
    else:
        print(f"HRF dataset path '{hrf_path}' not found")


if __name__ == "__main__":
    main()