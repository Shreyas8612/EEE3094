import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import color, morphology


def threshold_level(image):
    # Convert to uint8 and flatten
    image_uint8 = (image * 255).astype(np.uint8)
    hist = cv2.calcHist([image_uint8], [0], None, [256], [0, 256])
    hist = hist.flatten()
    bin_centers = np.arange(256)

    # Normalise histogram
    cumulative_sum = np.cumsum(hist)
    t = np.zeros(100)  # Allocate an array for thresholds

    # Initial threshold
    t[0] = np.sum(bin_centers * hist) / cumulative_sum[-1]
    t[0] = np.round(t[0])

    # Calculate mean below a threshold and mean the above threshold
    i = 0

    # Function to calculate means below and above a threshold
    def calculate_means(threshold_idx):
        idx = int(threshold_idx)
        if idx >= 256: idx = 255
        if idx <= 0: idx = 1

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

        return mbt, mat

    # Calculate initial means
    mbt, mat = calculate_means(t[i])

    # Next threshold
    i = 1
    t[i] = np.round((mat + mbt) / 2)

    # Iterate until convergence
    while abs(t[i] - t[i - 1]) >= 1 and i < 98:
        mbt, mat = calculate_means(t[i])
        i += 1
        t[i] = np.round((mat + mbt) / 2)

    threshold = t[i]
    level = threshold / 255.0
    return level

def detect_fundus_boundary(img):
    # Convert to float in range [0,1]
    converted_image = img.astype(np.float32) / np.max(img)

    # Convert BGR to RGB for proper LAB conversion
    converted_image_rgb = cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB)

    # Convert to LAB color space
    lab_image = color.rgb2lab(converted_image_rgb)

    # Store original channels for visualization
    gray = lab_image[:, :, 0]

    # Normalize to [0,1]
    gray_min = np.min(gray)
    gray_max = np.max(gray)
    gray = (gray - gray_min) / (gray_max - gray_min)

    # Apply median filter
    # Convert to uint8 for median filter
    gray_uint8 = (gray * 255).astype(np.uint8)
    blurred = cv2.medianBlur(gray_uint8, 25)

    # Convert back to float [0,1]
    blurred = blurred.astype(np.float32) / 255.0

    # Use the custom threshold_level function for thresholding
    level = threshold_level(blurred)

    # Create a binary image using the threshold level
    # Try both and select the one that produces a better mask
    bias = 0.15
    binary1 = (blurred > level - bias).astype(np.uint8)
    binary2 = (blurred < level - bias).astype(np.uint8)

    # Select the binary image that is more likely to be the fundus mask
    cc1 = get_largest_component(binary1)
    cc2 = get_largest_component(binary2)

    if np.sum(cc1) > np.sum(cc2):
        largest_component = cc1
    else:
        largest_component = cc2

    # Erode the mask slightly to avoid boundary artifacts
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.erode(largest_component, erode_kernel, iterations=1)

    return mask

def get_largest_component(binary_image):
    # Get connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # Skip label 0, which is the background
    if num_labels == 1:
        return binary_image  # No components found

    # Find the largest component by area (excluding a background)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    # Create a new image with only the largest component
    largest_component = (labels == largest_label).astype(np.uint8)

    return largest_component


def apply_mask(image, mask):
    # Normalize mask to [0,1] if needed
    if np.max(mask) > 1.0:
        mask = mask.astype(np.float32) / 255.0

    # Apply mask according to image dimensions
    if len(image.shape) > 2:
        # RGB image
        mask_3d = np.stack([mask, mask, mask], axis=2)
        masked_image = image * mask_3d
    else:
        # Grayscale image
        masked_image = image * mask

    return masked_image

def apply_lab_weights(lab_image, l_weight, a_weight, b_weight):
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    # Apply weights
    weighted_image = l_weight * l_channel + a_weight * a_channel + b_weight * b_channel

    # Normalize to [0,1]
    min_val = np.min(weighted_image)
    max_val = np.max(weighted_image)
    normalized_image = (weighted_image - min_val) / (max_val - min_val)

    return normalized_image


def process_fundus_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = cv2.resize(image, (565, 584))

    # Convert to float in range [0,1]
    converted_image = resized_image.astype(np.float32) / np.max(resized_image)

    # Convert BGR to RGB for proper LAB conversion
    converted_image_rgb = cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB)
    converted_image_uint8 = (converted_image_rgb * 255).astype(np.uint8)

    # Detect fundus boundary early on original image
    fundus_mask = detect_fundus_boundary(converted_image_uint8)

    # Convert to LAB color space
    lab_image = color.rgb2lab(converted_image_rgb)

    # Apply weights to the LAB imager
    gray_image = apply_lab_weights(lab_image, 0.3, -0.3, 0.0)

    # Apply CLAHE for contrast enhancement
    gray_image_uint8 = (gray_image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    enhanced_image_uint8 = clahe.apply(gray_image_uint8)
    enhanced_image = enhanced_image_uint8.astype(np.float32) / np.max(enhanced_image_uint8)

    # Apply average filter
    kernel_size = 7
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(enhanced_image, -1, kernel)

    # Subtract enhanced from filtered
    subtracted_image = filtered_image - enhanced_image

    # Apply custom thresholding
    level = threshold_level(subtracted_image)
    binary_image = (subtracted_image > (level - 0.485121)).astype(np.uint8)

    # Remove small objects
    clean_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=150, connectivity=200)
    clean_image = clean_image.astype(np.uint8)

    # Fill holes in clean image using dilation followed by erosion
    kernel_dial = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean_image_dial = cv2.dilate(clean_image, kernel_dial, iterations=1)
    kernel_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    clean_image_erode = cv2.erode(clean_image_dial, kernel_ero, iterations=1)

    # Skeletonize the image
    skeleton_image = morphology.skeletonize(clean_image_erode.astype(bool))

    # Apply fundus mask to all result images
    masked_erode_image = apply_mask(clean_image_erode, fundus_mask)
    masked_skeleton_image = apply_mask(skeleton_image.astype(np.float32), fundus_mask)

    return {
        'converted_image_rgb': converted_image_rgb,
        'masked_erode_image': masked_erode_image,
        'masked_skeleton_image': masked_skeleton_image
    }

# Isolate vessels from fundus image
image_path = 'Fundus Databases/STARE Vessel Isolation/7.tif'
isolate_vessels = process_fundus_image(image_path)

# Plot the images
fig, axs = plt.subplots(1, 3, figsize=(15, 8))
axs[0].imshow(isolate_vessels['converted_image_rgb'])
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow(isolate_vessels['masked_erode_image'], cmap='grey')
axs[1].set_title('Isolated Vessels')
axs[1].axis('off')
axs[2].imshow(isolate_vessels['masked_skeleton_image'], cmap='grey')
axs[2].set_title('Skeletonised Image')
axs[2].axis('off')
plt.tight_layout()
#plt.show()