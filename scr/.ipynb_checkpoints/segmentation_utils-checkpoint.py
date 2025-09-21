import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
import radiomics
from radiomics import featureextractor
import argparse
import numpy as np
from sklearn.cluster import KMeans
import scipy.ndimage as ndimage
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz
from skimage.filters import threshold_otsu
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from scipy.ndimage import distance_transform_edt, binary_erosion
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import pandas as pd
from matplotlib import colors
import cv2
from skimage import measure


def make_bold(text):
    return f"\033[1m{text}\033[0m"

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = itkimage.GetOrigin()
    numpySpacing = itkimage.GetSpacing()
    return numpyImage, numpyOrigin, numpySpacing

def normalize_image_to_uint8(image, lower_bound=-1000, upper_bound=100):
    clipped_img = np.clip(image, lower_bound, upper_bound)
    normalized_img = ((clipped_img - lower_bound) / (upper_bound - lower_bound)) * 255.0
    normalized_img = normalized_img.astype(np.uint8)
    return normalized_img




def segment_nodule_kmeans(ct_image, bbox_center, bbox_whd, margin=5, n_clusters=2):
    """
    Segments a nodule in a 3D CT image using k-means clustering with a margin around the bounding box.

    Parameters:
    - ct_image: 3D NumPy array representing the CT image.
    - bbox_center: Tuple of (x, y, z) coordinates for the center of the bounding box.
    - bbox_whd: Tuple of (w, h, d) representing the width, height, and depth of the bounding box.
    - margin: Margin to add around the bounding box (default is 5).
    - n_clusters: Number of clusters to use in k-means (default is 2).

    Returns:
    - segmented_image: 3D NumPy array with the segmented nodule.
    """

    x_center, y_center, z_center = bbox_center
    w, h, d = bbox_whd

    # Calculate the bounding box with margin
    x_start, x_end = max(0, x_center - w//2 - margin), min(ct_image.shape[0], x_center + w//2 + margin)
    y_start, y_end = max(0, y_center - h//2 - margin), min(ct_image.shape[1], y_center + h//2 + margin)
    z_start, z_end = max(0, z_center - d//2 - margin), min(ct_image.shape[2], z_center + d//2 + margin)

    bbox_region = ct_image[x_start:x_end, y_start:y_end, z_start:z_end]

    # Reshape the region for k-means clustering
    flat_region = bbox_region.reshape(-1, 1)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(flat_region)
    labels = kmeans.labels_

    # Reshape the labels back to the original bounding box shape
    clustered_region = labels.reshape(bbox_region.shape)

    # Assume the nodule is in the cluster with the highest mean intensity
    nodule_cluster = np.argmax(kmeans.cluster_centers_)

    # Create a binary mask for the nodule
    nodule_mask = (clustered_region == nodule_cluster)

    # Apply morphological operations to refine the segmentation
    nodule_mask = ndimage.binary_closing(nodule_mask, structure=np.ones((3, 3, 3)))
    nodule_mask = ndimage.binary_opening(nodule_mask, structure=np.ones((2, 2, 2)))

    # Initialize the segmented image
    segmented_image = np.zeros_like(ct_image, dtype=np.uint8)

    # Place the nodule mask in the correct position in the segmented image
    segmented_image[x_start:x_end, y_start:y_end, z_start:z_end] = nodule_mask

    return segmented_image


def segment_nodule_gmm(ct_image, bbox_center, bbox_whd, margin=5, n_components=2):
    """
    Segments a nodule in a 3D CT image using a Gaussian Mixture Model with a margin around the bounding box.

    Parameters:
    - ct_image: 3D NumPy array representing the CT image.
    - bbox_center: Tuple of (x, y, z) coordinates for the center of the bounding box.
    - bbox_whd: Tuple of (w, h, d) representing the width, height, and depth of the bounding box.
    - margin: Margin to add around the bounding box (default is 5).
    - n_components: Number of components to use in the Gaussian Mixture Model (default is 2).

    Returns:
    - segmented_image: 3D NumPy array with the segmented nodule.
    """

    x_center, y_center, z_center = bbox_center
    w, h, d = bbox_whd

    # Calculate the bounding box with margin
    x_start, x_end = max(0, x_center - w//2 - margin), min(ct_image.shape[0], x_center + w//2 + margin)
    y_start, y_end = max(0, y_center - h//2 - margin), min(ct_image.shape[1], y_center + h//2 + margin)
    z_start, z_end = max(0, z_center - d//2 - margin), min(ct_image.shape[2], z_center + d//2 + margin)

    bbox_region = ct_image[x_start:x_end, y_start:y_end, z_start:z_end]

    # Reshape the region for GMM
    flat_region = bbox_region.reshape(-1, 1)

    # Perform GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0).fit(flat_region)
    labels = gmm.predict(flat_region)

    # Reshape the labels back to the original bounding box shape
    clustered_region = labels.reshape(bbox_region.shape)

    # Assume the nodule is in the component with the highest mean intensity
    nodule_component = np.argmax(gmm.means_)

    # Create a binary mask for the nodule
    nodule_mask = (clustered_region == nodule_component)

    # Apply morphological operations to refine the segmentation
    nodule_mask = ndimage.binary_closing(nodule_mask, structure=np.ones((3, 3, 3)))
    nodule_mask = ndimage.binary_opening(nodule_mask, structure=np.ones((3, 3, 3)))

    # Initialize the segmented image
    segmented_image = np.zeros_like(ct_image, dtype=np.uint8)

    # Place the nodule mask in the correct position in the segmented image
    segmented_image[x_start:x_end, y_start:y_end, z_start:z_end] = nodule_mask

    return segmented_image


def segment_nodule_fcm(ct_image, bbox_center, bbox_whd, margin=5, n_clusters=2):
    """
    Segments a nodule in a 3D CT image using Fuzzy C-means clustering with a margin around the bounding box.

    Parameters:
    - ct_image: 3D NumPy array representing the CT image.
    - bbox_center: Tuple of (x, y, z) coordinates for the center of the bounding box.
    - bbox_whd: Tuple of (w, h, d) representing the width, height, and depth of the bounding box.
    - margin: Margin to add around the bounding box (default is 5).
    - n_clusters: Number of clusters to use in Fuzzy C-means (default is 2).

    Returns:
    - segmented_image: 3D NumPy array with the segmented nodule.
    """

    x_center, y_center, z_center = bbox_center
    w, h, d = bbox_whd

    # Calculate the bounding box with margin
    x_start, x_end = max(0, x_center - w//2 - margin), min(ct_image.shape[0], x_center + w//2 + margin)
    y_start, y_end = max(0, y_center - h//2 - margin), min(ct_image.shape[1], y_center + h//2 + margin)
    z_start, z_end = max(0, z_center - d//2 - margin), min(ct_image.shape[2], z_center + d//2 + margin)

    bbox_region = ct_image[x_start:x_end, y_start:y_end, z_start:z_end]

    # Reshape the region for FCM
    flat_region = bbox_region.reshape(-1, 1)

    # Perform FCM clustering
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(flat_region.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

    # Assign each voxel to the cluster with the highest membership
    labels = np.argmax(u, axis=0)

    # Reshape the labels back to the original bounding box shape
    clustered_region = labels.reshape(bbox_region.shape)

    # Assume the nodule is in the cluster with the highest mean intensity
    nodule_cluster = np.argmax(cntr)

    # Create a binary mask for the nodule
    nodule_mask = (clustered_region == nodule_cluster)

    # Apply morphological operations to refine the segmentation
    nodule_mask = ndimage.binary_closing(nodule_mask, structure=np.ones((3, 3, 3)))
    nodule_mask = ndimage.binary_opening(nodule_mask, structure=np.ones((3, 3, 3)))

    # Initialize the segmented image
    segmented_image = np.zeros_like(ct_image, dtype=np.uint8)

    # Place the nodule mask in the correct position in the segmented image
    segmented_image[x_start:x_end, y_start:y_end, z_start:z_end] = nodule_mask

    return segmented_image



def segment_nodule_otsu(ct_image, bbox_center, bbox_whd, margin=5):
    """
    Segments a nodule in a 3D CT image using Otsu's thresholding with a margin around the bounding box.

    Parameters:
    - ct_image: 3D NumPy array representing the CT image.
    - bbox_center: Tuple of (x, y, z) coordinates for the center of the bounding box.
    - bbox_whd: Tuple of (w, h, d) representing the width, height, and depth of the bounding box.
    - margin: Margin to add around the bounding box (default is 5).

    Returns:
    - segmented_image: 3D NumPy array with the segmented nodule.
    """

    x_center, y_center, z_center = bbox_center
    w, h, d = bbox_whd

    # Calculate the bounding box with margin
    x_start, x_end = max(0, x_center - w//2 - margin), min(ct_image.shape[0], x_center + w//2 + margin)
    y_start, y_end = max(0, y_center - h//2 - margin), min(ct_image.shape[1], y_center + h//2 + margin)
    z_start, z_end = max(0, z_center - d//2 - margin), min(ct_image.shape[2], z_center + d//2 + margin)

    bbox_region = ct_image[x_start:x_end, y_start:y_end, z_start:z_end]

    # Flatten the region for thresholding
    flat_region = bbox_region.flatten()

    # Calculate the Otsu threshold
    otsu_threshold = threshold_otsu(flat_region)

    # Apply the threshold to create a binary mask
    nodule_mask = bbox_region >= otsu_threshold

    # Apply morphological operations to refine the segmentation
    nodule_mask = ndimage.binary_closing(nodule_mask, structure=np.ones((3, 3, 3)))
    nodule_mask = ndimage.binary_opening(nodule_mask, structure=np.ones((3, 3, 3)))

    # Initialize the segmented image
    segmented_image = np.zeros_like(ct_image, dtype=np.uint8)

    # Place the nodule mask in the correct position in the segmented image
    segmented_image[x_start:x_end, y_start:y_end, z_start:z_end] = nodule_mask

    return segmented_image



def expand_mask_by_distance(segmented_nodule_gmm, spacing, expansion_mm):
    """
    Expands the segmentation mask by a given distance in mm in all directions by directly updating pixel values.

    Parameters:
    segmented_nodule_gmm (numpy array): 3D binary mask of the nodule (1 for nodule, 0 for background).
    spacing (tuple): Spacing of the image in mm for each voxel, given as (spacing_x, spacing_y, spacing_z).
    expansion_mm (float): Distance to expand the mask in millimeters.

    Returns:
    numpy array: Expanded segmentation mask.
    """
    # Reorder spacing to match the numpy array's (z, y, x) format
    spacing_reordered = (spacing[2], spacing[1], spacing[0])  # (spacing_z, spacing_y, spacing_x)

    # Calculate the number of pixels to expand in each dimension
    expand_pixels = np.array([int(np.round(expansion_mm / s)) for s in spacing_reordered])

    # Create a new expanded mask with the same shape
    expanded_mask = np.zeros_like(segmented_nodule_gmm)

    # Get the coordinates of all white pixels in the original mask
    white_pixel_coords = np.argwhere(segmented_nodule_gmm == 1)

    # Expand each white pixel by adding the specified number of pixels in each direction
    for coord in white_pixel_coords:
        z, y, x = coord  # Extract the z, y, x coordinates of each white pixel

        # Define the range to expand for each coordinate
        z_range = range(max(0, z - expand_pixels[0]), min(segmented_nodule_gmm.shape[0], z + expand_pixels[0] + 1))
        y_range = range(max(0, y - expand_pixels[1]), min(segmented_nodule_gmm.shape[1], y + expand_pixels[1] + 1))
        x_range = range(max(0, x - expand_pixels[2]), min(segmented_nodule_gmm.shape[2], x + expand_pixels[2] + 1))

        # Update the new mask by setting all pixels in this range to 1
        for z_new in z_range:
            for y_new in y_range:
                for x_new in x_range:
                    expanded_mask[z_new, y_new, x_new] = 1

    return expanded_mask


def find_nodule_lobe(cccwhd, lung_mask, class_map):
    """
    Determine the lung lobe where a nodule is located based on a 3D mask and bounding box.

    Parameters:
    cccwhd (list or tuple): Bounding box in the format [center_x, center_y, center_z, width, height, depth].
    lung_mask (numpy array): 3D array representing the lung mask with different lung regions.
    class_map (dict): Dictionary mapping lung region labels to their names.

    Returns:
    str: Name of the lung lobe where the nodule is located.
    """
    center_x, center_y, center_z, width, height, depth = cccwhd

    # Calculate the bounding box limits
    start_x = int(center_x - width // 2)
    end_x = int(center_x + width // 2)
    start_y = int(center_y - height // 2)
    end_y = int(center_y + height // 2)
    start_z = int(center_z - depth // 2)
    end_z = int(center_z + depth // 2)

    # Ensure the indices are within the mask dimensions
    start_x = max(0, start_x)
    end_x = min(lung_mask.shape[0], end_x)
    start_y = max(0, start_y)
    end_y = min(lung_mask.shape[1], end_y)
    start_z = max(0, start_z)
    end_z = min(lung_mask.shape[2], end_z)

    # Extract the region of interest (ROI) from the mask
    roi = lung_mask[start_x:end_x, start_y:end_y, start_z:end_z]

    # Count the occurrences of each lobe label within the ROI
    unique, counts = np.unique(roi, return_counts=True)
    label_counts = dict(zip(unique, counts))

    # Exclude the background (label 0)
    if 0 in label_counts:
        del label_counts[0]

    # Find the label with the maximum count
    if label_counts:
        nodule_lobe = max(label_counts, key=label_counts.get)
    else:
        nodule_lobe = None

    # Map the label to the corresponding lung lobe
    if nodule_lobe is not None:
        nodule_lobe_name = class_map["lungs"][nodule_lobe]
    else:
        nodule_lobe_name = "Undefined"

    return nodule_lobe_name


def find_nodule_lobe_and_distance(cccwhd, lung_mask, class_map,spacing):
    """
    Determine the lung lobe where a nodule is located and measure its distance from the lung wall.

    Parameters:
    cccwhd (list or tuple): Bounding box in the format [center_x, center_y, center_z, width, height, depth].
    lung_mask (numpy array): 3D array representing the lung mask with different lung regions.
    class_map (dict): Dictionary mapping lung region labels to their names.

    Returns:
    tuple: (Name of the lung lobe, Distance from the lung wall)
    """
    center_x, center_y, center_z, width, height, depth = cccwhd

    # Calculate the bounding box limits
    start_x = int(center_x - width // 2)
    end_x = int(center_x + width // 2)
    start_y = int(center_y - height // 2)
    end_y = int(center_y + height // 2)
    start_z = int(center_z - depth // 2)
    end_z = int(center_z + depth // 2)

    # Ensure the indices are within the mask dimensions
    start_x = max(0, start_x)
    end_x = min(lung_mask.shape[0], end_x)
    start_y = max(0, start_y)
    end_y = min(lung_mask.shape[1], end_y)
    start_z = max(0, start_z)
    end_z = min(lung_mask.shape[2], end_z)

    # Extract the region of interest (ROI) from the mask
    roi = lung_mask[start_x:end_x, start_y:end_y, start_z:end_z]

    # Count the occurrences of each lobe label within the ROI
    unique, counts = np.unique(roi, return_counts=True)
    label_counts = dict(zip(unique, counts))

    # Exclude the background (label 0)
    if 0 in label_counts:
        del label_counts[0]

    # Find the label with the maximum count
    if label_counts:
        nodule_lobe = max(label_counts, key=label_counts.get)
    else:
        nodule_lobe = None

    # Map the label to the corresponding lung lobe
    if nodule_lobe is not None:
        nodule_lobe_name = class_map["lungs"][nodule_lobe]
    else:
        nodule_lobe_name = "Undefined"

    # Calculate the distance from the nodule centroid to the nearest lung wall
    nodule_centroid = np.array([center_x, center_y, center_z])
    
    # Create a binary lung mask where lung region is 1 and outside lung is 0
    lung_binary_mask = lung_mask > 0

    # Create the lung wall mask by finding the outer boundary
    # Use binary erosion to shrink the lung mask, then subtract it from the original mask to get the boundary
    lung_eroded    = binary_erosion(lung_binary_mask)
    lung_wall_mask = lung_binary_mask & ~lung_eroded  # Lung wall mask is the outermost boundary (contour)

    # Compute the distance transform from the lung wall
    distance_transform = distance_transform_edt(~lung_wall_mask)  # Compute distance to nearest lung boundary

    
    
    # Get the distance from the nodule centroid to the nearest lung wall in voxel units
    voxel_distance_to_lung_wall = distance_transform[center_x, center_y, center_z]

    # Convert voxel distance to real-world distance in mm
    physical_distance_to_lung_wall = voxel_distance_to_lung_wall * np.sqrt(
        spacing[0]**2 + spacing[1]**2 + spacing[2]**2
    )



    return nodule_lobe_name, voxel_distance_to_lung_wall,physical_distance_to_lung_wall


# +
def expand_mask_by_distance(segmented_nodule_gmm, spacing, expansion_mm):
    """
    Expands the segmentation mask by a given distance in mm in all directions by directly updating pixel values.

    Parameters:
    segmented_nodule_gmm (numpy array): 3D binary mask of the nodule (1 for nodule, 0 for background).
    spacing (tuple): Spacing of the image in mm for each voxel, given as (spacing_x, spacing_y, spacing_z).
    expansion_mm (float): Distance to expand the mask in millimeters.

    Returns:
    numpy array: Expanded segmentation mask.
    """
    # Reorder spacing to match the numpy array's (z, y, x) format
    spacing_reordered = (spacing[2], spacing[1], spacing[0])  # (spacing_z, spacing_y, spacing_x)

    # Calculate the number of pixels to expand in each dimension
    expand_pixels = np.array([int(np.round(expansion_mm / s)) for s in spacing_reordered])

    # Create a new expanded mask with the same shape
    expanded_mask = np.zeros_like(segmented_nodule_gmm)

    # Get the coordinates of all white pixels in the original mask
    white_pixel_coords = np.argwhere(segmented_nodule_gmm == 1)

    # Expand each white pixel by adding the specified number of pixels in each direction
    for coord in white_pixel_coords:
        z, y, x = coord  # Extract the z, y, x coordinates of each white pixel

        # Define the range to expand for each coordinate
        z_range = range(max(0, z - expand_pixels[0]), min(segmented_nodule_gmm.shape[0], z + expand_pixels[0] + 1))
        y_range = range(max(0, y - expand_pixels[1]), min(segmented_nodule_gmm.shape[1], y + expand_pixels[1] + 1))
        x_range = range(max(0, x - expand_pixels[2]), min(segmented_nodule_gmm.shape[2], x + expand_pixels[2] + 1))

        # Update the new mask by setting all pixels in this range to 1
        for z_new in z_range:
            for y_new in y_range:
                for x_new in x_range:
                    expanded_mask[z_new, y_new, x_new] = 1

    return expanded_mask



# Function to plot the contours of a mask
def plot_contours(ax, mask, color, linewidth=1.5):
    contours = measure.find_contours(mask, level=0.5)  # Find contours at a constant level
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)

