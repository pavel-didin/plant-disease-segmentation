import cv2
import numpy as np
from scipy.sparse.csgraph import connected_components

def merge_enclaves(mask, threshold=15, hole_threshold=None):
    if hole_threshold is None:
        hole_threshold = threshold
    result = mask.copy()
    # Remove small white regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < threshold:
            result[labels == i] = 0
    # Remove small black holes
    inverted = cv2.bitwise_not(result)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < hole_threshold:
            result[labels == i] = 255
    return result

def detectleafborder(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 100)
    return edges

def gradient_segmentation_with_boosting(image, merged_mask, model, scaler,
                                        enclave_threshold=10000,
                                        min_region_area=15,
                                        boundary_window=5,
                                        debug=False):
    edges = detectleafborder(image)
    modified_edges = edges.copy()
    modified_edges[merged_mask == 1] = 0

    contours, _ = cv2.findContours(merged_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(modified_edges, contours, -1, 255, 1)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(modified_edges, kernel, iterations=1)
    closed_edges = cv2.erode(dilated_edges, kernel, iterations=1)

    closed_edges_inv = cv2.bitwise_not(closed_edges)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed_edges_inv, connectivity=4)

    output = image.copy()
    region_pixels_dict = {label: [] for label in range(1, num_labels)}

    for label in range(1, num_labels):
        region_mask = (labels == label) & (merged_mask == 0)
        if np.any(region_mask):
            region_pixels_dict[label] = image[region_mask]
            mean_color = np.mean(region_pixels_dict[label], axis=0)
            output[region_mask] = mean_color

    boundary_mask = (closed_edges_inv == 0) & (merged_mask == 0)
    half_window = boundary_window // 2
    y_indices, x_indices = np.where(boundary_mask)
    updated_labels = labels.copy()

    for y, x in zip(y_indices, x_indices):
        min_dist = float('inf')
        best_label = None
        for dy in range(-half_window, half_window + 1):
            for dx in range(-half_window, half_window + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
                    if not boundary_mask[ny, nx] and not merged_mask[ny, nx]:
                        neighbor_label = labels[ny, nx]
                        if neighbor_label > 0:
                            color_dist = np.linalg.norm(image[y, x] - image[ny, nx])
                            if color_dist < min_dist:
                                min_dist = color_dist
                                best_label = neighbor_label
        if best_label is not None:
            updated_labels[y, x] = best_label
            region_pixels_dict[best_label] = np.vstack([
                region_pixels_dict[best_label],
                image[y, x]
            ])
            output[y, x] = np.mean(region_pixels_dict[best_label], axis=0)

    region_colors = {}
    region_areas = {}
    region_features = {}

    for label in range(1, num_labels):
        if len(region_pixels_dict[label]) > 0:
            mean_bgr = np.mean(region_pixels_dict[label], axis=0)
            region_colors[label] = mean_bgr
            region_mask = (updated_labels == label) & (merged_mask == 0)
            region_areas[label] = np.sum(region_mask)

            pixels = region_pixels_dict[label]
            area = region_areas[label]
            var_bgr = np.var(pixels, axis=0)
            brightness = np.mean(mean_bgr)

            mean_bgr_uint8 = np.round(mean_bgr).astype(np.uint8).reshape(1,1,3)
            mean_hsv = cv2.cvtColor(mean_bgr_uint8, cv2.COLOR_BGR2HSV)[0,0]

            region_features[label] = {
                'area': area,
                'mean_B': mean_bgr[0], 'mean_G': mean_bgr[1], 'mean_R': mean_bgr[2],
                'mean_H': mean_hsv[0], 'mean_S': mean_hsv[1], 'mean_V': mean_hsv[2],
                'var_B': var_bgr[0], 'var_G': var_bgr[1], 'var_R': var_bgr[2],
                'brightness': brightness
            }

    infection_mask = np.zeros_like(merged_mask, dtype=np.uint8)
    feature_order = ['area', 'mean_B', 'mean_G', 'mean_R',
                     'mean_H', 'mean_S', 'mean_V',
                     'var_B', 'var_G', 'var_R', 'brightness']
    for label, feats in region_features.items():
        if feats['area'] < min_region_area:
            continue
        X = np.array([[feats[k] for k in feature_order]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        if pred == 0:
            region_mask = (updated_labels == label) & (merged_mask == 0)
            infection_mask[region_mask] = 1

    combined_mask = np.logical_or(merged_mask, infection_mask).astype(np.uint8)*255
    merged_combined_mask = merge_enclaves(combined_mask, threshold=enclave_threshold)

    updated_infection_mask = np.where(
        (merged_combined_mask == 255) & (merged_mask == 0), 1, 0
    ).astype(np.uint8)

    contours, _ = cv2.findContours(updated_infection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = output.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 1)
    contour_image[merged_mask == 1] = image[merged_mask == 1]
    output[merged_mask == 1] = image[merged_mask == 1]

    if debug:
        print(f"Found {len(region_features)} regions, infected (class 0): {np.sum(infection_mask>0)} pixels")

    return updated_infection_mask, contour_image, merged_combined_mask
