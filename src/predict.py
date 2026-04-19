import argparse
import pickle
import cv2
import numpy as np
from segmentation_utils import merge_enclaves, gradient_segmentation_with_boosting

def load_models(knn_path, xgb_path):
    with open(knn_path, 'rb') as f:
        knn = pickle.load(f)
    with open(xgb_path, 'rb') as f:
        saved = pickle.load(f)
    model = saved['model']
    scaler = saved['scaler']
    return knn, model, scaler

def segment_image(image_path, knn, xgb_model, scaler):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)
    pred_knn = knn.predict(pixels)
    green_mask = pred_knn.reshape(h, w).astype(np.uint8) * 255
    green_mask = merge_enclaves(green_mask, threshold=50, hole_threshold=15)
    merged_mask = merge_enclaves(green_mask, threshold=10000)

    # Internal infection (enclaves)
    internal = cv2.bitwise_and(merged_mask, cv2.bitwise_not(green_mask))
    internal = merge_enclaves(internal, threshold=30, hole_threshold=10)

    # External infection via XGBoost
    external, contour_img, _ = gradient_segmentation_with_boosting(
        image, merged_mask, xgb_model, scaler, debug=False
    )

    full_infection = cv2.bitwise_or(internal, external)
    full_infection_binary = (full_infection > 0).astype(np.uint8)

    # Draw contours on original image
    result_img = image.copy()
    cnts, _ = cv2.findContours(full_infection_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, cnts, -1, (0, 0, 255), 1)

    return result_img, full_infection_binary

def main():
    parser = argparse.ArgumentParser(description='Segment diseased areas on a plant leaf image.')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', help='Path to save output image with contours')
    parser.add_argument('--mask_output', help='Path to save binary mask')
    parser.add_argument('--knn_model', default='models/knn_classifier.pkl', help='Path to KNN model')
    parser.add_argument('--xgb_model', default='models/xgboost_model.pkl', help='Path to XGBoost model')
    args = parser.parse_args()

    knn, xgb_model, scaler = load_models(args.knn_model, args.xgb_model)
    result_img, mask = segment_image(args.image, knn, xgb_model, scaler)

    if args.output:
        cv2.imwrite(args.output, result_img)
        print(f"Output image saved to {args.output}")
    if args.mask_output:
        cv2.imwrite(args.mask_output, mask * 255)
        print(f"Binary mask saved to {args.mask_output}")

if __name__ == '__main__':
    main()
