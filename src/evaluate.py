import os
import argparse
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import f1_score, fbeta_score, jaccard_score
from segmentation_utils import merge_enclaves, gradient_segmentation_with_boosting

def compute_metrics(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    iou = jaccard_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='binary', zero_division=0)
    f2 = fbeta_score(y_true_flat, y_pred_flat, beta=2, average='binary', zero_division=0)
    return iou, f1, f2

def load_models(knn_path, xgb_path):
    with open(knn_path, 'rb') as f:
        knn = pickle.load(f)
    with open(xgb_path, 'rb') as f:
        saved = pickle.load(f)
    model = saved['model']
    scaler = saved['scaler']
    return knn, model, scaler

def process_image_full(image_path, knn, xgb_model, scaler):
    image = cv2.imread(image_path)
    if image is None:
        return None
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)
    pred_knn = knn.predict(pixels)
    green_mask = pred_knn.reshape(h, w).astype(np.uint8) * 255
    green_mask = merge_enclaves(green_mask, threshold=50, hole_threshold=15)
    merged_mask = merge_enclaves(green_mask, threshold=10000)

    internal = cv2.bitwise_and(merged_mask, cv2.bitwise_not(green_mask))
    internal = merge_enclaves(internal, threshold=30, hole_threshold=10)

    external, _, _ = gradient_segmentation_with_boosting(
        image, merged_mask, xgb_model, scaler, debug=False
    )

    full = cv2.bitwise_or(internal, external)
    full = (full > 0).astype(np.uint8)
    return full

def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation on a dataset of images and ground truth masks.')
    parser.add_argument('--images', required=True, help='Directory with input images')
    parser.add_argument('--masks', required=True, help='Directory with ground truth masks')
    parser.add_argument('--knn_model', default='models/knn_classifier.pkl', help='Path to KNN model')
    parser.add_argument('--xgb_model', default='models/xgboost_model.pkl', help='Path to XGBoost model')
    parser.add_argument('--visualize', action='store_true', help='Show each pair (requires GUI)')
    args = parser.parse_args()

    knn, xgb_model, scaler = load_models(args.knn_model, args.xgb_model)

    # Find matching pairs
    mask_files = [f for f in os.listdir(args.masks) if f.lower().endswith(('.png','.jpg','.jpeg'))]
    valid_pairs = []
    for mf in mask_files:
        base = os.path.splitext(mf)[0]
        for ext in ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG']:
            img_path = os.path.join(args.images, base + ext)
            if os.path.exists(img_path):
                valid_pairs.append((mf, img_path))
                break
    print(f"Found {len(valid_pairs)} image-mask pairs.")

    all_metrics = []
    for mask_file, img_path in tqdm(valid_pairs):
        gt_mask = cv2.imread(os.path.join(args.masks, mask_file), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        gt_binary = (gt_mask > 0).astype(np.uint8)

        pred_binary = process_image_full(img_path, knn, xgb_model, scaler)
        if pred_binary is None:
            continue

        if gt_binary.shape != pred_binary.shape:
            pred_binary = cv2.resize(pred_binary, (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST)

        iou, f1, f2 = compute_metrics(gt_binary, pred_binary)
        all_metrics.append((iou, f1, f2))

        if args.visualize:
            cv2.imshow('GT', gt_binary*255)
            cv2.imshow('Pred', pred_binary*255)
            cv2.waitKey(0)
    cv2.destroyAllWindows()

    if all_metrics:
        avg_iou = np.mean([m[0] for m in all_metrics])
        avg_f1  = np.mean([m[1] for m in all_metrics])
        avg_f2  = np.mean([m[2] for m in all_metrics])
        print("\n=== Evaluation Results ===")
        print(f"Pairs processed: {len(all_metrics)}")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average F1 : {avg_f1:.4f}")
        print(f"Average F2 : {avg_f2:.4f}")
    else:
        print("No valid pairs found.")

if __name__ == '__main__':
    main()
