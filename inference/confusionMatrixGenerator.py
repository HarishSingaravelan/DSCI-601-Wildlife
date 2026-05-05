# import os
# import sys
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from torch.utils.data import DataLoader
# from typing import Tuple, List, Dict, Any, Callable
# from tqdm import tqdm # Import tqdm for progress bar

# # -------------------------------------------------------------------
# # 1. SETUP PATHS & IMPORTS
# # -------------------------------------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# try:
#     from modeling.model import get_model
#     from modeling.trainer import Trainer
#     from turbine_processing.transforms import get_val_transform
#     from turbine_processing.dataset import TurbineCocoDataset 
# except ImportError as e:
#     print(f"[FATAL ERROR] Project import failed. Check paths/names: {e}")
#     sys.exit(1)

# # -------------------------------------------------------------------
# # 2. CONFIGURATION
# # -------------------------------------------------------------------

# SHARED_ROOT = "../../../shared/rc/turbine/" 
# TEST_ANN_FILE = os.path.join(SHARED_ROOT, "turbine_split", "test", "annotations.json")

# # --- Model and Analysis Config ---
# MODEL_WEIGHTS_PATH = 'fasterrcnn_turbine_adam_transforms_newClass.pth' 
# NUM_OBJECT_CLASSES = 5
# NUM_CLASSES = NUM_OBJECT_CLASSES + 1 # 6 total (1-5 objects, 0 background)
# CLASS_NAMES = ['Background', 'red_winged_blackbird', 'mammal_carcass', 'hawk_species', 'gull_species', 'trash']
# CONFIDENCE_THRESHOLD = 0.5        # Score threshold for a prediction to be considered
# IOU_THRESHOLD = 0.5               # IoU threshold for a prediction to be a True Positive
# OUTPUT_IMAGE_PATH = 'normalized_confusion_matrix_final.png'

# # -------------------------------------------------------------------
# # 3. UTILITY FUNCTIONS
# # -------------------------------------------------------------------

# def get_config():
#     """Helper to structure configuration."""
#     return {
#         "data": {
#             "test_data_root": SHARED_ROOT,
#             "test_ann_file": TEST_ANN_FILE
#         },
#         "training": {
#             "device": "cuda",
#             "output_model_path": MODEL_WEIGHTS_PATH 
#         }
#     }

# def collate_fn(batch: List[Tuple[Any, Dict[str, torch.Tensor]]]):
#     """Custom collate function for object detection batching."""
#     return tuple(zip(*batch))

# def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
#     """Calculates IoU between two bounding boxes ([xmin, ymin, xmax, ymax])."""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     unionArea = float(boxAArea + boxBArea - interArea)
#     if unionArea == 0:
#         return 0.0
        
#     iou = interArea / unionArea
#     return iou

# # -------------------------------------------------------------------
# # 4. OBJECT DETECTION EVALUATION CORE
# # -------------------------------------------------------------------

# def run_evaluation_and_match(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
#     """
#     Runs inference and performs IoU matching to generate matched (True, Pred) label pairs.
#     """
#     model.eval()
#     all_matched_true_labels = []
#     all_matched_pred_labels = []
    
#     # Wrap the test_loader with tqdm for a progress bar
#     progress_bar = tqdm(test_loader, desc="Running Inference & IoU Matching", total=len(test_loader))
    
#     with torch.no_grad():
#         for batch_idx, (images, targets) in enumerate(progress_bar):
#             images = [img.to(device) for img in images]
#             outputs = model(images) 
            
#             for target, output in zip(targets, outputs):
#                 # 1. Prepare Data
#                 gt_boxes = target['boxes'].cpu().numpy()
#                 gt_labels = target['labels'].cpu().numpy()
                
#                 pred_scores = output['scores'].cpu().numpy()
#                 high_conf_indices = pred_scores >= CONFIDENCE_THRESHOLD
#                 pred_boxes = output['boxes'].cpu().numpy()[high_conf_indices]
#                 pred_labels = output['labels'].cpu().numpy()[high_conf_indices]

#                 # Initialize tracking lists for this image
#                 matched_pred_indices = set()
#                 image_true_labels = []
#                 image_pred_labels = []

#                 # 2. Match Ground Truths (TPs and FNs)
#                 for i_gt, gt_box in enumerate(gt_boxes):
#                     best_iou = -1
#                     best_pred_idx = -1
                    
#                     # Find the best prediction that hasn't been used yet
#                     for i_pred, pred_box in enumerate(pred_boxes):
#                         if i_pred in matched_pred_indices:
#                             continue

#                         iou = calculate_iou(gt_box, pred_box)
#                         if iou > best_iou:
#                             best_iou = iou
#                             best_pred_idx = i_pred
                    
#                     if best_iou >= IOU_THRESHOLD:
#                         # True Positive (TP): GT matched
#                         image_true_labels.append(gt_labels[i_gt])
#                         image_pred_labels.append(pred_labels[best_pred_idx])
#                         matched_pred_indices.add(best_pred_idx) 
#                     else:
#                         # False Negative (FN): GT missed
#                         image_true_labels.append(gt_labels[i_gt])
#                         image_pred_labels.append(0) # Predicted as 'Background' (Missed)
                
#                 # 3. False Positives (FPs)
#                 # Predictions that didn't match any GT box
#                 all_pred_indices = set(range(len(pred_boxes)))
#                 unmatched_pred_indices = all_pred_indices - matched_pred_indices
                
#                 for i_pred in unmatched_pred_indices:
#                     # False Positive (FP): A detection that hit nothing real
#                     image_true_labels.append(0) # True label is 'Background'
#                     image_pred_labels.append(pred_labels[i_pred]) # Predicted as object X

#                 # 4. Append results
#                 all_matched_true_labels.extend(image_true_labels)
#                 all_matched_pred_labels.extend(image_pred_labels)
                
#             # Log total matched samples to the progress bar postfix
#             progress_bar.set_postfix_str(f"Total Samples: {len(all_matched_true_labels)}")

#     return all_matched_true_labels, all_matched_pred_labels

# # -------------------------------------------------------------------
# # 5. PLOTTING FUNCTION
# # -------------------------------------------------------------------

# def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
#     """Plots a normalized confusion matrix and saves it to a file."""
    
#     from sklearn.metrics import classification_report
    
#     # 1. Print Detailed Report (Now that the samples are consistent length)
#     target_labels = list(range(1, len(class_names)))
#     target_names = class_names[1:] 
    
#     print("\nPer-Class Classification Report (IoU Matched)")
#     print(f"(Metrics calculated based on IoU={IOU_THRESHOLD} and Conf={CONFIDENCE_THRESHOLD})")
    
#     report = classification_report(
#         y_true, 
#         y_pred, 
#         target_names=target_names, 
#         labels=target_labels,
#         zero_division=0 
#     )
#     print(report)
    
#     # 2. Generate the Confusion Matrix and Normalize
#     cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
#     # Normalize by row (True Label) to show classification percentage
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

#     # 3. Setup the Plotting Environment
#     plt.figure(figsize=(12, 10))
#     plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(f'Normalized OD Confusion Matrix (IoU={IOU_THRESHOLD}, Conf={CONFIDENCE_THRESHOLD})')
#     plt.colorbar(fraction=0.046, pad=0.04)
    
    
#     tick_marks = np.arange(len(class_names))
#     plt.xticks(tick_marks, class_names, rotation=45, ha='right')
#     plt.yticks(tick_marks, class_names)

#     # 4. Add Text Labels
#     thresh = cm_normalized.max() / 2.
#     for i in range(cm_normalized.shape[0]):
#         for j in range(cm_normalized.shape[1]):
#             if cm_normalized[i, j] > 0.01: 
#                  plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
#                          horizontalalignment="center",
#                          color="white" if cm_normalized[i, j] > thresh else "black",
#                          fontsize=10)

#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.tight_layout()
    
#     # 5. Save the figure
#     plt.savefig(output_path, dpi=300, bbox_inches='tight') 
#     print(f"\n Confusion Matrix saved to {output_path}")

# # -------------------------------------------------------------------
# # 6. MAIN EXECUTION
# # -------------------------------------------------------------------

# if __name__ == '__main__':
#     cfg = get_config()
#     device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

#     try:
#         # 1. Load Model
#         model = get_model(num_classes=NUM_CLASSES)
#         model_path_full = os.path.join(parent_dir, cfg["training"]["output_model_path"])
        
#         if not os.path.exists(model_path_full):
#             model_path_full = cfg["training"]["output_model_path"] 
        
#         model.load_state_dict(torch.load(model_path_full, map_location=device))
#         model.to(device)
#         print(f"[INFO] Model loaded successfully from {model_path_full}")

#         # 2. Load Data
#         val_transform = get_val_transform()
#         test_dataset = TurbineCocoDataset(
#             images_dir=cfg["data"]["test_data_root"],
#             ann_file=cfg["data"]["test_ann_file"],
#             transforms=val_transform
#         )
#         test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
#         print(f"[INFO] Test dataset loaded with {len(test_dataset)} images.")

#         # 3. Run Evaluation & Matching
#         true_labels, pred_labels = run_evaluation_and_match(model, test_loader, device)
        
#         if len(true_labels) == 0:
#             print("\n[INFO] No samples were generated. Check data paths or IoU/Confidence thresholds.")
#         elif len(true_labels) != len(pred_labels):
#             # This should ideally not happen if IoU matching is correct
#             print(f"\n[FATAL ERROR] IoU matching failed. Final lists still mismatch: {len(true_labels)} vs {len(pred_labels)}")
#         else:
#             print(f"\n[INFO] Successfully matched {len(true_labels)} samples.")
#             # 4. Plot the Confusion Matrix and print report
#             plot_confusion_matrix(true_labels, pred_labels, CLASS_NAMES, OUTPUT_IMAGE_PATH)
            
#     except FileNotFoundError as e:
#         print(f"\n[FATAL ERROR] File not found. Check paths in config or imports: {e}")
#     except Exception as e:
#         print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")