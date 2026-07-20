from ultralytics import YOLO
from src.config import BASE_PATH, DATA_YAML_PATH
import os

def evaluate_model():
    """
    Evaluates the trained YOLO model on the test dataset split.
    Uses the existing data.yaml configuration.
    """
    # Path to the best trained weights
    model_path = os.path.join(BASE_PATH, "runs", "atm_surveillance_run", "weights", "best.pt")
    
    # Initialize the model
    model = YOLO(model_path)
    
    print(f"Starting evaluation on the test split using {DATA_YAML_PATH}...")
    
    # Perform validation on the test set
    # Note: Ensure your data.yaml has a 'test' key pointing to your test images
    metrics = model.val(
        data=DATA_YAML_PATH,
        split='test',
        imgsz=640
    )
    
    # Output the primary mAP result
    print(f"\nEvaluation Results:")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    evaluate_model()