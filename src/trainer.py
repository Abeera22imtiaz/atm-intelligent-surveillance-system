import os
from ultralytics import YOLO
# Permanent Drive paths and training hyperparameters are imported from the config file
from src.config import DATA_YAML_PATH, TRAINING_EPOCHS, BATCH_SIZE, BASE_PATH

def start_training():
    """
    Initialized with YOLOv8s for focused training on the specific environment.
    Only essential parameters are kept to ensure high precision.
    """
    print("\n" + "="*50)
    print("INITIALIZING FOCUSED SURVEILLANCE TRAINING")
    print("="*50)
    
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"Configuration mismatch! 'data.yaml' not found at: {DATA_YAML_PATH}")

    project_output_dir = os.path.join(BASE_PATH, "runs")
    os.makedirs(project_output_dir, exist_ok=True)

    # Load the pre-trained YOLOv8s model
    print("Loading pre-trained weights (YOLOv8s)...")
    model = YOLO("yolov8s.pt")

    print(f"Starting training for {TRAINING_EPOCHS} epochs with batch size {BATCH_SIZE}...")

    # Train the model with essential settings
    model.train(
        data=DATA_YAML_PATH,
        epochs=TRAINING_EPOCHS,          # Total training cycles
        batch=BATCH_SIZE,                # Images per batch
        imgsz=640,                       # Input resolution
        device=0,                        # GPU usage
        project=project_output_dir,      # Results directory
        name="atm_surveillance_run",     # Run folder name
        exist_ok=True,                   # Overwrite existing
        pretrained=True,                 # Transfer learning from COCO
        patience=20                      # Slightly increased patience for stable convergence
    )

    print("\n" + "="*50)
    print("TRAINING SUCCESSFUL! Model optimized for current environment.")
    print("="*50)

if __name__ == "__main__":
    start_training()
