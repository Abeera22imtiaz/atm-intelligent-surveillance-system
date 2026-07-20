import os
from ultralytics import YOLO
from src.config import BASE_PATH, VIDEO_SOURCE

def run_atms_detection():
    # 1. Model path define karein
    model_path = os.path.join(BASE_PATH, "runs", "atm_surveillance_run", "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 2. Permanent Save Path (Drive folder)
    # Note: Google Drive notebook mein pehle se mount honi chahiye
    output_project = "/content/drive/MyDrive/ATM_Surveillance_Results"
    output_dir_name = "detection_output_v1"
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Detecting objects in: {VIDEO_SOURCE}...")
    
    # 3. Detection parameters
    results = model.predict(
        source=VIDEO_SOURCE,
        save=True,
        stream=True, 
        conf=0.45,
        iou=0.4,
        max_det=50,
        imgsz=640,
        project=output_project,
        name=output_dir_name,
        exist_ok=True,
        verbose=False,     # Terminal spam rukne ke liye
        vid_stride=3           # Har 3rd frame process hoga (Memory load kam karne ke liye)
    )
    
    # 4. Iteration loop
    for i, r in enumerate(results):
        if i % 50 == 0:
            print(f"Processing frame {i}...")
    
    print("="*50)
    print(f"Detection complete! Output saved at: {output_project}/{output_dir_name}")
    print("="*50)

if __name__ == "__main__":
    run_atms_detection()