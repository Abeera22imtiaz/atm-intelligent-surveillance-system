import os

# 1. Model Execution & Inference Parameters
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
TRAINING_EPOCHS = 50
BATCH_SIZE = 16

# 2. Tracking Core Parameters (ATM Surveillance Specific)
TRASH_ALARM_THRESHOLD_SECS = 50.0  

# 3. Permanent Drive Paths Configuration
BASE_PATH = "/content/drive/MyDrive/ATM_Surveillance_Project"
RAW_DATA_DIR = os.path.join(BASE_PATH, "data")                
FINAL_DATASET_DIR = os.path.join(BASE_PATH, "final_dataset")  
DATA_YAML_PATH = os.path.join(BASE_PATH, "data.yaml")          
VIDEO_SOURCE = os.path.join(BASE_PATH, "atm20260525_135421.mp4")            
OUTPUT_VIDEO_PATH = os.path.join(BASE_PATH, "output_tracked.mp4")
FINAL_DATASET_DIRS = os.path.join(BASE_PATH, "final_datasets")  

# Path to the best trained weights (used by tracker.py)
MODEL_WEIGHTS_PATH = os.path.join(BASE_PATH, "runs", "atm_surveillance_run", "weights", "best.pt")

# Where the end-of-video dwell-time report gets written
REPORT_PATH = os.path.join(BASE_PATH, "trash_report.txt")

# 5. Trash Spatial-Memory Tracker Parameters (occlusion-robustness tuning)

# How long (seconds) a trash object may go undetected with NO person nearby
# before it is declared "removed". Absorbs normal detection flicker/misses.
OCCLUSION_GRACE_SECS = 5.0

# Centroid-distance threshold for matching a new detection to an existing
# trash ID, expressed as a ratio of the frame diagonal (resolution-independent).
MATCH_DIST_RATIO = 0.06

# How close a person box's centroid must be to a trash object's last known
# position (as a ratio of frame diagonal) to count as "occluding" it.
PERSON_PROXIMITY_RATIO = 0.10
# After a trash object is marked "removed", how long (seconds) a reappearing
# detection at the same spot can still revive its ORIGINAL ID rather than
# getting a new one. Safety net for brief detection dropouts.
REOPEN_WINDOW_SECS = 3.0
 
# 4. Custom Surveillance Categories Mapping (0: trash, 1: person)
CLASS_MAP = {
    0: 'trash',
    1: 'person'
}
