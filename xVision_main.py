import argparse
# src folder ke andar se files import ho rahi hain
from src.dataset_preprocessor import run_preprocessing
from src.trainer import start_training
from src.xVision_evalute import evaluate_model
from src.xvision_detect import run_atms_detection
from src.tracker import run_trash_tracking

def main():
    parser = argparse.ArgumentParser(description="ATM Surveillance Pipeline Manager")
    parser.add_argument('--task', choices=['preprocess', 'train', 'eval','track' ,'detect'],
                        required=True, help="Task to perform")
    
    args = parser.parse_args()

    if args.task == 'preprocess':
        run_preprocessing()
    elif args.task == 'train':
        start_training()
    elif args.task == 'eval':
        evaluate_model()
    elif args.task == 'track':
        run_trash_tracking()

    elif args.task == 'detect':
        run_atms_detection()

if __name__ == "__main__":
    main()