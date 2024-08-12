import argparse

import torch
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from ultralytics import YOLO




parser = argparse.ArgumentParser(description='choose model path')
parser.add_argument('--model', type=str, default=f'C:/Users\dan12138\Downloads\simple_YOLO2\simple_YOLO-main\models\yolov8x_DW_swin_FOCUS2_sppc.yaml',
                    help='trained model path')
parser.add_argument('--source', type=str, default=os.path.join(current_dir, 'assets/bus.jpg'))
args = parser.parse_args()

model = YOLO(args.model, verbose=True)
result = model(args.source)
