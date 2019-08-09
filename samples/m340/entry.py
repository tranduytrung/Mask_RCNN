"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights. Also auto download COCO dataset
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet --download=True

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

from mrcnn import model as modellib, utils
from mrcnn.config import Config
from samples.m340.dataset import CapturedRGBDataset, CapturedRGBDDataset
import os
import sys
import time
import numpy as np
import shutil
import skimage

ROOT_DIR = os.path.abspath(".")
print(f'root={ROOT_DIR}')


# # Root directory of the project
# ROOT_DIR = os.path.abspath("../../")
# # Import Mask RCNN
# sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
ROOT_DIR = os.path.abspath(".")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# BATCH SIZE
BATCH_SIZE = 4

############################################################
#  Configurations
############################################################


class M340Config(Config):
    """Configuration for training on M340
    Derives from the base Config class and overrides values specific
    to the M340 dataset.
    """
    # Give the configuration a recognizable name
    NAME = "m340"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = BATCH_SIZE

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6
    IMAGES_PER_GPU = 4
    STEPS_PER_EPOCH = 0  # define by trainer
    VALIDATION_STEPS = 0  # define by trainer
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 768
    IMAGE_SOURCES = 2
    IMAGE_CHANNEL_COUNT = 4
    BACKBONE = "resnet50"
    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    PRE_NMS_LIMIT = 3000

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 200

    RPN_ANCHOR_STRIDE = 1
    TRAIN_BN = False

############################################################
#  Training
############################################################


def train(weights_path, data_dir, log_dir):
    # init dataset
    cat2eid = {
        'capot': 1,
        'cartefille': 2,
        'cartemere': 3,
        'faceavant': 4,
        'tag': 5,
        'visu': 6,
    }
    # Training dataset.
    train_data_dir = os.path.join(data_dir, 'train')
    train_meta_path = os.path.join(train_data_dir, 'annotations.csv')
    train_meta_d_path = os.path.join(train_data_dir, 'annotations_d.csv')
    # dataset_train = CapturedRGBDataset(train_data_dir, train_meta_path, cat2eid=cat2eid)
    dataset_train = CapturedRGBDDataset(
        train_data_dir, [train_meta_path, train_meta_d_path], cat2eid=cat2eid)
    # Validation dataset
    val_data_dir = os.path.join(data_dir, 'val')
    val_meta_path = os.path.join(val_data_dir, 'annotations.csv')
    val_meta_d_path = os.path.join(val_data_dir, 'annotations_d.csv')
    # dataset_val = CapturedRGBDataset(val_data_dir, val_meta_path, cat2eid=cat2eid, augmentations=False)
    dataset_val = CapturedRGBDDataset(val_data_dir, [
                                      val_meta_path, val_meta_d_path], cat2eid=cat2eid, augmentations=False)

    # crate config
    config = M340Config()

    # update spe
    train_spe = len(dataset_train) // BATCH_SIZE + 1
    val_spe = len(dataset_val) // BATCH_SIZE + 1
    config.STEPS_PER_EPOCH = train_spe
    config.VALIDATION_STEPS = val_spe
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=log_dir)

    # Select weights file to load
    if weights_path is not None:
        exclude = None
        if weights_path.lower() == "coco":
            weights_path = COCO_MODEL_PATH
            exclude = ["mrcnn_class_logits",
                       "mrcnn_bbox_fc", "mrcnn_bbox"]
        elif weights_path.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif weights_path.lower() == "imagenet":
            exclude = exclude = ["conv1"]
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()

        # load weights
        model.load_weights(weights_path, by_name=True, exclude=exclude)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 2,
                epochs=20,
                layers='4+')

    print('Training completed!')


def infer(weights_path, data_dir, log_dir):
    class InferenceConfig(M340Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.7
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=log_dir)


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on M340.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on M340")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--weights', required=False, type=str,
                        default=None,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    # parser.add_argument('--epochs', required=False, type=int,
    #                 default=40,
    #                 metavar="40",
    #                 help='Number of epochs to train')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        train(args.weights, args.dataset, args.logs)
    else:
        infer(args.weights, args.dataset, args.logs)
