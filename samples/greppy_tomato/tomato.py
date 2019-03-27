"""
Mask R-CNN
Train on the tomato dataset .

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Ganesh Nagaraja

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 tomato.py train --dataset=/path/to/tomato/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 tomato.py train --dataset=/path/to/tomato/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 tomato.py train --dataset=/path/to/tomato/dataset --weights=imagenet

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import time
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.insert(0, ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# import sys
# sys.path.append("/home/gani/deeplearning/cocoapi/PythonAPI/pycocotools")
# from coco import COCO
# from cocoeval import COCOeval
# import mask as maskUtils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class TomatoConfig(Config):
    """Configuration for training on the tomato farm  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tomato"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    #resize input image dimensions
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1


############################################################
#  Dataset
############################################################

class TomatoDataset(utils.Dataset):

    def load_tomato(self, dataset_dir, subset):
        """Load a subset of the Tomato dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("tomato", 1, "stem")
        self.add_class("tomato", 2, "peduncle")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        dataset_dir_annotations = os.path.join(dataset_dir, 'annotations')
        
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        for root,dirs,files in os.walk( dataset_dir_annotations ):
            for filename in files:
                polygons = []
                labels = []
                with open(os.path.join(dataset_dir_annotations,filename)) as f:
                    data = json.load(f)
                    image_path = os.path.join(dataset_dir, 'images', data['imagePath'])
                    height, width = data['imageHeight'], data['imageWidth']
                    for polygon in data['shapes']:
                        labels.append(polygon['label'])
                        polygons.append(polygon['points'])

                # load_mask() needs the image size to convert polygons to masks. 
                self.add_image(
                    "tomato",
                    image_id=data['imagePath'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    label=labels,
                    polygons=polygons)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
     
        image_info = self.image_info[image_id]
        if image_info["source"] != "tomato":
            return super(TomatoDataset, self).load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(info["polygons"]):
            x = []
            y = []
            for xaxis,yaxis in p:
                x.append(xaxis)
                y.append(yaxis)
            class_ids.append(list(filter(lambda person: person["name"] == info["label"][i], self.class_info))[0]["id"])
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(y, x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "tomato":
            return info["path"]
        else:
            super(TomatoDataset, self).image_reference(image_id)


###################################################
#Evaluation code
###################################################

def eval_apply_mask(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def inference_on_image(model, image_path=None):
    assert image_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.test_image))
        # Read image
        image = skimage.io.imread(args.test_image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = eval_apply_mask(image, r['masks'])
        # Save output
        file_name = "tomato_test_1.png"
        skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'eval' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--test_image', required=False,
                        metavar="/path/to/testimages",
                        help='Images for evaluation')

#   python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print('test_image :', args.test_image)

# Configurations
    if args.command == "train":
        config = TomatoConfig()
    else:
        class InferenceConfig(TomatoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()



# Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

# Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_WEIGHTS_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

# Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

# Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = TomatoDataset()
        dataset_train.load_tomato(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = TomatoDataset()
        dataset_val.load_tomato(args.dataset, "val")
        dataset_val.prepare()

        

# *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')


    elif args.command == "eval":
        inference_on_image(model, image_path=args.test_image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))