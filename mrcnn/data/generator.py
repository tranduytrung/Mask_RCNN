import logging
import traceback
import numpy as np
from mrcnn import utils
from mrcnn.data.load_image import load_image_gt
from mrcnn.data.detection_targets import build_detection_targets
from mrcnn.data.rois import generate_random_rois
from mrcnn.data.rpn_targets import build_rpn_targets

def data_generator(dataset, config, shuffle=True, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = list(range(len(dataset)))
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = utils.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            image, image_meta, gt_class_ids, gt_boxes = \
                load_image_gt(dataset, config, image_id, augmentation=augmentation)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            # if random_rois:
            #     rpn_rois = generate_random_rois(
            #         image.shape, random_rois, gt_class_ids, gt_boxes)
            #     if detection_targets:
            #         rois, mrcnn_class_ids, mrcnn_bbox =\
            #             build_detection_targets(
            #                 rpn_rois, gt_class_ids, gt_boxes, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, config.IMAGE_SOURCES, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                # if random_rois:
                #     batch_rpn_rois = np.zeros(
                #         (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                #     if detection_targets:
                #         batch_rois = np.zeros(
                #             (batch_size,) + rois.shape, dtype=rois.dtype)
                #         batch_mrcnn_class_ids = np.zeros(
                #             (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                #         batch_mrcnn_bbox = np.zeros(
                #             (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)

            # If more instances than fits in the array, sub-sample from them.
            master_gt_boxes = gt_boxes[0]
            if master_gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(master_gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                master_gt_boxes = master_gt_boxes[ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = image
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :master_gt_boxes.shape[0]] = master_gt_boxes
            # if random_rois:
            #     batch_rpn_rois[b] = rpn_rois
            #     if detection_targets:
            #         batch_rois[b] = rois
            #         batch_mrcnn_class_ids[b] = mrcnn_class_ids
            #         batch_mrcnn_bbox[b] = mrcnn_bbox
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes]
                outputs = []

                # if random_rois:
                #     inputs.extend([batch_rpn_rois])
                #     if detection_targets:
                #         inputs.extend([batch_rois])
                #         # Keras requires that output and targets have the same number of dimensions
                #         batch_mrcnn_class_ids = np.expand_dims(
                #             batch_mrcnn_class_ids, -1)
                #         outputs.extend(
                #             [batch_mrcnn_class_ids, batch_mrcnn_bbox])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(image_id))
            logging.debug(traceback.format_exc())
            error_count += 1
            if error_count > 5:
                raise