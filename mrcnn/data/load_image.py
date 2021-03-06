import logging
import random
import numpy as np
from mrcnn import utils

def load_image_gt(dataset, config, image_id, augmentation=None):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.

    Returns:
    image: [height, width, IMAGE_SOURCES * 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, IMAGE_SOURCES, (y1, x1, y2, x2)]
    """
    # Load image and mask
    image, mask, class_ids = dataset[image_id]
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image.reshape((original_shape[0], original_shape[1], -1)),
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask.reshape((original_shape[0], original_shape[1], -1)),
        scale, padding, crop)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug        

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=__hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    mask = mask.reshape((mask.shape[0], mask.shape[1], -1, config.IMAGE_SOURCES))
    master_mask = mask[:, :, :, 0]
    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out => let dataset handle it
    # _idx = np.sum(master_mask, axis=(0, 1)) > 0
    # mask = mask[:, :, _idx, :]
    # class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, IMAGE_SOURCES, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale)

    return image, image_meta, class_ids, bbox

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale]                       # size=1
    )
    return meta

# Augmenters that are safe to apply to masks
# Some, such as Affine, have settings that make them unsafe, so always
# test your augmentation on masks
__MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                    "Fliplr", "Flipud", "CropAndPad",
                    "Affine", "PiecewiseAffine"]

def __hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in __MASK_AUGMENTERS