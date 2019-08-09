import csv
import os
import json
import numpy as np
import PIL.Image
import imgaug as ia
import imgaug.augmenters as iaa
from samples.m340.utils import resize_to, resize_bbox, bboxes2masks, normalize_rgb, normalize_d
from samples.m340.augmenter import AddFloat

def __read_meta__(meta_path, cat2eid):
    with open(meta_path, 'r', newline='') as f:
        attribute_signature = '# ATTRIBUTE = '
        end_signature = '# CSV_HEADER ='
        attrs: dict = None
        assert_error = 'Broken meta CSV file. Please use https://gitlab.com/vgg/via version 3.x.x'
        for i, data in enumerate(f):
            if data.startswith(attribute_signature):
                attrs = json.loads(data[len(attribute_signature):])

            if data.startswith(end_signature):
                break
        assert attrs is not None, assert_error

        # find list of categories
        iid2cat: dict = None
        attr_id: str = None
        for k, v in attrs.items():
            if v['aname'] == 'category':
                iid2cat = v['options']
                attr_id = k
                break

        assert iid2cat is not None, assert_error
        # convert internal id external id
        if cat2eid is None:
            cat2eid = {c: idx for idx, c in enumerate(iid2cat.values())}
        assert len(cat2eid) == len(
            iid2cat), 'the number of categories must match'
        iid2eid = {iid: cat2eid[cat] for iid, cat in iid2cat.items()}

        # record data samples
        samples = {}
        reader = csv.reader(f)
        for line in reader:
            sid = json.loads(line[1])[0].split('.')[0]
            x, y, w, h = json.loads(line[4])[1:]
            bbox = [y, x, y + h, x + w]
            cat = iid2eid[json.loads(line[5])[attr_id]]
            if sid in samples:
                sample = samples[sid]
                sample['bbox'].append(bbox)
                sample['cat'].append(cat)
            else:
                samples[sid] = {
                    'sid': sid,
                    'bbox': [bbox],
                    'cat': [cat]
                }
    return samples

class CapturedRGBDataset:
    def __init__(self, data_dir, meta_path, cat2eid=None, augmentations=True):
        self.data_dir = data_dir
        self.samples = __read_meta__(meta_path, cat2eid)
        self.sids = list(samples.keys())

        self.aug = None
        if augmentations:
            def sometimes(aug): return iaa.Sometimes(0.25, aug)
            self.aug = iaa.Sequential([
                iaa.Fliplr(0.5, name='Fliplr'),
                iaa.Flipud(0.5, name='Flipud'),
                sometimes(iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -20 to +20 percent (per axis)
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-30, 30),  # rotate by -45 to +45 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    cval=0,  # if mode is constant, use a cval between 0 and 255
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode='constant',
                    name='Affine'
                )),
                sometimes(iaa.Dropout(per_channel=True, p=0.01, name='Dropout')),
                sometimes(iaa.GaussianBlur(sigma=(0, 1.0), name='GaussianBlur')),
                sometimes(iaa.Add((-40, 40), name='Add')),
                sometimes(iaa.AddToHueAndSaturation((-20, 20), name='AddToHueAndSaturation')),
                sometimes(iaa.GammaContrast(gamma=(0.50, 2.50), name='GammaContrast'))
            ], random_order=True)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.sids[index]
        sample = self.samples[index]

        # load rgb data
        rgb_path = os.path.join(self.data_dir, f'{sample["sid"]}.png')
        pil_image = PIL.Image.open(rgb_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        rgb_data = np.array(pil_image)
        rgb_shape = np.shape(rgb_data)

        bbox = np.array(sample['bbox'], dtype=np.float32)
        masks = bboxes2masks(bbox, rgb_shape[:2]) # uint8

        # augmentation
        if self.aug is not None:
            det = self.aug.to_deterministic()
            rgb_data = det.augment_image(rgb_data)
            masks = det.augment_image(masks, hooks=ia.HooksImages(activator=_mask_activator_))

        # pil_im = PIL.Image.fromarray(rgb_data)
        # pil_im.save('rgb.png')
        # pil_im = PIL.Image.fromarray(masks[:, :, 0])
        # pil_im.save('mask.png')

        # normalize and cast to required types
        in_data = normalize_rgb(rgb_data)
        masks = masks[..., np.newaxis].astype(np.bool)
        cat = np.array(sample['cat'], dtype=np.int32)

        return in_data, masks, cat

    def __len__(self):
        return len(self.samples)

class CapturedRGBDDataset:
    def __init__(self, data_dir, meta_paths, cat2eid=None, augmentations=True):
        assert len(meta_paths) == 2, 'require 2 meta files, rgb meta and depth meta'
        rgb_samples = __read_meta__(meta_paths[0], cat2eid)
        d_samples = __read_meta__(meta_paths[1], cat2eid)
        samples = dict()

        # merge 2 info together
        for sid, rgb_meta in rgb_samples.items():
            # bbox for both rgb and d, (y1, x1, y2, x2, y1, x1, y2, x2)
            bboxes = []
            # for rgb
            rgb_bbox = np.array(rgb_meta['bbox'])
            rgb_cat = np.array(rgb_meta['cat'])
            rgb_acat = np.argsort(rgb_cat)
            rgb_scat = rgb_cat[rgb_acat] # sorted cat
            rgb_sbbox = rgb_bbox[rgb_acat] # sorted bbox

            # get coressponding depth
            if sid in d_samples:
                d_meta = d_samples[sid]
                d_bbox = np.array(d_meta['bbox'])
                d_cat = np.array(d_meta['cat'])
                d_acat = np.argsort(d_cat)
                d_scat = d_cat[d_acat] # sorted cat
                d_sbbox = d_bbox[d_acat] # sorted bbox
            else:
                d_scat = np.copy(rgb_scat)
                d_sbbox = np.copy(rgb_sbbox)

            # check for valid meta and pad
            for idx in range(len(rgb_scat)):
                if idx >= len(d_scat):
                    bboxes.append([*rgb_sbbox[idx], *rgb_sbbox[idx]])
                else:
                    assert d_scat[idx] == rgb_scat[idx]
                    bboxes.append([*rgb_sbbox[idx], *d_sbbox[idx]])

            samples[sid] = {
                'sid': sid,
                'bbox': bboxes,
                'cat': rgb_scat
            }

        self.samples = samples
        self.data_dir = data_dir
        self.sids = list(samples.keys())

        self.both_aug = None
        if augmentations:
            def sometimes(aug): return iaa.Sometimes(0.25, aug)
            self.both_aug = iaa.Sequential([
                iaa.Fliplr(0.5, name='Fliplr'),
                iaa.Flipud(0.5, name='Flipud'),
                sometimes(iaa.Affine(
                    # scale images to 80-120% of their size, individually per axis
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # translate by -20 to +20 percent (per axis)
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-30, 30),  # rotate by -45 to +45 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    order=[0, 1],
                    cval=0,  # if mode is constant, use a cval between 0 and 255
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    mode='constant',
                    name='Affine'
                ))
            ], random_order=True)

            self.rgb_aug = iaa.Sequential([
                sometimes(iaa.Dropout(per_channel=True, p=0.01, name='Dropout')),
                sometimes(iaa.GaussianBlur(sigma=(0, 1.0), name='GaussianBlur')),
                sometimes(iaa.Add((-40, 40), name='Add')),
                sometimes(iaa.AddToHueAndSaturation((-20, 20), name='AddToHueAndSaturation')),
                sometimes(iaa.GammaContrast(gamma=(0.50, 2.50), name='GammaContrast'))
            ], random_order=True)

            self.d_aug = AddFloat((-0.1, 0.4))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if isinstance(index, int):
            index = self.sids[index]
        sample = self.samples[index]

        # boxes
        np_bboxes = np.array(sample['bbox'], dtype=np.float32)
        # cats
        np_cats = np.array(sample['cat'], dtype=np.int32)

        # load rgb data
        rgb_path = os.path.join(self.data_dir, f'{sample["sid"]}.png')
        pil_image = PIL.Image.open(rgb_path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        rgb_data = np.array(pil_image)
        rgb_shape = np.shape(rgb_data)
        rgb_bbox = np_bboxes[:, :4]
        rgb_masks = bboxes2masks(rgb_bbox, rgb_shape[:2]) # uint8

        # load depth data
        d_path = os.path.join(self.data_dir, f'{sample["sid"]}.npy')
        np_d = np.load(d_path)
        d_shape = np_d.shape
        d_bbox = np_bboxes[:, 4:]
        # resize to rgb size
        d_data = resize_to(np_d, rgb_shape[:2])
        d_bbox = resize_bbox(d_bbox, d_shape[:2], rgb_shape[:2])
        d_masks = bboxes2masks(d_bbox, rgb_shape[:2]) # uint8

        # augmentation
        if self.both_aug is not None:
            rgb_data = self.rgb_aug.augment_image(rgb_data)
            d_data = self.d_aug.augment_image(d_data)
            
            det = self.both_aug.to_deterministic()
            rgb_data = det.augment_image(rgb_data)
            d_data = det.augment_image(d_data)
            rgb_masks = det.augment_image(rgb_masks, hooks=ia.HooksImages(activator=_mask_activator_))
            d_masks = det.augment_image(d_masks, hooks=ia.HooksImages(activator=_mask_activator_))

        # pil_im = PIL.Image.fromarray(rgb_data)
        # pil_im.save('rgb.png')
        # pil_im = PIL.Image.fromarray(rgb_masks[:, :, 0])
        # pil_im.save('rgb_mask.png')
        # pil_im = PIL.Image.fromarray(d_masks[:, :, 0])
        # pil_im.save('d_mask.png')

        # normalize and cast to required types
        in_data = np.concatenate((normalize_rgb(rgb_data), normalize_d(d_data)), axis=2)
        
        # ensure non zero masks
        _idx = np.sum(rgb_masks, axis=(0, 1)) > 0
        rgb_masks = rgb_masks[:, :, _idx]
        d_masks = d_masks[:, :, _idx]
        np_cats = np_cats[_idx]
        # for depth, fill with rgb mask
        _idx = np.sum(d_masks, axis=(0, 1)) == 0
        if np.sum(_idx):
            d_masks[:, :, _idx] = rgb_masks[:, :, _idx]
        
        masks = np.stack((rgb_masks, d_masks), axis=-1)
        return in_data, masks, np_cats



__MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                    "Fliplr", "Flipud", "CropAndPad",
                    "Affine", "PiecewiseAffine"]

def _mask_activator_(images, augmenter, parents, default):
    if augmenter.__class__.__name__ in __MASK_AUGMENTERS:
        return default
    return False

if __name__ == "__main__":
    dataset = CapturedRGBDDataset(
        'datasets/m340/train', ['datasets/m340/train/annotations.csv', 'datasets/m340/train/annotations_d.csv'])
    in_data, masks, cat = dataset[0]
    in_data, masks, cat = dataset[1]
    in_data, masks, cat = dataset[2]

