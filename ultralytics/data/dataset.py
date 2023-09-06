# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
from itertools import repeat, islice
from multiprocessing.pool import ThreadPool
from multiprocessing import Lock
from pathlib import Path
import copy
import torchvision.transforms as transforms
import PIL.Image
import torchdata

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import webdataset as wds
import io
import json
import os
from functools import partial

from ultralytics.utils import LOCAL_RANK, DEFAULT_CFG, NUM_THREADS, TQDM_BAR_FORMAT, colorstr, is_dir_writeable

from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms, CopyPaste, RandomFlip, Albumentations, RandomPerspective, RandomHSV
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label, get_img_files

from torch.utils.data import IterableDataset
import torchvision.transforms as T

from ultralytics.utils.ops import segments2boxes
import random
import math
from torchdata.datapipes.iter import IterDataPipe, FileLister, FileOpener, LineReader, ParagraphAggregator, Mapper, Dropper, Zipper, RoutedDecoder, Shuffler, ShardingFilter, UnZipper, Header, Batcher, Collator
from torch.utils.data.datapipes.utils.decoder import imagehandler, mathandler
from typing import List, TypeVar, Tuple, Optional, Iterator

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = '1.0.3'
WDS_SHARDED = True

T_co = TypeVar('T_co', covariant=True)

class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path('./labels.cache')):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(f'{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}')

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        assert len(labels), f'No valid labels found, please check your dataset. {HELP_URL}'
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            raise ValueError(f'All labels empty in {cache_path}, can not start training without labels. {HELP_URL}')
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments', [])
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch

class RandomMultiBufferIterDataPipe(IterDataPipe[T_co]):
    """
    Buffer samples and return m tuple of n samples, the first one being the last yield by the datapipe,
    and the others randomly picked by the buffer.
    When buffer is full, samples are randomly replaced by the last yield by the datapipe.
    """

    datapipe: IterDataPipe[T_co]
    buffer_size: int
    n: int
    m: int
    _buffer: List[T_co]
    _seed: Optional[int]
    _rng: random.Random
    
    def __init__(self, datapipe: IterDataPipe[T_co], buffer_size: int = 1_000, m: int = 2, n: int = 4, seed: Optional[int] = None):
        super().__init__()
        assert buffer_size > 0, "buffer_size should be larger than 0"
        assert n > 0, "n should be larger than 0"
        self.datapipe = datapipe
        self.buffer_size = buffer_size
        self.n = n
        self.m = m
        self._buffer: List[T_co] = []
        self._seed = seed
        self._rng = random.Random(self._seed)

    def __iter__(self) -> Iterator[Tuple[Tuple[T_co, ...], ...]]:
        for x in self.datapipe:
            samples = []
            for i in range(self.m):
                s = (x,) if i == 0 else (self._buffer[self._rng.randint(0, len(self._buffer) - 1)],)
                if not len(self._buffer):
                    samples.append(s + tuple(x for i in range(self.n-1)))
                else:
                    samples.append(s + tuple(self._buffer[self._rng.randint(0, len(self._buffer) - 1)] for i in range(self.n-1)))
                if len(self._buffer) == self.buffer_size:
                    self._buffer[self._rng.randint(0, len(self._buffer) - 1)] = x
                else:
                    self._buffer.append(x)
            yield copy.deepcopy(tuple(samples))

    def reset(self) -> None:
        self._buffer.clear()

    def __getstate__(self):
        state = (
            self.datapipe,
            self.buffer_size,
            self.m,
            self.n,
            self._seed,
            self._buffer,
            self._rng.getstate(),
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.buffer_size,
            self.m,
            self.n,
            self._seed,
            self._buffer,
            rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    def __del__(self):
        self._buffer.clear()

class Mosaic:

    def __init__(self, imgsz: int, n: int = 4, p: float = 1.0):
        assert n in (4, 9), "grid must be equal to 4 or 9"
        assert 0 <= p <= 1.0, f'The probability should be in range [0, 1], but got {p}.'
        self.imgsz = imgsz
        self.n = n
        self.p = p
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height

    def _update_labels(self, labels, padw, padh):
        """Update labels."""
        nh, nw = labels['img'].shape[:2]
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(nw, nh)
        labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        final_labels = {
            'im_file': mosaic_labels[0]['im_file'],
            'ori_shape': mosaic_labels[0]['ori_shape'],
            'resized_shape': (imgsz, imgsz),
            'cls': np.concatenate(cls, 0),
            'instances': Instances.concatenate(instances, axis=0),
            'mosaic_border': self.border}  # final_labels
        final_labels['instances'].clip(imgsz, imgsz)
        good = final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls'] = final_labels['cls'][good]
        return final_labels

    def _mosaic4(self, labels):
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i, labels_patch in enumerate(labels):
            labels_patch = copy.deepcopy(labels_patch)
            # Load image
            img = labels_patch['img']
            h, w = labels_patch['resized_shape']

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        return final_labels

    def _mosaic9(self, labels):
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1  # height, width previous
        for i, labels_patch in enumerate(labels):
            # Load image
            img = labels_patch['img']
            h, w = labels_patch['resized_shape']

            # Place img in img9
            if i == 0:  # center
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                h0, w0 = h, w
                c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
            elif i == 1:  # top
                c = s, s - h, s + w, s
            elif i == 2:  # top right
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:  # right
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:  # bottom right
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:  # bottom
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:  # bottom left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Image
            img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous for next iteration

            # Labels assuming imgsz*2 mosaic size
            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)

        final_labels['img'] = img9[-self.border[0]:self.border[0], -self.border[1]:self.border[1]]
        return final_labels

    def __call__(self, labels):
        assert len(labels) == self.n, f"number of samples in feeding datapipes {len(labels)} is not consistent with grid size {self.n}"
        if random.uniform(0, 1) > self.p:
            return labels[0]
        else:
            if self.n == 4:
                return self._mosaic4(labels)
            elif self.n == 9:
                return self._mosaic9(labels)


class MixUp:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, labels):
        assert len(labels) == 2, "mixup expect 2 labels at the same time"
        labels, labels2 = labels
        labels = copy.deepcopy(labels)
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels['img'] = (labels['img'] * r + labels2['img'] * (1 - r)).astype(np.uint8)
        labels['instances'] = Instances.concatenate([labels['instances'], labels2['instances']], axis=0)
        labels['cls'] = np.concatenate([labels['cls'], labels2['cls']], 0)
        return labels


class YOLOIterDataset(IterDataPipe):

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0,
                 data=None,
                 use_segments=False,
                 use_keypoints=False,
                 shuffle_buffer_size=1e5):
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.classes = classes
        # self.im_files = self.get_img_files(self.img_path)
        # self.labels = self.get_labels()
        # self.update_labels(include_class=classes)  # single_cls and include_class
        # self.ni = len(self.labels)  # number of images
        self.ni = len(os.listdir(img_path)) # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        self.shuffle_buffer_size = shuffle_buffer_size
        if self.rect:
            raise NotImplementedError("Rectangle mode not supported")
            # assert self.batch_size is not None
            # self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size #TODO: Implement mosaic later
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0 #TODO: Implement mosaic later

        # Cache stuff
        # if cache == 'ram' and not self.check_cache_ram():
        #     cache = False
        # self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni #TODO: needed?
        # self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files] #TODO: needed?
        # if cache:
        #     self.cache_images(cache)

        # Transforms
        # self.transforms = self.build_transforms(hyp=hyp)
        self.datapipe = self.build_datapipe(hyp=hyp)

    #TODO: From BaseDataset
    def __len__(self):
        return self.ni
    
    # def v8_transforms(self, imgsz, hyp, stretch=False):
    #     """Convert images to a size suitable for YOLOv8 training."""
    #     pre_transform = Compose([
    #         Mosaic(self, imgsz=imgsz, p=hyp.mosaic),
    #         CopyPaste(p=hyp.copy_paste),
    #         RandomPerspective(
    #             degrees=hyp.degrees,
    #             translate=hyp.translate,
    #             scale=hyp.scale,
    #             shear=hyp.shear,
    #             perspective=hyp.perspective,
    #             pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
    #         )])
    #     flip_idx = self.data.get('flip_idx', [])  # for keypoints augmentation
    #     if self.use_keypoints:
    #         kpt_shape = self.data.get('kpt_shape', None)
    #         if len(flip_idx) == 0 and hyp.fliplr > 0.0:
    #             hyp.fliplr = 0.0
    #             LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
    #         elif flip_idx and (len(flip_idx) != kpt_shape[0]):
    #             raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    #     return Compose([
    #         pre_transform,
    #         MixUp(self, pre_transform=pre_transform, p=hyp.mixup),
    #         Albumentations(p=1.0),
    #         RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
    #         RandomFlip(direction='vertical', p=hyp.flipud),
    #         RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms

    # # TODO: From YOLODataset
    # def build_transforms(self, hyp=None):
    #     """Builds and appends transforms to the list."""
    #     if self.augment:
    #         hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
    #         hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
    #         transforms = v8_transforms(self, self.imgsz, hyp)
    #     else:
    #         transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
    #     transforms.append(
    #         Format(bbox_format='xywh',
    #                normalize=True,
    #                return_mask=self.use_segments,
    #                return_keypoint=self.use_keypoints,
    #                batch_idx=True,
    #                mask_ratio=hyp.mask_ratio,
    #                mask_overlap=hyp.overlap_mask))
    #     return transforms

    # TODO: Implement close_mosaic() -> reference Mosaic / CopyPaste / MixUp and set p=0.0
    def close_mosaic(self, hyp):
        # TODO: Duplicated from YOLODataset, to replace
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic

    def buid_datapipe(self, hyp):
        labels_dirpath = self.img_path.replace('/images/', '/labels/')

        images_dp = FileLister(self.img_path) # (filepath)
        labels_dp = FileLister(labels_dirpath) # (filepath)
        dp = Zipper(images_dp, labels_dp)
        if self.fraction != 1.0:
            dp = Header(dp, int(self.fraction*self.ni))
        dp = Shuffler(dp, buffer_size=self.shuffle_buffer_size)
        dp = ShardingFilter(dp)
        images_dp, labels_dp = UnZipper(dp, sequence_length=2)

        images_dp = FileOpener(images_dp, mode="b") # (filepath, stream)
        images_dp = RoutedDecoder(images_dp, imagehandler('rgb8')) # (filepath, numpy 255 uint8 image)

        labels_dp = FileOpener(labels_dp) # (filepath, stream)
        labels_dp = LineReader(labels_dp, strip_newline=True) # (filepath, line)
        labels_dp = ParagraphAggregator(labels_dp, joiner=lambda ls: ls) # (filepath, lines)

        dp = Zipper(images_dp, labels_dp) # ((image filepath, numpy 255 uint8 image), (label filepath, lines))
        keypoint, imgsz, augment, include_class, single_cls = self.use_keypoints, self.imgsz, self.augment, self.classes, self.single_cls or False
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        num_cls = len(self.data['names'])
        dp = Mapper(dp, partial(decode, keypoint, imgsz, augment, include_class, single_cls, nkpt, ndim, num_cls)) # labels dict
        if self.augment:
            dp = RandomMultiBufferIterDataPipe(dp, buffer_size=100, m=2, n=4) # (labels dict)*n
            dp1, dp2 = UnZipper(dp, sequence_length=2)
            # TODO: Implement close_mosaic() -> reference Mosaic / CopyPaste / MixUp and set p=0.0
            def pre_transform(dp, hyp, imgsz, augment):
                dp = Mapper(dp, Mosaic(n=4, imgsz=imgsz, p=hyp.mosaic if augment else 0.0)) # labels dict
                dp = Mapper(dp, CopyPaste(p=hyp.copy_paste)) # labels dict
                dp = Mapper(dp, RandomPerspective(
                            degrees=hyp.degrees,
                            translate=hyp.translate,
                            scale=hyp.scale,
                            shear=hyp.shear,
                            perspective=hyp.perspective,
                            # pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)), # stretch = False
                            pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
                        )) # labels dict
                return dp

            dp = Zipper(pre_transform(dp1, hyp, imgsz, augment), pre_transform(dp2, hyp, imgsz, augment))
            dp = Mapper(dp, MixUp(hyp.mixup))
            dp = Mapper(dp, Albumentations(p=1.0))
            dp = Mapper(dp, RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v))
            dp = Mapper(dp, RandomFlip(direction='vertical', p=hyp.flipud))
            # TODO: Implement flip_idx logic from v8_transforms
            # dp = Mapper(dp, RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx))
        else:
            dp = Mapper(dp, LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False))
        dp = Mapper(dp, Format(bbox_format='xywh',
                        normalize=True,
                        return_mask=self.use_segments,
                        return_keypoint=self.use_keypoints,
                        batch_idx=True,
                        mask_ratio=hyp.mask_ratio,
                        mask_overlap=hyp.overlap_mask))

        dp = Batcher(dp, self.batch_size)
        dp = Collator(dp, YOLODataset.collate_fn)

        return dp

    def __iter__(self):
        # TODO: CONTINUE HERE
        yield from self.datapipe

def decode(keypoint, imgsz, augment, include_class, single_cls, nkpt, ndim, num_cls, entry):
    (image_filepath, cv_image), (label_filepath, lines) = entry
    # TODO: implement 'verify_image_label'
    segments = []
    shape = cv_image.shape[:2] # hw
    lb = [x.split() for x in lines if len(x)]
    if any(len(x) > 6 for x in lb) and (not keypoint):  # is segment
        classes = np.array([x[0] for x in lb], dtype=np.float32)
        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
    lb = np.array(lb, dtype=np.float32)
    nl = len(lb)
    if nl:
        if keypoint:
            assert lb.shape[1] == (5 + nkpt * ndim), f'labels require {(5 + nkpt * ndim)} columns each'
            assert (lb[:, 5::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
            assert (lb[:, 6::ndim] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
        else:
            assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
            assert (lb[:, 1:] <= 1).all(), \
                f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
            assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
        # All labels
        max_cls = int(lb[:, 0].max())  # max label count
        assert max_cls <= num_cls, \
            f'Label class {max_cls} exceeds dataset class count {num_cls}. ' \
            f'Possible class labels are 0-{num_cls - 1}'
        _, i = np.unique(lb, axis=0, return_index=True)
        if len(i) < nl:  # duplicate row check
            lb = lb[i]  # remove duplicates
            if segments:
                segments = [segments[x] for x in i]
            msg = f'WARNING ⚠️ {label_filepath}: {nl - len(i)} duplicate labels removed'
    else:
        ne = 1  # label empty
        lb = np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32) if keypoint else np.zeros(
            (0, 5), dtype=np.float32)
    if keypoint:
        keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
        if ndim == 2:
            kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
            kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
            kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
            keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)
    lb = lb[:, :5]
    # nm = 0 # label not missing
    # nf = 1 # label found
    # nc = 0 # not corrupted
    # return image_filepath, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    label = dict(
        im_file=image_filepath,
        shape=shape,
        cls=lb[:, 0:1], # n, 1
        bboxes=lb[:, 1:], # n, 4
        segments=segments,
        keypoints=keypoints,
        normalized=True,
        bbox_format='xywh'
    )
    # TODO: implement update_labels
    include_class_array = np.array(include_class).reshape(1, -1)
    if include_class is not None:
        cls = label['cls']
        bboxes = label['bboxes']
        segments = label['segments']
        keypoints = label['keypoints']
        j = (cls == include_class_array).any(1)
        label['cls'] = cls[j]
        label['bboxes'] = bboxes[j]
        if segments:
            label['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
        if keypoints is not None:
            label['keypoints'] = keypoints[j]
    if single_cls:
        label['cls'][:, 0] = 0
    # TODO: get_image_and_label
    label.pop('shape', None)  # shape is for rect, remove it
    ## TODO: load_image
    im = cv_image
    h0, w0 = im.shape[:2]  # orig hw
    r = imgsz / max(h0, w0)
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (augment or r > 1) else cv2.INTER_AREA
        im = cv2.resize(im, (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz)),
                        interpolation=interp)
    label['img'] = im
    label['ori_shape'] = (h0, w0)
    label['resized_shape'] = im.shape[:2]
    # TODO: Add img, ori_shape, resize_shape to buffer for mosaic & mixup
    label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                            label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
    # TODO: implement update_labels_info
    bboxes = label.pop('bboxes')
    segments = label.pop('segments', [])
    keypoints = label.pop('keypoints', None)
    bbox_format = label.pop('bbox_format')
    normalized = label.pop('normalized')
    label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
    return label

# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False, prefix=''):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f'{prefix}: ') if prefix else ''
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.torch_transforms = classify_transforms(args.imgsz)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f'{self.prefix}Scanning {self.root}...'
        path = Path(self.root).with_suffix('.cache')  # *.cache file path

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop('results')  # found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f'{desc} {nf} images, {nc} corrupt'
                tqdm(None, desc=d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)
                if cache['msgs']:
                    LOGGER.info('\n'.join(cache['msgs']))  # display warnings
            return samples

        # Run scan if *.cache retrieval failed
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = tqdm(results, desc=desc, total=len(self.samples), bar_format=TQDM_BAR_FORMAT)
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f'{desc} {nf} images, {nc} corrupt'
            pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        x['hash'] = get_hash([x[0] for x in self.samples])
        x['results'] = nf, nc, len(samples), samples
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return samples


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')


def load_wds_info_file(info_filepath: Path):
    return load_dataset_cache_file(info_filepath)


def save_wds_info_file(prefix, info_filepath, cache):
    cache['version'] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(info_filepath.parent):
        if info_filepath.exists():
            info_filepath.unlink()
        np.save(str(info_filepath), cache)  # save cache for next time
        info_filepath.with_suffix('.wdsinfo.npy').rename(info_filepath)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {info_filepath}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {info_filepath.parent} is not writeable, cache not saved.')


def wds_encode_sample(sink, lock, sample):
    s = {
        '__key__': Path(sample['im_file']).stem
    }
    img = sample.pop('img')

    _, im_buf_arr = cv2.imencode('.png', cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
    s['png'] = im_buf_arr.tobytes()

    for k in ['cls', 'bboxes', 'keypoints', 'batch_idx']:
        if k in sample.keys() and isinstance(sample[k], torch.Tensor):
            sample[k] = sample[k].tolist()
    for k in ['resized_shape']:
        if k in sample.keys() and isinstance(sample[k], np.ndarray):
            sample[k] = sample[k].tolist()
    s['json'] = bytes(json.dumps(sample), 'utf-8')

    lock.acquire()
    try:
        sink.write(s)
    finally:
        lock.release()


def _cache_wds(info_filepath, prefix, cfg, img_path, batch, data, mode, rect, stride):

    dataset = YOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=False,  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=False, # Force to false
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == 'train' else 0.5,
        prefix=colorstr(f'{mode}: '),
        use_segments=cfg.task == 'segment',
        use_keypoints=cfg.task == 'pose',
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == 'train' else 1.0)

    if WDS_SHARDED:
        # TODO: move sharded to a subdir
        pattern = info_filepath.parent / (info_filepath.stem + "-%09d.tar")
        sink = wds.ShardWriter(
            pattern=pattern.as_posix(),
            maxcount=int(os.environ.get('WDS_SHARDED_MAXCOUNT', str(int(1e3)))),
            maxsize=int(os.environ.get('WDS_SHARDED_MAXSIZE', str(int(16 * 1e9)))), # 16GB
            encoder=False)
    else:
        tar_filepath = info_filepath.parent / (info_filepath.stem + ".tar")
        if tar_filepath.exists():
            os.unlink(tar_filepath.as_posix())
        sink = wds.TarWriter(tar_filepath.as_posix(), encoder=False)

    desc = f'{prefix}Caching as webdataset...'
    total = len(dataset)
    pbar = tqdm(dataset, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
    lock = Lock()

    with ThreadPool(int(os.environ.get('WDS_THREAD_POOL_SIZE', os.cpu_count()))) as pool:
        for _ in pool.imap(func=partial(wds_encode_sample, sink, lock), iterable=pbar):
            pass
    pbar.close()

    if WDS_SHARDED:
        shards = sink.shard - 1 if sink.count == 0 else sink.shard
    else:
        shards = None

    sink.close()

    return total, shards


def wds_decoder(s):

    sample = json.load(s['.json'])

    img = np.asarray(bytearray(s['.png'].read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    sample['img'] = img

    for k in ['resized_shape', 'cls', 'bboxes', 'keypoints', 'batch_idx']:
        if k in sample:
            sample[k] = np.array(sample[k])

    return sample

def wds_format(sample):

    img = np.ascontiguousarray(sample['img'].transpose(2, 0, 1)[::-1])
    sample['img'] = torch.from_numpy(img)

    for k in ['cls', 'bboxes', 'keypoints', 'batch_idx']:
        if k in sample:
            sample[k] = torch.from_numpy(sample[k])

    return sample


def single_node_only(src, group=None):
    # rank, world_size, worker, num_workers = utils.pytorch_worker_info(group=group)
    # if world_size > 1:
    #     raise ValueError(
    #         "you need to add an explicit nodesplitter to your input pipeline for multi-node training"
    #     )
    for s in src:
        yield s


def wds_update_labels(sample):
    sample['bbox_format'] = 'xywh'
    sample['normalized'] = True
    return YOLODataset.update_labels_info(None, sample)


def build_transforms(hyp, data):
    """Builds and appends transforms to the list."""
    """Convert images to a size suitable for YOLOv8 training."""
    stretch = False
    use_segments = hyp.task == 'segment'
    use_keypoints = hyp.task == 'pose'
    imgsz = hyp.imgsz
    pre_transform = Compose([
        # Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
        CopyPaste(p=hyp.copy_paste),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
        )])
    flip_idx = data.get('flip_idx', [])  # for keypoints augmentation
    if use_keypoints:
        kpt_shape = data.get('kpt_shape', None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f'data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}')

    transforms = Compose([
        pre_transform,
        # MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr, flip_idx=flip_idx)])  # transforms

    transforms.append(
        Format(bbox_format='xywh',
                normalize=True,
                return_mask=use_segments,
                return_keypoint=use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask))
    return transforms


def split_by_node(src, group=None):
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info(group=group)
    LOGGER.info(f"########## splitter: rank {rank} / {world_size}")
    if world_size > 1:
        for s in islice(src, rank*num_workers+worker, None, world_size*num_workers):
            yield s
    else:
        for s in src:
            yield s


def get_wds_dataset(cfg, img_path, batch, data, mode, rect, stride):

    prefix = colorstr(f'{mode}: ')

    check_cache = os.environ.get('WDS_CHECK_CACHE', '1').lower() in ('yes', 'true', '1')

    if check_cache:
        im_files = get_img_files(img_path, cfg.fraction, prefix)
        label_files = img2label_paths(im_files)
        cache_hash = get_hash(label_files + im_files)
        info_filepath = Path(label_files[0]).parent.with_suffix('.wdsinfo')
    else:
        LOGGER.warn("Skip webdataset cache checking")
        info_filepath = Path(img_path.replace('/images/', '/labels/')).with_suffix('.wdsinfo')
        LOGGER.info(f"Webdataset info filepath: {info_filepath.as_posix()}")
    try:
        info = load_wds_info_file(info_filepath)
        assert info['version'] == DATASET_CACHE_VERSION  # matches current version
        assert not check_cache or info['hash'] == cache_hash  # identical hash
    except (FileNotFoundError, AssertionError, AttributeError):
        LOGGER.info("Start webdataset caching ...")
        if not check_cache:
            im_files = get_img_files(img_path, cfg.fraction, prefix)
            label_files = img2label_paths(im_files)
            cache_hash = get_hash(label_files + im_files)
        total, shards = _cache_wds(info_filepath, prefix, cfg, img_path, batch, data, mode, rect, stride)
        info = {
            'hash': cache_hash,
            'total': total,
            'shards': shards
        }
        save_wds_info_file(prefix, info_filepath, info)

    LOGGER.info("Create webdataset ...")

    # args = []
    # if info['shards'] is not None:
    #     url = info_filepath.parent.as_posix() + "/" + (info_filepath.stem + ("-{%09d..%09d}.tar" % (0, info['shards']-1)))
    #     args.append(wds.ResampledShards(url))
    # else:
    #     url =  (info_filepath.parent / (info_filepath.stem + ".tar")).as_posix()
    #     args.append(wds.SimpleShardList([url]))
    # if mode == "train":
    #     pool_size = min(int(os.environ.get('WDS_SHUFFLE_POOL_SIZE', '1000')), info['total'])
    #     args.append(wds.shuffle(pool_size))
    # # args.append(wds.split_by_node)
    # args.append(split_by_node)
    # args.append(wds.tarfile_to_samples())
    # # args.append(wds.decode('torchrgb8'))
    # # args.append(wds.to_tuple('png', 'json'))
    # args.append(wds.map(wds_decoder))
    
    # dataset = wds.DataPipeline(*args)
    # dataset = wds.with_epoch(dataset, info['total'])
    # dataset = wds.with_length(dataset, info['total'])
    # dataset.collate_fn = YOLODataset.collate_fn
    # return dataset
        
    # if info['shards'] is not None:
    #     url = info_filepath.parent.as_posix() + "/" + (info_filepath.stem + ("-{%09d..%09d}.tar" % (0, info['shards']-1)))
    #     dataset = wds.WebDataset(url, shardshuffle=mode == "train", nodesplitter=split_by_node)
    # else:
    #     tar_filepath =  info_filepath.parent / (info_filepath.stem + ".tar")
    #     dataset = wds.WebDataset(tar_filepath.as_posix())
    # dataset = dataset.with_length(info['total'])
    # if mode == "train":
    #     pool_size = min(int(os.environ.get('WDS_SHUFFLE_POOL_SIZE', '1000')), info['total'])
    #     dataset = dataset.shuffle(pool_size)
    # dataset = dataset.map(wds_decoder)
    # dataset.collate_fn = YOLODataset.collate_fn
    # return dataset

    if info['shards'] is not None:
        urls = [(info_filepath.parent / f"{info_filepath.stem}-{i:09d}.tar").as_posix() for i in range(info['shards'])]
    else:
        urls =  [info_filepath.parent / (info_filepath.stem + ".tar")]

    dp = torchdata.datapipes.iter.FileOpener(urls, mode="b")
    dp = dp.load_from_tar(length=info['total'])
    dp = dp.webdataset()
    if mode == "train":
        dp = dp.shuffle()
    dp = dp.sharding_filter()
    dp = dp.map(wds_decoder)
    if mode == "train":
        dp = dp.map(wds_update_labels)
        transforms = build_transforms(cfg, data)
        dp = dp.map(transforms)
    else:
        dp = dp.map(wds_format)
    dp.collate_fn = YOLODataset.collate_fn
    return dp





# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()
