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

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT, colorstr, is_dir_writeable

from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms, CopyPaste, RandomFlip, Albumentations, RandomPerspective, RandomHSV
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label, get_img_files

from torch.utils.data import IterableDataset
import torchvision.transforms as T

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = '1.0.3'
WDS_SHARDED = True

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
            maxsize=int(os.environ.get('WDS_SHARDED_MAXSIZE', str(int(1 * 1e9)))), # 1GB
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
