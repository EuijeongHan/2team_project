"""
detr/dataset.py
===============
DETR / Deformable DETR 학습·평가용 Dataset & DataLoader

[DETRDataset]
    표준 DETR용. torchvision transforms로 정규화 후
    (cx, cy, w, h) 정규화 포맷의 target을 반환합니다.

[DeformableDetrDataset]
    Deformable DETR용. HuggingFace DeformableDetrImageProcessor와
    함께 사용합니다. PIL Image + COCO annotation을 그대로 반환하고,
    collate_fn 안에서 processor가 일괄 처리합니다.

사용법:
    # 표준 DETR
    from src.models.detr.dataset import get_detr_loaders
    train_loader, val_loader, idx2cat = get_detr_loaders(base_dir=BASE_DIR)

    # Deformable DETR
    from src.models.detr.dataset import get_deformable_loaders
    train_loader, val_loader, idx2cat = get_deformable_loaders(base_dir=BASE_DIR)
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


class DETRDataset(Dataset):
    """
    DETR 학습용 PyTorch Dataset.

    COCO JSON을 읽어 이미지당 (image_tensor, target) 쌍을 반환합니다.

    BBox 포맷 변환:
        COCO  : [x_min, y_min, w, h]  (픽셀 절댓값)
        DETR  : [cx, cy, w, h]        (0~1 정규화)

    레이블:
        원본 category_id → 0-based 연속 인덱스 (cat2idx)
        역매핑: idx2cat = {v: k for k, v in cat2idx.items()}

    Args:
        json_path   : letterbox 처리된 COCO JSON 경로
        img_dir     : letterbox 이미지 폴더 경로
        target_size : 이미지 해상도 (Letterbox 규격과 일치해야 함)
        transforms  : torchvision transforms (None이면 기본 ImageNet 정규화 적용)
    """

    def __init__(self, json_path, img_dir, target_size=800, transforms=None):
        with open(json_path, 'r') as f:
            coco = json.load(f)

        self.img_dir     = img_dir
        self.target_size = target_size

        self.images      = {img['id']: img for img in coco['images']}
        cats             = sorted([c['id'] for c in coco['categories']])
        self.cat2idx     = {c: i for i, c in enumerate(cats)}
        self.num_classes = len(cats)
        self.img_ids     = list(self.images.keys())

        self.annots = {img_id: [] for img_id in self.img_ids}
        for ann in coco['annotations']:
            if ann['image_id'] in self.annots:
                self.annots[ann['image_id']].append(ann)

        self.transforms = transforms or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]

        image = Image.open(
            os.path.join(self.img_dir, img_info['file_name'])
        ).convert('RGB')
        W, H = image.size  # Letterbox 후엔 target_size x target_size

        boxes, labels = [], []
        for ann in self.annots[img_id]:
            x, y, w, h = ann['bbox']
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            boxes.append([cx, cy, w / W, h / H])
            labels.append(self.cat2idx[ann['category_id']])

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([img_id]),
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def get_detr_loaders(base_dir, target_size=800, batch_size=4, num_workers=2):
    """
    train / val DataLoader와 idx2cat 역매핑을 반환합니다.

    Args:
        base_dir    : letterbox 산출물이 있는 데이터 루트
        target_size : Letterbox 해상도 (800 or 1024 등)
        batch_size  : 배치 크기 (고해상도일수록 줄여야 함)
        num_workers : DataLoader 워커 수

    Returns:
        train_loader, val_loader, idx2cat
    """
    suffix = f'_{target_size}' if target_size != 800 else ''

    train_json = os.path.join(base_dir, f'train_letterbox{suffix}.json')
    val_json   = os.path.join(base_dir, f'val_letterbox{suffix}.json')
    train_img  = os.path.join(base_dir, f'letterbox_images{suffix}', 'train')
    val_img    = os.path.join(base_dir, f'letterbox_images{suffix}', 'val')

    # 800px이면 기존 산출물 그대로 사용
    if target_size == 800:
        train_json = os.path.join(base_dir, 'train_letterbox.json')
        val_json   = os.path.join(base_dir, 'val_letterbox.json')
        train_img  = os.path.join(base_dir, 'letterbox_images', 'train')
        val_img    = os.path.join(base_dir, 'letterbox_images', 'val')

    train_ds = DETRDataset(train_json, train_img, target_size=target_size)
    val_ds   = DETRDataset(val_json,   val_img,   target_size=target_size)

    idx2cat = {v: k for k, v in train_ds.cat2idx.items()}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, collate_fn=collate_fn)

    print(f'✅ target_size : {target_size}px')
    print(f'✅ train       : {len(train_ds)}장')
    print(f'✅ val         : {len(val_ds)}장')
    print(f'✅ num_classes : {train_ds.num_classes}')

    return train_loader, val_loader, idx2cat


# ---------------------------------------------------------------------------
# Deformable DETR용 Dataset
# ---------------------------------------------------------------------------
class DeformableDetrDataset(Dataset):
    """
    Deformable DETR 학습용 Dataset.

    PIL Image와 COCO 포맷 annotation을 그대로 반환합니다.
    변환·정규화는 collate_fn 안의 DeformableDetrImageProcessor가 담당합니다.

    반환 포맷:
        image  : PIL.Image (RGB)
        target : {
            "image_id"   : int,
            "annotations": [{"id", "image_id", "category_id", "bbox", "area", "iscrowd"}, ...]
        }
        → processor(images=[image], annotations=[target]) 형태로 배치 처리

    레이블:
        원본 category_id를 0-based cat2idx로 변환하여 저장합니다.
        역매핑: idx2cat = {v: k for k, v in cat2idx.items()}
    """

    def __init__(self, json_path, img_dir):
        with open(json_path, 'r') as f:
            coco = json.load(f)

        self.img_dir = img_dir
        self.images  = {img['id']: img for img in coco['images']}

        cats             = sorted([c['id'] for c in coco['categories']])
        self.cat2idx     = {c: i for i, c in enumerate(cats)}
        self.num_classes = len(cats)
        self.img_ids     = list(self.images.keys())

        self.annots = {img_id: [] for img_id in self.img_ids}
        for ann in coco['annotations']:
            if ann['image_id'] in self.annots:
                self.annots[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]

        image = Image.open(
            os.path.join(self.img_dir, img_info['file_name'])
        ).convert('RGB')

        annotations = []
        for i, ann in enumerate(self.annots[img_id]):
            annotations.append({
                'id':          i,
                'image_id':    img_id,
                'category_id': self.cat2idx[ann['category_id']],  # 0-based
                'bbox':        ann['bbox'],                        # [x, y, w, h]
                'area':        ann['bbox'][2] * ann['bbox'][3],
                'iscrowd':     0,
            })

        target = {
            'image_id':    img_id,
            'annotations': annotations,
        }

        return image, target


def get_deformable_loaders(base_dir, batch_size=4, num_workers=2,
                           model_name='SenseTime/deformable-detr'):
    """
    Deformable DETR용 train / val DataLoader와 idx2cat을 반환합니다.

    DeformableDetrImageProcessor가 collate_fn 안에서 이미지 전처리를 담당하며
    pixel_values, pixel_mask, labels 키를 가진 dict를 배치로 반환합니다.

    Args:
        base_dir   : letterbox 산출물이 있는 데이터 루트
        batch_size : 배치 크기 (Deformable DETR은 표준 DETR보다 메모리 효율적)
        num_workers: DataLoader 워커 수
        model_name : HuggingFace 모델 ID

    Returns:
        train_loader, val_loader, idx2cat
    """
    from transformers import DeformableDetrImageProcessor

    processor = DeformableDetrImageProcessor.from_pretrained(model_name)

    train_json = os.path.join(base_dir, 'train_letterbox.json')
    val_json   = os.path.join(base_dir, 'val_letterbox.json')
    train_img  = os.path.join(base_dir, 'letterbox_images', 'train')
    val_img    = os.path.join(base_dir, 'letterbox_images', 'val')

    train_ds = DeformableDetrDataset(train_json, train_img)
    val_ds   = DeformableDetrDataset(val_json,   val_img)

    idx2cat = {v: k for k, v in train_ds.cat2idx.items()}

    def deformable_collate_fn(batch):
        images, targets = zip(*batch)
        encoding = processor(
            images=list(images),
            annotations=list(targets),
            return_tensors='pt',
        )
        return encoding, list(targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,
                              collate_fn=deformable_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers,
                              collate_fn=deformable_collate_fn)

    print(f'✅ train       : {len(train_ds)}장')
    print(f'✅ val         : {len(val_ds)}장')
    print(f'✅ num_classes : {train_ds.num_classes}')

    return train_loader, val_loader, idx2cat
