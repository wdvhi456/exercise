import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
from typing import Union
from datasetsmodule.meta import PairedImageWithLightnessInput
from transform.pipelines import load_transforms



##opencv读图返回RGB
def read_image_cv2(path: Union[str, Path]) -> np.ndarray:
    """Read an image from a path."""
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

##实现从数据集中依次取预处理过的数据
class FSD(Dataset):
    MAPPING = {
        'train': 'train',
        'val': 'val',
        'test': 'test',
    }
    def __init__(
        self,
        root: Path,
        pair_transform: Callable,
        split: str = "train",
        preload: bool = False,
        start_idx: int = 0,
        limit: Optional[int] = None,
    ):
        path = root / FSD.MAPPING[split]

        self.pair_transform = pair_transform

        self.image_names = sorted(
            (path / "low/").glob("*.png"), key=lambda x: int(x.stem.split('_')[1])
        )
        self.target_names = sorted(
            (path / "high/").glob("*.png"), key=lambda x: int(x.stem.split('_')[1])
        )

        if limit is not None:
            self.image_names = self.image_names[start_idx:start_idx+limit]
            self.target_names = self.target_names[start_idx:start_idx+limit]

        self.loaded = preload
        if self.loaded:
            self.load_all_images()
    def load_all_images(self) -> None:
        self.loaded_images_ = []
        self.loaded_targets_ = []

        for index in range(len(self)):
            self.loaded_images_.append(read_image_cv2(self.image_names[index]))
            self.loaded_targets_.append(read_image_cv2(self.target_names[index]))

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> PairedImageWithLightnessInput:
        image = (
            self.loaded_images_[index]
            if self.loaded
            else read_image_cv2(self.image_names[index])
        )
        target = (
            self.loaded_targets_[index]
            if self.loaded
            else read_image_cv2(self.target_names[index])
        )

        transformed = self.pair_transform(image=image, target=target)
        image, target = transformed["image"], transformed["target"]

        source_lightness = transformed["source_lightness"]
        target_lightness = transformed["target_lightness"]

        return PairedImageWithLightnessInput(
            image=image,
            target=target,
            source_lightness=source_lightness,
            target_lightness=target_lightness,
        )

class FSDDataModule(pl.LightningDataModule):
        def __init__(self, config):
            super().__init__()
            self.root = Path(config.path)
            self.config = config

            self.train_transform, self.test_transform = load_transforms(config.transform)

        def setup(self, stage: Optional[str] = None):
            self.train_ds = FSD(
                self.root,
                split='train',
                pair_transform=self.train_transform,
                preload=self.config.preload,
                start_idx=self.config.start_idx,
                limit=self.config.limit,
            )

            self.val_ds = FSD(
                self.root,
                split='val',
                pair_transform=self.test_transform,
                preload=self.config.preload,
            )

            self.test_ds = FSD(
                self.root,
                split='test',
                pair_transform=self.test_transform,
                preload=self.config.preload,
            )

        def train_dataloader(self):
            return DataLoader(
                self.train_ds,
                batch_size=self.config.batch_size,
                pin_memory=self.config.pin_memory,
                num_workers=self.config.num_workers,
                shuffle=True,
            )

        def val_dataloader(self):
            return DataLoader(
                self.val_ds,
                batch_size=1,
                pin_memory=self.config.pin_memory,
                num_workers=self.config.num_workers,
            )

        def test_dataloader(self):
            return DataLoader(
                self.test_ds,
                batch_size=1,
                pin_memory=self.config.pin_memory,
                num_workers=self.config.num_workers,
            )

        def predict_dataloader(self):
            return self.test_dataloader()
