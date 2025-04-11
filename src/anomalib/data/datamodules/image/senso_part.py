"""SensoPart AD Data Module.

This module provides a PyTorch Lightning DataModule for the SensoPart AD dataset.
The dataset has to be provided locally.

Example:
    Create a SensoPartAD datamodule::

        >>> from anomalib.data import SensoPartAD
        >>> datamodule = SensoPartAD(
        ...     root="./datasets/SensoPartAD",
        ...     category="2_Anwesenheit_und_Vollstaendigkeit"
        ...     sub_category="2_1_BMW_Blechmontage"
        ... )

Notes:
    The dataset has to be provided locally. The directory structure after preparation will be::

        datasets/
        └── SensoPartAD/
            ├── 2_Anwesenheit_und_Vollstaendigkeit/
            |   ├── 2_1_BMW_Blechmontage/
            |   ├── 2_2_Presence_O-Ring
            |   └── ... 
            ├── 3_Montagepruefung/
            |   ├── 3_1_Tritecnica_Colored_pen/
            |   ├── 3_2_DAIMLER_Pyro-Ring
            |   └── ... 
            └── ...
"""


import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.senso_part import SensoPartADDataset
from anomalib.data.utils import Split, TestSplitMode, ValSplitMode

logger = logging.getLogger(__name__)


class SensoPartAD(AnomalibDataModule):
    """SensoPartAD Datamodule.

    Args:
        root (Path | str): Path to the root of the dataset.
            Defaults to ``"./datasets/SensoPartAD"``.
        category (str): Category of the SensoPartAD dataset (e.g. ``"2_Anwesenheit_und_Vollstaendigkeit"`` or
            ``"3_Montagepruefung"``). Defaults to ``"2_Anwesenheit_und_Vollstaendigkeit"``.
        sub_category (str): Sub-category of the SensoPartAD dataset (e.g. ``"2_1_BMW_Blechmontage"``)
          Defaults to ``'2_1_BMW_Blechmontage'``
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Test batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        train_augmentations (Transform | None): Augmentations to apply dto the training images
            Defaults to ``None``.
        val_augmentations (Transform | None): Augmentations to apply to the validation images.
            Defaults to ``None``.
        test_augmentations (Transform | None): Augmentations to apply to the test images.
            Defaults to ``None``.
        augmentations (Transform | None): General augmentations to apply if stage-specific
            augmentations are not provided.
        test_split_mode (TestSplitMode): Method to create test set.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Method to create validation set.
            Defaults to ``ValSplitMode.SAME_AS_TEST``.
        val_split_ratio (float): Fraction of data to use for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed for reproducibility.
            Defaults to ``None``.
        num_train_imgs (int | None, optional): Limit the number of samples used for trainig
            Defaults to ``None`` 

    Example:
        Create SensoPart datamodule with default settings::

            >>> datamodule = SensoPartAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image'])

            >>> data["image"].shape
            torch.Size([32, 3, 900, 900])
    """

    def __init__(
        self,
        root: Path | str = "./datasets/SensoPartAD",
        category: str = "2_Anwesenheit_und_Vollstaendigkeit",
        sub_category: str = "2_1_BMW_Blechmontage",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
        num_train_imgs: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category
        self.sub_category = sub_category
        self.test_split_ratio = test_split_ratio
        self.num_train_imgs = num_train_imgs

    def _setup(self, _stage: str | None = None) -> None:
        """Set up the datasets and perform dynamic subset splitting.

        This method may be overridden in subclass for custom splitting behaviour.

        Note:
            The stage argument is not used here. This is because, for a given
            instance of an AnomalibDataModule subclass, all three subsets are
            created at the first call of setup(). This is to accommodate the
            subset splitting behaviour of anomaly tasks, where the validation set
            is usually extracted from the test set, and the test set must
            therefore be created as early as the `fit` stage.
        """
        self.train_data = SensoPartADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
            sub_category=self.sub_category,
            test_split_ratio=self.test_split_ratio,
            num_train_imgs=self.num_train_imgs,
        )
        self.test_data = SensoPartADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
            sub_category=self.sub_category,
            test_split_ratio=self.test_split_ratio,
            num_train_imgs=self.num_train_imgs
        )

    def prepare_data(self) -> None:
        """Download the dataset if not available.

        This method checks if the specified dataset is available in the file
        system. If not, it downloads and extracts the dataset into the
        appropriate directory.

        Example:
            Create SensoPart datamodule with default settings::

            >>> datamodule = SensoPartAD()
            >>> datamodule.setup()
            >>> i, data = next(enumerate(datamodule.train_dataloader()))
            >>> data.keys()
            dict_keys(['image_path', 'label', 'image'])

            >>> data["image"].shape
            torch.Size([32, 3, 900, 900]) 
        """
        if (self.root / self.category / self.sub_category).is_dir():
            logger.info("Found the dataset.")
        else:
            msg = f"Dataset not found in {self.root / self.category / self.sub_category}."
            raise RuntimeError(msg)
