"""SensoPart AD Dataset

This module provides PyTorch implementations for the internal SensoPart Anomaly Dataset

This dataset is local on this machine and has some small modificatins from the one on F:
For each category there a multiple subdatasets and those need to have 2 folders: ``nIO`` and ``IO``

The Categories are:
[
    2_Anwesenheit_und_Vollstaendigkeit,
    3_Montagepruefung, 
    4_Leerkontrolle,
    5_Qualitaet_allgemein,
    6_Quality_Oberflaeche,
    7_Bauch_Ruecken,
]

e.g. 2_Anwesenheit_und_Vollstaendigkeit contains 
    category_datatsets: [
        2_1_BMW_Blechmontage,
        2_2_Presence_O-Ring,
        2_3_BMW-USA_Schweissnaht,
        2_4_SP-USA_WeldSeam_presence,
        2_5_Zahoransky_Completeness_MP,
        2_6_BMW_MUC_Band50ST150Heber05_Kettenkontrolle_li,
        2_7_BMW_MUC_Band50ST150Heber05_Kettenkontrolle_re, 
        2_8_Weckerle, 
    ]
"""
import logging
import pandas as pd
import math
from pathlib import Path 
from collections.abc import Sequence
from pandas import DataFrame

from anomalib.data.datasets import AnomalibDataset
from anomalib.data.utils import LabelName, validate_path, Split

logger = logging.getLogger(__name__)

IMG_EXTENSIONS = (".png", ".PNG", ".bmp", ".BMP", ".jpg", ".JPG",  ".jpeg",  ".JPEG")
CATEGORIES = (
    "2_Anwesenheit_und_Vollstaendigkeit",
    "3_Montagepruefung", 
    "4_Leerkontrolle",
    "5_Qualitaet_allgemein",
    "6_Quality_Oberflaeche",
    "7_Bauch_Ruecken",
)

SUB_CATEGORIES = (
    '2_1_BMW_Blechmontage', 
    '2_2_Presence_O-Ring', 
    '2_3_BMW-USA_Schweissnaht', 
    '2_4_SP-USA_WeldSeam_presence', 
    '2_5_Zahoransky_Completeness_MP', 
    '2_6_BMW_MUC_Band50ST150Heber05_Kettenkontrolle_li', 
    '2_7_BMW_MUC_Band50ST150Heber05_Kettenkontrolle_re', 
    '2_8_Weckerle', 
    '13_Tuben', 
    '3_1_Tritecnica_Colored_pen', 
    '3_2_DAIMLER_Pyro', 
    '3_3_WeilEN_Rohre_', 
    '3_4_KrausMaffeySchweden_ARu', 
    '3_5_DAIMLER_Steuereinheit_Positionierung', 
    '3_6_BMW-USA_Montagepruefung'
    '4_1_CJa_Hengst'
    '5_1_SP_FR_Print1', 
    '5_1_SP_FR_Print2', 
    '5_2_Tritecnica_Electric_contacts', 
    '5_3_SP_FR_42', 
    '5_4_Tritecnica_aerosol', 
    '5_5_Tritecnica_cable_cam35', 
    '5_6_BMW_Gummidichtung_LoecherTHD_G02_12-04-23', 
    '5_7_ARu AUDI Griffe', 
    '5_8_DAIMLER_Pyrosicherung'
    '6_1_Kunstleder_Diss_KBE', 
    '6_2_BMW_MetalNut', 
    '6_3_curaprox_ARu_Druck_close', 
    '6_4_curaprox_ARu_Druck_c', 
    '6_5_Hella_PlasticBurn', 
    '6_6_Steatit-Bruch', 
    '6_7_SPKorea_Welding',
    '7_1_SP_FR_41', 
    '7_2_SensoPart_Zufuehrung_Kunststoffteil',
)

class SensoPartADDataset(AnomalibDataset):
    """SensoPart dataset class.
    
    Args: 
        root (Path | str): Path to the directory containing the dataset
            ``Defaults to ./dataset/SensoPartAD``
        category (str): Categroy name, must be one of ``CATEGORIES`` 
            Defaults to ``2_Anwesenheit_und_Vollstaendigkeit```
        sub_category (str): Subcategory, must be one of ``SUB_CATEGORIES``
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        num_train_imgs (int | None, optional): Limit the number of training images
            Defaults to ``None``

    Example: 
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import SensoPartADDataset
        >>> dataset = SensoPartADDataset(
        ...     root=Path("./dataset/SensoPartAD"),
        ...     category="2_Anwesenheit_und_Vollstaendigkeit", 
        ...     sub_category="2_1_BMW_Blechmontage",
        ... )

        # For classification tasks each sample contains:
        >>> sample = dataset[0]
        >>> list(samples.keys())
        ['image_path', 'label',  'image']

        # Images are PyTorch Tensors with shape ``(C, H, W)``
        sample["image"].shape
    """

    def __init__(
        self, 
        root: Path | str = "./datasets/SensoPartAD", 
        category: str = "2_Anwesenheit_und_Vollstaendigkeit",
        sub_category: str = "2_1_BMW_Blechmontage",
        split: Split = Split.TRAIN ,
        test_split_ratio : float = 0.2,
        num_train_imgs: int | None = None,
    ) -> None:
        super().__init__()

        self.category = category
        self.split = split
        self.sub_category = sub_category
        self.root_category = Path(root) / Path(category)
        self.root_subcategory = Path(root) / Path(category) / Path(sub_category)
        self.num_train_imgs = num_train_imgs
        self.samples = make_senso_part_dataset(
            self.root_subcategory, 
            split=self.split,
            test_split_ratio=test_split_ratio,
            extensions=IMG_EXTENSIONS,
            num_train_imgs=self.num_train_imgs
        )

def make_senso_part_dataset(
    root: str | Path, 
    split: Split = Split.TRAIN,
    test_split_ratio: float = 0.2,
    extensions: Sequence[str] | None = None,
    num_train_imgs: int | None = None,
) -> DataFrame:
    """Create SensoPart AD samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``path/to/dataset//category/sub_category/IO/image_filename.png``
        ``path/to/dataset//category/sub_category/nIO/image_filename.png``
        

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        test_split_ratio (float): Fraction of data to use for testing.
            Defaults to ``0.2``.
        extension (Sequence[str] | None, optional): Valid file extension
            Defaults to ``None``.
        num_train_imgs (int | None, optional): Limit the number of training images
            Defaults to ``None``

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - label: Class label
            - image_path: Path to image file
            - label_index: Numeric label (0=normal, 1=abnormal)
            - split: Dataset split (train/test)

    Example:
        >>> root = Path("./datasets/SensoPartAD")
        >>> category="2_Anwesenheit_und_Vollstaendigkeit"
        >>> sub_category="2_1_BMW_Blechmontage"
        >>> samples = make_mvtec_dataset(root, categrory, sub_category)
        >>> samples.head()
           path                                                                             label image_path         label_index
        0  datasets/SensoPartAD/2_Anwesenheit_und_Vollstaendigkeit/2_1_BMW_Blechmontage/    OK    [...]/IO/105.png   0
        1  datasets/SensoPartAD/2_Anwesenheit_und_Vollstaendigkeit/2_1_BMW_Blechmontage/    OK    [...]/IO/017.png   0

    Raises:
        RuntimeError: If no valid images are found
    """
    if extensions is None:
        extensions = IMG_EXTENSIONS
    
    root = validate_path(root)

    sample_list = [(str(root),) + f.parts[-2:] for f in root.glob(r"**/*") if f.suffix in extensions]
    if not sample_list:
        msg = f"Found 0 images in {root}"
        raise RuntimeError(msg) 
    
    samples = DataFrame(
        sample_list, columns=["path", "label", "image_path"]
    )

    # Modify by converting the path to an absolute path
    samples["image_path"] = samples["path"] + "/" + samples["label"] + "/" + samples["image_path"]
    samples["split"] = split
    samples["mask_path"] = ""
    
    
    # infer the task type
    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"
       
    # create labels for noraml(0) and abnormal(1) images.
    samples.loc[(samples.label == "IO"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label == "nIO"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    # divide into normal ananomalous samples
    normal_samples = samples.drop(samples[samples.label_index != LabelName.NORMAL].index)
    anomalous_samples = samples.drop(samples[samples.label_index != LabelName.ABNORMAL].index)
    
    # Set ``num_train_imgs`` for splitting the data
    if num_train_imgs is not None:
        # check that num_train_images does not exceed number of normal images
        if len(normal_samples) < num_train_imgs:
            logger.info(f"Number of normal images in dataset is smaller than 'num_train_imgs'. So set num_train_imgs to {len(normal_samples)}")
            num_train_imgs = len(normal_samples)    
    else:
        # divide normal samples by split ratio 
        num_train_imgs = math.floor(len(normal_samples) * (1.0 - test_split_ratio))
    
    # assign samples to train and test splits    
    if split == Split.TRAIN:
        samples = normal_samples.iloc[0:num_train_imgs]
    elif split == Split.TEST:
        if len(normal_samples) > 64: # take first test image with idx=64
            normal_samples = normal_samples.iloc[64:]
            samples = pd.concat([normal_samples, anomalous_samples]) 
        else:
            samples = anomalous_samples
    return samples
