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

from pathlib import Path 
from collections.abc import Sequence
from pandas import DataFrame

from anomalib.data.datasets import AnomalibDataset
from anomalib.data.utils import LabelName, validate_path


IMG_EXTENSIONS = (".png", ".PNG", ".bmp", ".BMP", ".jpg", ".JPG",  ".jpeg",  ".JPEG")
CATEGORIES = (
    "2_Anwesenheit_und_Vollstaendigkeit",
    "3_Montagepruefung", 
    "4_Leerkontrolle",
    "5_Qualitaet_allgemein",
    "6_Quality_Oberflaeche",
    "7_Bauch_Ruecken",
)

class SensoPartDataset(AnomalibDataset):
    """SensoPart dataset class.
    
    Args: 
        root (Path | str): Path to the directory containing the dataset
            ``Defaults to ./dataset/SensoPartAD``
        category (str): Categroy name, must be one of ``CATEGORIES`` 
            Defaults to ``2_Anwesenheit_und_Vollstaendigkeit```
        sub_category (str): Subcategory, must be one of ``SUB_CATEGORIES``

    Example: 
        >>> from pathlib import Path
        >>> from anomalib.data.datasets import SensoPartDataset
        >>> dataset = SensoPartDataset(
        ...     root=Path("./dataset/SensoPartAD"),
        ...     category="2_Anwesenheit_und_Vollstaendigkeit", 
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
    ) -> None:
        super().__init__()

        self.root_category = Path(root) / Path(category)
        self.category = category
        self.sub_category = sub_category
        self.sample = make_senso_part_dataset(
            self.root_category, 
            extensions=IMG_EXTENSIONS
        )

def make_senso_part_dataset(
    root: str | Path, 
    extension: Sequence[str] | None = None,
) -> DataFrame:
    """Create SensoPart AD samples by parsing the data directory structure.

    The files are expected to follow the structure:
        ``path/to/dataset//category/image_filename.png``

    Args:
        root (Path | str): Path to dataset root directory
        split (str | Split | None, optional): Dataset split (train or test)
            Defaults to ``None``.
        extension (Sequence[str] | None, optional): Valid file extension
            Defaults to ``None``.

    Returns:
        DataFrame: Dataset samples with columns:
            - path: Base path to dataset
            - label: Class label
            - image_path: Path to image file
            - label_index: Numeric label (0=normal, 1=abnormal)

    Example:
        >>> root = Path("./datasets/SensoPartAD")
        >>> category="2_Anwesenheit_und_Vollstaendigkeit"
        >>> samples = make_mvtec_dataset(root, categrory)
        >>> samples.head()
           path                                                    label image_path           label_index
        0  datasets/SensoPartAD/2_Anwesenheit_und_Vollstaendigkeit OK    [...]/good/105.png   0
        1  datasets/SensoPartAD/2_Anwesenheit_und_Vollstaendigkeit OK    [...]/good/017.png   0

    Raises:
        RuntimeError: If no valid images are found
    """
    if extension is None:
        extension = IMG_EXTENSIONS
    
    root = validate_path(root)

