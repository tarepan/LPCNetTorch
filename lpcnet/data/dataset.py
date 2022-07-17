"""Datasets"""


from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from hashlib import md5

from torch.utils.data import Dataset
from tqdm import tqdm
from omegaconf import MISSING
from speechdatasety.interface.speechcorpusy import AbstractCorpus, ItemId               # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.archive import try_to_acquire_archive_contents, save_archive # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.adress import dataset_adress                                 # pyright: ignore [reportMissingTypeStubs]
from speechdatasety.helper.access import generate_saver_loader                          # pyright: ignore [reportMissingTypeStubs]

from ..domain import HogeFugaBatch
from .domain import HogeFuga_, HogeFugaDatum
from .transform import ConfTransform, load_raw, preprocess, augment, collate


"""
(delele here when template is used)

[Design Notes - Corpus instance]
    'Corpus in Dataset' enables corpus lazy evaluation, which enables corpus contents download skip.
    For this reason, `AbstractCorpus` instances are passed to the Dataset.

[Design Notes - Corpus item]
    Corpus split is logically separated from Dataset.
    If we get corpus items by corpus instance method call in the dataset, we should write split logic in the dataset.
    If we pass splitted items to the dataset, we can separate split logic from the dataset.
    For this reason, both corpus instance and selected item list are passed as arguments.

[Design Notes - Corpus item path]
    Corpus instance has path-getter method.
    If we have multiple corpuses, we should distinguish the corpus that the item belongs to.
    If we pass paths as arguments, this problem disappear.
    For this reason, corpus_instance/selected_item_list/item_path are passed as arguments.

[Design Notes - Init interface]
    Dataset could consume multiple corpuses (corpus tuples), and the number is depends on project.
    For example, TTS will consumes single corpus, but voice conversion will consumes 'source' corpus and 'target' corpus.
    It means that init arguments are different between projects.
    For this reason, the Dataset do not have common init Inferface, it's up to you.

[Disign Notes - Responsibility]
    Data transformation itself is logical process (independent of implementation).
    In implementation/engineering, load/save/archiving etc... is indispensable.
    We could contain both data transform and enginerring in Dataset, but it can be separated.
    For this reason, Dataset has responsibility for only data handling, not data transform.
"""


CorpusItems = Tuple[AbstractCorpus, List[Tuple[ItemId, Path]]]


@dataclass
class ConfHogeFugaDataset:
    """Configuration of HogeFuga dataset.
    Args:
        adress_data_root - Root adress of data
        att1 - Attribute #1
    """
    adress_data_root: Optional[str] = MISSING
    attr1: int = MISSING
    transform: ConfTransform = ConfTransform()

class HogeFugaDataset(Dataset[HogeFugaDatum]):
    """The Hoge/Fuga dataset from the corpus.
    """
    def __init__(self, conf: ConfHogeFugaDataset, items: CorpusItems):
        """
        Args:
            conf: The Configuration
            items: Corpus instance and filtered item information (ItemId/Path pair)
        """

        # Store parameters
        self._conf = conf
        self._corpus = items[0]
        self._items = items[1]

        # Calculate data path
        conf_specifier = f"{conf.attr1}{conf.transform}"
        item_specifier = f"{list(map(lambda item: item[0], self._items))}"
        exp_specifier = md5((conf_specifier+item_specifier).encode()).hexdigest()
        self._adress_archive, self._path_contents = dataset_adress(
            conf.adress_data_root, self._corpus.__class__.__name__, "HogeFuga", exp_specifier
        )
        self._save_hogefuga, self._load_hogefuga = generate_saver_loader(HogeFuga_, ["hoge", "fuga"], self._path_contents)

        # Deploy dataset contents
        ## Try to 'From pre-generated dataset archive'
        contents_acquired = try_to_acquire_archive_contents(self._adress_archive, self._path_contents)
        ## From scratch
        if not contents_acquired:
            print("Dataset archive file is not found.")
            self._generate_dataset_contents()

    def _generate_dataset_contents(self) -> None:
        """Generate dataset with corpus auto-download and static preprocessing.
        """

        print("Generating new dataset...")

        # Lazy contents download
        self._corpus.get_contents()

        # Preprocessing - Load/Transform/Save
        for item_id, item_path in tqdm(self._items, desc="Preprocessing", unit="item"):
            piyo = load_raw(self._conf.transform.load, item_path)
            hoge_fuga = preprocess(self._conf.transform.preprocess, piyo)
            self._save_hogefuga(item_id, hoge_fuga)

        print("Archiving new dataset...")
        save_archive(self._path_contents, self._adress_archive)
        print("Archived new dataset.")

        print("Generated new dataset.")

    def __getitem__(self, n: int) -> HogeFugaDatum:
        """(API) Load the n-th datum from the dataset with tranformation.
        """
        item_id = self._items[n][0]
        return augment(self._conf.transform.augment, self._load_hogefuga(item_id))

    def __len__(self) -> int:
        return len(self._items)

    def collate_fn(self, items: List[HogeFugaDatum]) -> HogeFugaBatch:
        """(API) datum-to-batch function."""
        return collate(items)
