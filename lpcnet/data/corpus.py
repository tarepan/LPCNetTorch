"Corpus splitting"

from typing import Tuple
from dataclasses import dataclass

from omegaconf import MISSING, SI
from speechcorpusy import load_preset # pyright: ignore [reportMissingTypeStubs]; bacause of library
from speechcorpusy.interface import ConfCorpus # pyright: ignore [reportMissingTypeStubs]; bacause of library

from .dataset import CorpusItems


@dataclass
class ConfCorpora:
    """Configuration of Corpora.

    Args:
        root - Corpus data root
        n_val - The number of validation items, for corpus split
        n_test - The number of test items, for corpus split
    """
    root: str = MISSING
    train: ConfCorpus = ConfCorpus(
        root=SI("${..root}"))
    val: ConfCorpus = ConfCorpus(
        root=SI("${..root}"))
    test: ConfCorpus = ConfCorpus(
        root=SI("${..root}"))
    n_val: int = MISSING
    n_test: int = MISSING

def prepare_corpora(conf: ConfCorpora) -> Tuple[CorpusItems, CorpusItems, CorpusItems]:
    """Instantiate corpuses and split them for datasets.

    Returns - CorpusItems for train/val/test
    """

    # Instantiation
    ## No needs of content init. It is a duty of consumer (Dataset).
    corpus_train, corpus_val, corpus_test = load_preset(conf=conf.train), load_preset(conf=conf.val), load_preset(conf=conf.test)

    ## Pass-through
    items_train = corpus_train.get_identities()
    items_val   =   corpus_val.get_identities()
    items_test  =  corpus_test.get_identities()

    # CorpusItem-nize
    corpus_items_train: CorpusItems = (corpus_train, list(map(lambda item: (item, corpus_train.get_item_path(item)), items_train)))
    corpus_items_val:   CorpusItems = (corpus_val,   list(map(lambda item: (item,   corpus_val.get_item_path(item)), items_val)))
    corpus_items_test:  CorpusItems = (corpus_test,  list(map(lambda item: (item,  corpus_test.get_item_path(item)), items_test)))

    return corpus_items_train, corpus_items_val, corpus_items_test
