# coding=utf-8
import csv
import enum
import os
import os.path
import pickle
import shutil
import zipfile
from enum import Enum
from functools import reduce
from typing import Type, List, Tuple, Union
from urllib.request import urlretrieve

import torch
from torch.utils import data

from lang import AbstractVocabulary


class E2ESet(Enum):
    TRAIN = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()
    ALL_IN_ONE = enum.auto()


class E2E(data.Dataset):
    """`E2E <http://www.macs.hw.ac.uk/InteractionLab/E2E/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/train.pt``, ``processed/dev.pt``
            and  ``processed/test.pt`` exist.
        which_set (E2ESet): Determines which of the subsets to use.
    """
    _url = 'https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip'
    _csv_folder = 'csv'
    _train_file = 'train.pt'
    _dev_file = 'dev.pt'
    _test_file = 'test.pt'
    _all_in_one_file = 'all_in_one.pt'
    _vocabulary_file = 'vocabulary.pt'

    def __init__(self, root, which_set: E2ESet, vocabulary_class: Type[AbstractVocabulary]):
        super(E2E, self).__init__()
        self.root = os.path.realpath(os.path.expanduser(root))
        self.which_set = which_set
        self.processed_folder = vocabulary_class.__name__

        if _contains_all(os.path.join(self.root, self.processed_folder),
                         [self._train_file, self._dev_file, self._test_file, self._vocabulary_file]):
            with open(os.path.join(self.root, self.processed_folder, self._vocabulary_file), 'rb') as f:
                self.vocabulary = pickle.load(f)

            options = {
                E2ESet.TRAIN:      self._train_file,
                E2ESet.DEV:        self._dev_file,
                E2ESet.TEST:       self._test_file,
                E2ESet.ALL_IN_ONE: self._all_in_one_file
            }
            self.mr, self.ref = self._load_from_file(options[which_set])
        else:
            print('The dataset does not exist locally!')
            self.vocabulary = vocabulary_class()
            folder = self._download()
            self._process(folder)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (mr, ref)
        """
        return (torch.LongTensor(self.mr[index]),
                torch.LongTensor(self.ref[index]))

    def __len__(self) -> int:
        return len(self.mr)

    def __repr__(self) -> str:
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '\tNumber of instances: {}\n'.format(self.__len__())
        fmt_str += '\tSet type: {}\n'.format(self.which_set.name.lower())
        fmt_str += '\tRoot Location: {}\n'.format(self.root)
        return fmt_str

    def _load_from_file(self, src_file):
        src_data = torch.load(
                os.path.join(self.root, self.processed_folder, src_file))
        return [list(z) for z in zip(*src_data)]

    def _download(self):
        """Download and process the E2E data."""
        csv_folder = os.path.join(self.root, self._csv_folder)
        if _contains_all(csv_folder, files=['trainset.csv', 'devset.csv', 'testset.csv']):
            # No need to download again
            return os.path.join(self.root, self._csv_folder)

        # Clean before download
        try:
            shutil.rmtree(os.path.join(self.root))
        except FileNotFoundError:
            # That's ok
            pass
        os.makedirs(csv_folder)

        print('Downloading ' + self._url)
        zip_path = os.path.join(self.root, 'e2e-dataset.zip')
        urlretrieve(self._url, zip_path)

        print('Extracting zip archive')
        with zipfile.ZipFile(zip_path) as zip_f:
            zip_f.extractall(self.root)

        # Rename folder
        os.rename(os.path.join(self.root, 'e2e-dataset'), csv_folder)

        # Delete/rename files
        os.remove(os.path.join(csv_folder, 'README.md'))
        os.remove(os.path.join(csv_folder, 'testset.csv'))
        os.rename(os.path.join(csv_folder, 'testset_w_refs.csv'),
                  os.path.join(csv_folder, 'testset.csv'))

        # Create all_in_one.csv
        all_in_one_name = os.path.join(csv_folder, 'all_in_one.csv')
        seen = set()  # set for fast O(1) amortized lookup
        with open(all_in_one_name, 'w') as all_in_one:
            all_in_one.write('md, ref\n')
            for file in ['trainset.csv', 'devset.csv', 'testset.csv']:
                with open(os.path.join(csv_folder, file), 'r') as in_file:
                    next(in_file)
                    for line in in_file:
                        if line not in seen:
                            seen.add(line)
                            all_in_one.write(line)

        os.remove(zip_path)
        return csv_folder

    def _process(self, csv_folder):
        # Extract strings from CSV
        train_mr, train_ref = _extract_mr_ref(os.path.join(csv_folder, 'trainset.csv'))
        dev_mr, dev_ref = _extract_mr_ref(os.path.join(csv_folder, 'devset.csv'))
        test_mr, test_ref = _extract_mr_ref(os.path.join(csv_folder, 'testset.csv'))
        all_in_one_mr, all_in_one_ref = _extract_mr_ref(os.path.join(csv_folder, 'all_in_one.csv'))

        # Encode MR, REF as tensors and save them
        print('Encoding and saving examples')
        os.makedirs(os.path.join(self.root, self.processed_folder))

        train_list = self._strings_to_list(train_mr, train_ref)
        with open(os.path.join(self.root, self.processed_folder, self._train_file), 'wb') as f:
            torch.save(train_list, f)

        dev_list = self._strings_to_list(dev_mr, dev_ref)
        with open(os.path.join(self.root, self.processed_folder, self._dev_file), 'wb') as f:
            torch.save(dev_list, f)

        test_list = self._strings_to_list(test_mr, test_ref)
        with open(os.path.join(self.root, self.processed_folder, self._test_file), 'wb') as f:
            torch.save(test_list, f)

        all_in_one_list = self._strings_to_list(all_in_one_mr, all_in_one_ref)
        with open(os.path.join(self.root, self.processed_folder, self._all_in_one_file), 'wb') as f:
            torch.save(all_in_one_list, f)

        # Store the right list in local fields
        options = {
            E2ESet.TRAIN:      train_list,
            E2ESet.DEV:        dev_list,
            E2ESet.TEST:       test_list,
            E2ESet.ALL_IN_ONE: all_in_one_list
        }
        self.mr, self.ref = [list(z) for z in zip(*options[self.which_set])]

        # Save the dictionary
        with open(os.path.join(self.root, self.processed_folder, self._vocabulary_file), 'wb') as f:
            pickle.dump(self.vocabulary, f, pickle.HIGHEST_PROTOCOL)

        print('Done!')

    def _strings_to_list(self, meaning_representations: List[str], references: List[str]) -> List[List[List[int]]]:
        examples = []
        for mr, ref in zip(meaning_representations, references):
            mr = self.vocabulary.add_sentence(mr)
            ref = self.vocabulary.add_sentence(ref)
            examples.append([mr, ref])
        return examples

    def to_string(self, tensor: Union[torch.Tensor, list]):
        if type(tensor) is torch.Tensor:
            tensor = tensor.squeeze().tolist()
        return self.vocabulary.to_string(tensor)


def _contains_all(folder, files) -> bool:
    file_exist = [os.path.exists(os.path.join(folder, f)) for f in files]
    return reduce(lambda a, b: a and b, file_exist)


def _extract_mr_ref(file) -> Tuple[List[str], List[str]]:
    print('Processing ' + file)
    mr = []
    ref = []
    with open(file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        for row in reader:
            mr.append(row[0])
            ref.append(row[1])
    return mr, ref
