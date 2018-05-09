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
from random import randint
from typing import Type, List, Tuple
from urllib.request import urlretrieve

import torch
from torch.utils import data

from lang import AbstractVocabulary


class SetType(Enum):
    TRAIN = enum.auto()
    DEV = enum.auto()
    TEST = enum.auto()


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


def _contains_all(folder, files) -> bool:
    file_exist = map(lambda f: os.path.exists(os.path.join(folder, f)), files)
    return reduce(lambda a, b: a and b, file_exist)


class E2E(data.Dataset):
    """`E2E <http://www.macs.hw.ac.uk/InteractionLab/E2E/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/train.pt``, ``processed/dev.pt``
            and  ``processed/test.pt`` exist.
        which_set (SetType): Determines which of the subsets to use.
    """
    url = 'https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip'
    csv_folder = 'csv'
    train_file = 'train.pt'
    dev_file = 'dev.pt'
    test_file = 'test.pt'
    vocabulary_file = 'vocabulary.pt'

    def __init__(self, root, which_set: SetType, vocabulary_class: Type[AbstractVocabulary]):
        self.root = os.path.realpath(os.path.expanduser(root))
        self.which_set = which_set
        self.processed_folder = vocabulary_class.__name__

        if _contains_all(os.path.join(self.root, self.processed_folder),
                         [self.train_file, self.dev_file, self.test_file, self.vocabulary_file]):
            with open(os.path.join(self.root, self.processed_folder, self.vocabulary_file), 'rb') as f:
                self.vocabulary = pickle.load(f)

            options = {
                SetType.TRAIN: self.train_file,
                SetType.DEV:   self.dev_file,
                SetType.TEST:  self.test_file
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
        return map(list, zip(*src_data))

    def _download(self):
        """Download and process the E2E data."""
        csv_folder = os.path.join(self.root, self.csv_folder)
        if _contains_all(csv_folder, files=['trainset.csv', 'devset.csv', 'testset.csv']):
            # No need to download again
            return os.path.join(self.root, self.csv_folder)

        # Clean before download
        try:
            shutil.rmtree(os.path.join(self.root))
        except FileNotFoundError:
            # That's ok
            pass
        os.makedirs(csv_folder)

        print('Downloading ' + self.url)
        zip_path = os.path.join(self.root, 'e2e-dataset.zip')
        urlretrieve(self.url, zip_path)

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

        os.remove(zip_path)
        return csv_folder

    def _process(self, csv_folder):
        # Extract strings from CSV
        train_mr, train_ref = _extract_mr_ref(os.path.join(csv_folder, 'trainset.csv'))
        dev_mr, dev_ref = _extract_mr_ref(os.path.join(csv_folder, 'devset.csv'))
        test_mr, test_ref = _extract_mr_ref(os.path.join(csv_folder, 'testset.csv'))

        # Encode MR, REF as tensors and save them
        print('Encoding and saving examples')
        os.makedirs(os.path.join(self.root, self.processed_folder))

        train_list = self._strings_to_list(train_mr, train_ref)
        with open(os.path.join(self.root, self.processed_folder, self.train_file), 'wb') as f:
            torch.save(train_list, f)

        dev_list = self._strings_to_list(dev_mr, dev_ref)
        with open(os.path.join(self.root, self.processed_folder, self.dev_file), 'wb') as f:
            torch.save(dev_list, f)

        test_list = self._strings_to_list(test_mr, test_ref)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_list, f)

        # Store the right list in local fields
        options = {
            SetType.TRAIN: train_list,
            SetType.DEV:   dev_list,
            SetType.TEST:  test_list
        }
        self.mr, self.ref = map(list, zip(*options[self.which_set]))

        # Save the dictionary
        with open(os.path.join(self.root, self.processed_folder, self.vocabulary_file), 'wb') as f:
            pickle.dump(self.vocabulary, f, pickle.HIGHEST_PROTOCOL)

        print('Done!')

    def _strings_to_list(self, meaning_representations: List[str], references: List[str]) -> List[List[List[int]]]:
        examples = []
        for mr, ref in zip(meaning_representations, references):
            mr = self.vocabulary.add_sentence(mr)
            ref = self.vocabulary.add_sentence(ref)
            examples.append([mr, ref])
        return examples

    def to_string(self, tensor):
        return self.vocabulary.to_string(tensor)