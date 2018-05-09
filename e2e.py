import csv
import errno
import os
import os.path
import pickle
import zipfile
from enum import Enum, auto
from random import randint
from urllib import request

import torch
from torch.utils import data

from lang import WordVocabulary, EOS_token


class SetType(Enum):
    TRAIN = auto()
    DEV = auto()
    TEST = auto()


def _extract_mr_ref(file):
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


class E2E(data.Dataset):
    """`E2E <http://www.macs.hw.ac.uk/InteractionLab/E2E/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/train.pt``, ``processed/dev.pt``
            and  ``processed/test.pt`` exist.
        which_set (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
    """
    url = 'https://github.com/tuetschek/e2e-dataset/releases/download/v1.0.0/e2e-dataset.zip'
    csv_folder = 'csv'
    processed_folder = 'processed'
    train_file = 'train.pt'
    dev_file = 'dev.pt'
    test_file = 'test.pt'
    vocabulary_file = 'vocabulary.pt'

    def __init__(self, root, which_set: SetType):
        self.root = os.path.expanduser(root)
        self.which_set = which_set

        if self._check_exists():
            with open(os.path.join(self.root, self.processed_folder, self.vocabulary_file), 'rb') as f:
                self.vocabulary = pickle.load(f)
            if which_set == SetType.TRAIN:
                self.train_mr, self.train_ref = self._load_from_file(self.dev_file)
            elif which_set == SetType.DEV:
                self.dev_mr, self.dev_ref = self._load_from_file(self.dev_file)
            else:
                assert which_set == SetType.TEST
                self.test_mr, self.test_ref = self._load_from_file(self.test_file)
        else:
            print('The dataset does not exist locally!')
            self.vocabulary = WordVocabulary()
            self._download()

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (mr, ref)
        """
        if self.which_set == SetType.TRAIN:
            return self.train_mr[index], self.train_ref[index]
        elif self.which_set == SetType.DEV:
            return self.dev_mr[index], self.dev_ref[index]
        else:
            assert self.which_set == SetType.TEST
            return self.test_mr[index], self.test_ref[index]

    def __len__(self):
        if self.which_set == SetType.TRAIN:
            return len(self.train_mr)
        elif self.which_set == SetType.DEV:
            return len(self.dev_mr)
        else:
            assert self.which_set == SetType.TEST
            return len(self.test_mr)

    def __repr__(self):
        fmt_str = 'Dataset {}\n'.format(self.__class__.__name__)
        fmt_str += '\tNumber of instances: {}\n'.format(self.__len__())
        fmt_str += '\tSet type: {}\n'.format(self.which_set.name.lower())
        fmt_str += '\tRoot Location: {}\n'.format(self.root)
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.train_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.dev_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def _load_from_file(self, src_file):
        src_data = torch.load(
                os.path.join(self.root, self.processed_folder, src_file))
        return map(list, zip(*src_data))

    def _download(self):
        """Download the E2E data if it doesn't exist in processed_folder already."""
        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print('Downloading ' + self.url)
        downloaded_data = request.urlopen(self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, filename)
        with open(file_path, 'wb') as f:
            f.write(downloaded_data.read())

        print('Extracting zip archive')
        with zipfile.ZipFile(file_path) as zip_f:
            zip_f.extractall(self.root)
        os.unlink(file_path)
        csv_folder = os.path.join(self.root, self.csv_folder)
        os.rename(os.path.join(self.root, 'e2e-dataset'), csv_folder)
        # Delete useless files and rename new ones
        os.remove(os.path.join(csv_folder, 'README.md'))
        os.remove(os.path.join(csv_folder, 'testset.csv'))
        os.rename(os.path.join(csv_folder, 'testset_w_refs.csv'),
                  os.path.join(csv_folder, 'testset.csv'))

        # Extract strings from CSV
        self.train_mr, self.train_ref = _extract_mr_ref(os.path.join(csv_folder, 'trainset.csv'))
        self.dev_mr, self.dev_ref = _extract_mr_ref(os.path.join(csv_folder, 'devset.csv'))
        self.test_mr, self.test_ref = _extract_mr_ref(os.path.join(csv_folder, 'testset.csv'))

        # Encode MR, REF as tensors and save them
        print('Encoding and saving examples')
        tensor = self._strings_to_tensor(self.train_mr, self.train_ref)
        self.train_mr, self.train_ref = map(list, zip(*tensor))
        with open(os.path.join(self.root, self.processed_folder, self.train_file), 'wb') as f:
            torch.save(tensor, f)

        tensor = self._strings_to_tensor(self.dev_mr, self.dev_ref)
        self.dev_mr, self.dev_ref = map(list, zip(*tensor))
        with open(os.path.join(self.root, self.processed_folder, self.dev_file), 'wb') as f:
            torch.save(tensor, f)

        tensor = self._strings_to_tensor(self.test_mr, self.test_ref)
        self.test_mr, self.test_ref = map(list, zip(*tensor))
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(tensor, f)

        with open(os.path.join(self.root, self.processed_folder, self.vocabulary_file), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.vocabulary, f, pickle.HIGHEST_PROTOCOL)

        print('Done!')

    def _strings_to_tensor(self, meaning_representations, references):
        examples = []
        for mr, ref in zip(meaning_representations, references):
            mr = self.vocabulary.add_sentence(mr)
            ref = self.vocabulary.add_sentence(ref)
            mr.append(EOS_token)
            ref.append(EOS_token)
            examples.append([mr, ref])
        return examples

    def random_example(self):
        i = randint(0, len(self))
        mr = self[i][0]
        ref = self[i][1]
        return self.vocabulary.to_string(mr), self.vocabulary.to_string(ref)
