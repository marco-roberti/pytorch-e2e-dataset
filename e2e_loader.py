# coding=utf-8
from _ctypes import ArgumentError

import torch
from torch.utils.data import DataLoader

from e2e import E2E
from lang import EOS_token, PAD_token


class E2EDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        if not isinstance(dataset, E2E):
            raise ArgumentError('Your dataset must be an E2E instance!')
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, _collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn)


def _collate_fn(batch):
    batch = sorted(batch, key=lambda example: len(example[0]), reverse=True)

    mr_lengths = [len(example[0]) for example in batch]
    max_mr_len = max(mr_lengths)
    max_ref_len = max([len(example[1]) for example in batch])

    mrs = [_pad_sequence(mr, max_mr_len) for _, (mr, _) in enumerate(batch)]
    refs = [_pad_sequence(ref, max_ref_len) for _, (_, ref) in enumerate(batch)]

    return mrs, refs, mr_lengths, max_ref_len


def _pad_sequence(sequence, length):
    pad_length = length - len(sequence)
    sequence = sequence + [PAD_token for _ in range(pad_length)]
    return sequence


def sequence_lengths(sequences):
    return [(s == EOS_token).nonzero().item() for s in sequences]
