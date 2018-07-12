# coding=utf-8
from _ctypes import ArgumentError

import torch
from torch.utils.data import DataLoader

from e2e import E2E
from lang import EOS_token, PAD_token


class E2EDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, batch_first=True):
        if not isinstance(dataset, E2E):
            raise ArgumentError('Your dataset must be an E2E instance!')
        self.batch_first = batch_first
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, self._collate_fn,
                         pin_memory,
                         drop_last, timeout, worker_init_fn)

    def _collate_fn(self, batch):
        batch = sorted(batch, key=lambda example: len(example[0]), reverse=True)

        mr_lengths = [len(example[0]) for example in batch]
        max_mr_len = max(mr_lengths)
        max_ref_len = max([len(example[1]) for example in batch])

        mrs = torch.LongTensor([_pad_sequence(mr, max_mr_len) for _, (mr, _) in enumerate(batch)])
        refs = torch.LongTensor([_pad_sequence(ref, max_ref_len) for _, (_, ref) in enumerate(batch)])

        if not self.batch_first:
            mrs.transpose_(0, 1)
            refs.transpose_(0, 1)

        return (mrs, mr_lengths), refs

    def test_model(self, model, num_tests):
        self_iter = self.__iter__()
        mr = ref = lengths = None
        for i in range(num_tests):
            i = i % self.batch_size
            if i == 0:
                mr, ref = next(self_iter)
                mr, lengths = mr
            ref_ = model((mr[i, :lengths[i]].unsqueeze(0), [lengths[i]])).squeeze()
            _, ref_ = ref_.topk(1)
            print('MR : {}'.format(self.dataset.to_string(mr[i].tolist())))
            print('REF: {}'.format(self.dataset.to_string(ref[i].tolist())))
            print('GEN: {}'.format(self.dataset.to_string(ref_.data.squeeze().tolist())))


def _pad_sequence(sequence, length):
    pad_length = length - len(sequence)
    sequence = sequence + [PAD_token for _ in range(pad_length)]
    return sequence


def sequence_lengths(sequences):
    return [(s == EOS_token).nonzero().item() for s in sequences]
