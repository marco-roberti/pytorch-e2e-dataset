# PyTorch E2E DataSet
The [E2E Challenge Dataset](http://www.macs.hw.ac.uk/InteractionLab/E2E/),
packed as a [PyTorch DataSet](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset)
subclass.

## Getting started
The E2E class will automatically download and process the dataset, converting
the CSV files to a list of [MR, REF] matches.

Every string is represented as a list of the chosen Vocabulary class' keys: a
Vocabulary object can convert lists to strings and vice versa.
You will find in *lang.py* a word-based and a character-based vocabulary class,
as well as an abstract one that you can extend to create your own
implementation.

Once instantiated your E2E object, just use it as a normal PyTorch DataSet: pass
it to a [DataLoader](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader)
and enjoy!

```python
# coding=utf-8
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from e2e import E2E, SetType
from lang import WordVocabulary

dataset = E2E('./data', SetType.TEST, vocabulary_class=WordVocabulary)
data_loader = DataLoader(dataset, sampler=RandomSampler(dataset))

mr, ref = next(iter(data_loader))

print('mr is a {} {} of size {}.'.format(mr.dtype, type(mr).__name__, tuple(mr.size())))
print('ref is a {} {} of size {}.'.format(ref.dtype, type(ref).__name__, tuple(ref.size())))

# Output:
# mr is a torch.int64 Tensor of size (1, 42).
# ref is a torch.int64 Tensor of size (1, 29).

```

## Generated directories
Once initialized, the E2E dataset will be organized as follows, inside the root
directory passed as a constructor parameter:
 * **csv/** contains the plain dataset's three CSV files:
    * *trainset.csv*
    * *devset.csv*
    * *testset.csv* (with reference strings as well)
 * **${vocabulary-class-name}/** contains the processed subsets as well as the
   pickled vocabulary:
    * *train.pt*
    * *dev.pt*
    * *test.pt*
    * *vocabulary.pt*
   
   Please note that the three subset files need their corresponding Vocabulary
   to be interpreted. Clearly, you can use as many vocabulary classes as you
   want, and each one will use its own folder.

## The dataset
The E2E dataset is a dataset for training end-to-end, data-driven natural 
language generation systems in the restaurant domain, which is ten times bigger 
than existing, frequently used datasets in this area. 

The E2E dataset poses new challenges: 
1) its human reference texts show more lexical richness and syntactic
   variation, including discourse phenomena;
2) generating from this set requires content selection. 

As such, learning from this dataset promises more natural, varied and less 
template-like system utterances.

The E2E set was used in the [E2E NLG Challenge](http://www.macs.hw.ac.uk/InteractionLab/E2E/),
which provides an extensive list of results achieved on this data.

Please refer to the [SIGDIAL2017 paper](https://arxiv.org/abs/1706.09254) for a
detailed description of the dataset.

## License
This is an open source project, is distributed under the [GPL v3 license](http://www.gnu.org/licenses/gpl-3.0.html).
