# PyTorch E2E DataSet
The [E2E Challenge Dataset](http://www.macs.hw.ac.uk/InteractionLab/E2E/),
packed as a PyTorch [```DataSet```](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset)
subclass.

## Getting started
The ```E2E``` class will automatically download and process the dataset,
converting the CSV files to a list of [MR, REF] matches, sorted by ascending MR
length.

Every string is represented as a list of the chosen ```Vocabulary``` class'
keys (indices): a ```Vocabulary``` object can convert lists to strings and vice
versa. You will find in *lang.py* a word-based and a character-based vocabulary
class, as well as an abstract one that you can extend to create your own
implementation.

Once instantiated your ```E2E``` object, just use it as a normal PyTorch
```DataSet```: pass it to a [```DataLoader```](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader)
and enjoy!

### Example
```python
# coding=utf-8
from torch.utils.data import DataLoader

from e2e import E2E, E2ESet
from lang import WordVocabulary

dataset = E2E('./data', E2ESet.TEST, vocabulary_class=WordVocabulary)
data_loader = DataLoader(dataset, shuffle=True)

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
    * *all_in_one.csv* (the union of all the previous subsets)
 * **${vocabulary-class-name}/** contains the processed subsets as well as the
   vocabulary, as pickled files:
    * *train.pt*
    * *dev.pt*
    * *test.pt*
    * *all_in_one.pt*
    * *vocabulary.pt*
   
   Please note that the three subset files need their corresponding Vocabulary
   to be interpreted. Clearly, you can use as many vocabulary classes as you
   want, and each one will use its own folder.

## The ```E2EAttrSplit``` subclass
The vanilla E2E dataset's attributes share a lot of values between the three
subsets. This makes it difficult to test whether your model copies the input
values directly to the output sequence or if it just generates something it has
learnt during the training phase.

You can use the ```E2EAttrSplit``` class to prevent this issue: its constructor
requires an attribute on the basis of which the training and development set
will be created, the ratio between them and a tolerance. For instance,
```training_ratio=0.9``` and ```tolerance=0.01``` will generate a training set
containing (90 ± 0.9)% of the examples, and a development set containing all
the remaining ones.

The attributes are encapsulated in the ```E2EAttribute``` enumerator. Not that
*familyFriendly* and *area* are not included in it, because they have only two
possible values, which makes the corresponding split datasets unuseful. 

You can switch between the training set and the test set through the
```choose_set``` method.

### Example
```python
# coding=utf-8
from e2e import E2ESet
from e2e_attr_split import E2EAttrSplit, E2EAttribute
from lang import CharVocabulary

dataset = E2EAttrSplit('./data', E2ESet.TRAIN, CharVocabulary,
                       attribute=E2EAttribute.NAME, training_ratio=0.7, tolerance=0.05)

#
# Train your model here...
#

dataset.choose_set(E2ESet.DEV)

#
# Now you can test if your model can copy - not generate - restaurant names
# 

```

### Generated directories
The ```E2EAttrSplit``` subclass will create two additional files inside the
**${vocabulary-class-name}/** directory, *partitions.pt* and
*value_dictionaries.pt*, that contain some useful information to coherently
reuse partitions and avoid to repeat the same data processing from one
execution to another.

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
