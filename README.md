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
```DataSet```. If you want to take advantage of a [```DataLoader```](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader),
keep in mind that you can use the ```E2EDataLoader``` class, that wraps a
convenient collate function. Every batch it returns is a tuple containing
 * two lists (MRs and REFs) made up of ```batch_size``` lists (```max_mr_len```
   and ```max_ref_len``` long, respectively);
 * the list of the ```batch_size``` lengths of each MR;
 * the maximum REF size.
Note that the conversion from integer lists to PyTorch tensors it is up to you.

### Example
```python
# coding=utf-8

from e2e import E2E, E2ESet
from e2e_loader import E2EDataLoader
from lang import WordVocabulary

dataset = E2E('./data', E2ESet.TEST, vocabulary_class=WordVocabulary)
data_loader = E2EDataLoader(dataset, batch_size=10)

batch = next(iter(data_loader))
mrs, refs, mr_len, ref_max_len = batch
print(f'My batch is a {type(batch).__name__} containing {len(batch)} items:\n'
      f'\tmrs, a {type(mrs).__name__} of {len(mrs)} {type(mrs[0][0]).__name__} {type(mrs[0]).__name__}s\n'
      f'\trefs, a {type(refs).__name__} of {len(refs)} {type(refs[0][0]).__name__} {type(refs[0]).__name__}s\n'
      f'\tmr_lengths, a {type(mr_len).__name__} of {len(mr_len)} {type(mr_len[0]).__name__}s\n'
      f'\tref_max_length, an {type(ref_max_len).__name__}\n')

mr = mrs[0]
ref = refs[0]
print(f'mr is a {type(mr[0]).__name__} {type(mr).__name__} of size {len(mr)}:\n\t{dataset.to_string(mr)}')
print(f'ref is a {type(ref[0]).__name__} {type(ref).__name__} of size {len(ref)}:\n\t{dataset.to_string(ref)}')

# Output:
# My batch is a tuple containing 4 items:
#    mrs, a list of 10 int lists
#    refs, a list of 10 int lists
#    mr_lengths, a list of 10 ints
#    ref_max_length, an int
#
# mr is a int list of size [16]:
#    name [ Blue Spice ] , eatType [ pub ] , area [ riverside ]
# ref is a int list of size [14]:
#    There is a pub Blue Spice in the riverside area .
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
The ```E2EAttrSplit``` subclass will create an additional file inside the
**${vocabulary-class-name}/** directory, *partitions.pt*, that contains some
useful information to coherently reuse partitions and avoid to repeat the same
data processing from one execution to another.

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
