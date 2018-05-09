# PyTorch E2E DataSet
The [E2E Challenge Dataset](http://www.macs.hw.ac.uk/InteractionLab/E2E/),
packed as a [PyTorch DataSet](https://pytorch.org/docs/master/data.html#torch.utils.data.Dataset)
subclass.

## Getting started
The E2E class will automatically download and process the dataset, converting the CSV files to
a list of [MR, REF] matches.

Every string is represented as a list of indices, according to the chosen Vocabulary class: a
Vocabulary object can convert lists to strings and vice versa.
You will find in *lang.py* a word-based and a character-based vocabulary class, as well as an
abstract one that you can extend to create your own implementation.

Once instantiated your e2e object, just use it as a normal PyTorch DataSet: pass it to a
[DataLoader](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) and enjoy!

## Generated directories
Once initialized, the E2E dataset will be organized as follows, inside the root directory passed
as a constructor parameter:
 * **csv/** contains the plain dataset's three CSV files:
    * *trainset.csv*
    * *devset.csv*
    * *testset.csv* (contains reference strings as well)
 * **${vocabulary-class-name}/** contains the processed subsets as well as the pickled
   vocabulary:
    * *train.pt*
    * *dev.pt*
    * *test.pt*
    * *vocabulary.pt*
   
   Please note that the three subset files need the vocabulary to be interpreted. Clearly, you
   can use as many vocabulary classes as you want, and each one will use its own folder.  

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

Please refer to the [SIGDIAL2017 paper](https://arxiv.org/abs/1706.09254) for 
a detailed description of the dataset.

## License
This is an open source project, is distributed under the [GPL v3 license](http://www.gnu.org/licenses/gpl-3.0.html).
