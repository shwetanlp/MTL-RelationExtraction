# MTL-RelationExtraction


Relation Extraction from Biomedical and ClinicalText: Unified Multitask Learning Framework

Requirements

python 3.7

Please run the following command from the directory.
```
pip install -r requirements.txt
```


# How to Run:
 * Please download the pre-trained word embedding from [here](http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin) and put it in the directory data.
* Download all the dataset AiMED, BioInfer, i2b2 2010clinical information challenge and Semeval 2013 DDIExtraction challenge dataset.
* Split the data into train and test set and places them under the directory "data".
* The dataset files should be in the following file name structure:

     <dataset_name>-<fold_number>.train
     
     <dataset_name>-<fold_number>.test

     E.g.  aimed-2.train, aimed-2.test
     
     An example of the sample file is given in the directory "data" with the name “sample-data.txt”

* Go the “src” directory.
* Run the following command:
     ```
     python main.py
     ```

## Reference

If you are using this code then please cite our paper:



```
S. Yadav, S. Ramesh, S. Saha and A. Ekbal, "Relation Extraction from Biomedical and Clinical Text: Unified Multitask Learning Framework," in IEEE/ACM Transactions on Computational Biology and Bioinformatics, doi: 10.1109/TCBB.2020.3020016.

```


## License
This code is distributed under the [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.
