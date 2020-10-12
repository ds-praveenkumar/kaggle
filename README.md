# Kaggke cli commands

## Competitions
* `kaggle competitions list`: list the currently active competitions
* `kaggle competitions download -c [COMPETITION]`: download files associated with a competition
* `kaggle competitions submit -c [COMPETITION] -f [FILE] -m [MESSAGE]`: make a competition submission

### Submitting to a Competition
- `kaggle competitions submit -c [COMPETITION NAME] -f [FILE PATH]`
- `kaggle competitions submissions -c [COMPETITION NAME]`

### Interacting with Datasets
- `kaggle datasets list -s [KEYWORD]`: list datasets matching a search term
- `kaggle datasets download -d [DATASET]`: download files associated with a dataset

## Creating and Maintaining Datasets

### Create a New Dataset
- `kaggle datasets init -p /path/to/dataset` to generate a metadata file
- `kaggle datasets create -p /path/to/dataset` to create the dataset

## Interacting with Notebooks

- `kaggle kernels list -s [KEYWORD]`: list Notebooks matching a search term
- `kaggle kernels push -k [KERNEL] -p /path/to/kernel` : create and run a Notebook on Kaggle
- `kaggle kernels pull -k [KERNEL] -p /path/to/download -m`: download code files and metadata associated with a Notebook
- kaggle kernels pull  -p <path-to-download-kernel>
