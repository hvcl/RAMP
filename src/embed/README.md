# Environment Settings for Embedding Generation

Before explaining the setup below, we would like to inform you that you can test response prediction of RAMP using the pretrained embedding vectors existing in the 'RAMP/results/embedding' directory. The following steps are necessary to train a new embedding vectors with your own datasets.

## 1. Requirements

- OS : Ubuntu 16.04
- Python : Python 3.6

When using Docker, we tested our method with TensorFlow's image, tagged with `2.5.0-custom-op-gpu-ubuntu16`.

### 1-1. Python Libraries Installation

Required Python libraries are listed in 'RAMP/src/embed/requirements.txt' file. Use `pip install -r requirements.txt` command to install libs to your virtual environment.

```
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.6
sudo apt install libpython3.6-dev

virtualenv -p python3.6 ramp
source ramp/bin/activate
pip install -r requirements.txt
```

## 2. RA-NS

There are several works to be done for using RA-NS. We modified Gensim's training process utilized by Node2Vec, and we made the RA-NS works with Cython which requires new build task. After building RA-NS codes, you have to copy few files to original Gensim & Node2Vec libraries' path.

### 2-1. Build

```
$ cd src/embed/EnvCodes/Gensim
$ cp 10_10_well/word2vec_corpusfile.cpp gensim-negativesampling/gensim/models
$ cp 10_10_well/word2vec_inner.c gensim-negativesampling/gensim/models

$ cd gensim-negativesampling
$ python setup.py build_ext --inplace
```

If build is successful, 8 `.so` files will be generated to 'build/lib.linux-x86_64-3.6/gensim/models'.

### 2-2. Copy Files

For the last, you have to copy the 8 files from build and several extra files to your Gensim and Node2vec path. You can check the packages' installation path by `python -m site` command.

```
$ cd [RAMP_path]/src/EnvCodes
$ cp Node2Vec/node2vec.py [site_packages_path]/node2vec
$ cp Node2Vec/parallel.py [site_packages_path]/node2vec

$ cd Gensim
$ cp base_any2vec.py [site_packages_path]/gensim/models
$ cp word2vec.py [site_packages_path]/gensim/models

$ cp 10_10_well/word2vec_corpusfile.cpp [site_packages_path]/gensim/models
$ cp 10_10_well/word2vec_inner.c [site_packages_path]/gensim/models

$ cd gensim-negativesampling/build/lib.linux-x86_64-3.6/gensim/models
$ cp *.so [lib_path]/gensim/models
```

## 3. Train Embedding Vectors

### 3-1. To Use GDSC or TCGA Datasets

To test the module with our datasets, just folow the commands below.

```
$ cd scripts
# Train GDSC embedding vectors with legacy Negative Sampling
$ . ./generate_embedding.sh GDSC

# Train GDSC embedding vectors with RA-NS
$ . ./generate_embedding.sh GDSC RANS

# Train TCGA embedding vectors with RA-NS via External Validation 
$ . ./generate_embedding.sh GDSC_TCGA RANS
```

Trained embedding vectors will be saved into 'RAMP/results/embedding'.

To do external validation, the dataset to be validated have to be given as an argument like `BaseDataset_ValDataset`. The example command is shown in the code block above as `GDSC_TCGA`.

To try other datasets, you have to locate the datasets into 'RAMP/data', and just run the script file by replacing GDSC to the new dataset's directory name.

### 3-2. Running without Script

Note that you have to use absolute path for the `data_path` argument. To use relative path, the relativity must be based on the main.py file's location.