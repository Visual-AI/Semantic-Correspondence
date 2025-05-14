# Semantic Correspondence: Unified Benchmarking and a Strong Baseline

**📚[Paper List](paper_list.md) | [Arxiv]()**

[Kaiyan Zhang<sup>1</sup>](https://scholar.google.com.hk/citations?user=ef255KYAAAAJ&hl=en)
Jingyi Lu<sup>2</sup>, 
[Xinghui Li<sup>2</sup>](https://scholar.google.com/citations?user=XLlgbBoAAAAJ&hl=en),
[Kai Han<sup>1</sup>](https://www.kaihan.org/), 

[<sup>1</sup>Visual AI Lab, University of Hong Kong](https://visailab.github.io/)&nbsp;&nbsp;&nbsp;
[<sup>2</sup>Active Vision Lab
, University of Oxford](https://www.robots.ox.ac.uk/~lav/)


## Environment
The environment can be easily installed through [conda](https://docs.conda.io/projects/miniconda/en/latest/) and pip. After downloading the code, run the following command:
```shell
$conda create -n sc_baseline python=3.10
$conda activate sc_baseline

$conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
$conda install xformers -c xformers
$pip install yacs pandas scipy einops matplotlib triton timm diffusers accelerate transformers datasets tensorboard pykeops scikit-learn
```

## Data
Download the dataset you need under the 'asset' folder.

#### PF-Pascal
1. Download PF-Pascal dataset from [link](https://www.di.ens.fr/willow/research/proposalflow/).
2. Rename the outermost directory from `PF-dataset-PASCAL` to `pf-pascal`.
3. Download lists for image pairs from [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/pf-pascal_image_pairs.zip).
4. Place the lists for image pairs under `pf-pascal` directory.

#### PF-Willow
1. Download PF-Willow dataset from the [link](https://www.di.ens.fr/willow/research/proposalflow/).
2. Rename the outermost directory from `PF-dataset` to `pf-willow`.
3. Download lists for image pairs from [link](https://www.robots.ox.ac.uk/~xinghui/sd4match/test_pairs.csv).
4. Place the lists for image pairs under `pf-willow` directory.

#### SPair-71k
Download SPair-71k dataset from [link](https://cvlab.postech.ac.kr/research/SPair-71k/). After extraction,  No more action required.

#### AP-10k
Follow the instrcution of [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC) to prepare for the AP-10k dataset.

The structure should be :
```
asset
├── ap-10k
│   ├── annotations
│   ├── ImageAnnotation
│   ├── JPEGImages
│   ├── PairAnnotation
├── pf-pascal
│   ├── PF-dataset-PASCAL
│   │   ├── test_pairs.csv
│   │   ├── trn_pairs.csv
│   │   └── val_pairs.csv
├── pf-willow
│   ├── PF-dataset
│   │   └── test_pairs.csv
└── SPair-71k
    ├── devkit
    ├── ImageAnnotation
    ├── JPEGImages
    ├── Layout
    ├── PairAnnotation
    ├── Segmentation
    └── Visualization
```



## Training
The configuration file for training and testing can be access at config/base.py.
For example, to train the model, run:
```
sh train.sh
```

Some important parameters here include:
- `dataset`: dataset name, choose from 'spair', 'ap10k', 'pfwillow' or 'pfpascal'.
- `method`: set to 'dino' to use DINOv2 as the backbone.
- `pre_extract`: pre-extract image features to speed up validation.
- `train_sample` and `val_sample`: only used for the AP-10k dataset.`
- `save_thre`: threshold for saving the model within an epoch.
- `eval_interval`: iteration interval for validation.
- `ckpt_dir`: directory to save the model, train log and evaluation log.
- `resume_dir`: directory to resume training. If starting from scratch, set to 'None'.


## Testing
```
python test.py --dataset ap10k  --method dino --resolution 840  --batch_size 4 --ckpt_dir $directory_of_the_model$
```

We provided pretrained weights to reproduce the results in the paper, you can download it here.  


|      |   SPair-71k |  |  AP-10k |   |
| ---- | ---- | ---- |---- |---- |
|   Ours(DINOv2)   |  85.1% | [Google Drive](https://drive.google.com/drive/folders/1XgHdxYvan_LB85RpxkDKlEmXElTOC5D8?usp=drive_link) | 87.4% | [Google Drive](https://drive.google.com/drive/folders/1EQbuQFZ4CyjqvlohaZROXlYyNASxich9?usp=drive_link)   |


Note: Results may vary a bit depending on the GPU. 

## Citation
```
```
