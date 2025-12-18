# 🛎 Citation

If you find our work helpful for your research, please cite:

```bib
@article{zheng2024affsegnet,
  title={AFFSegNet: Adaptive Feature Fusion Segmentation Network for Microtumors and Multi-Organ Segmentation},
  author={Zheng, Fuchen and Chen, Xinyi and Chen, Xuhang and Li, Haolun and Guo, Xiaojiao and Liu, Weihuang and Pun, Chi-Man and Zhou, Shoujun},
  journal={arXiv preprint arXiv:2409.07779},
  year={2024}
}
Explain
```


# 📋ASSNet

AFFSegNet: Adaptive Feature Fusion Segmentation Network for Microtumors and Multi-Organ Segmentation

[Fuchen Zheng](https://lzeeorno.github.io/), Xinyi Chen, [Xuhang Chen](https://cxh.netlify.app/), Haolun Li, Xiaojiao Guo, Guoheng Huang, [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/) 📮and [Shoujun Zhou](https://people.ucas.edu.cn/~sjzhou?language=en) 📮( 📮 Corresponding authors)

**University of Macau, SIAT CAS, Southern University of Science and Technology, Guangdong University of Technology**


## 🚧 Installation 
Requirements: `Ubuntu 20.04`

1. Create a virtual environment: `conda create -n your_environment python=3.8 -y` and `conda activate your_environment `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) :`pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118`
Or you can use Tsinghua Source for installation
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```
3. `pip install tqdm scikit-learn albumentations==1.0.3 pandas einops axial_attention`



## 1. Prepare the dataset

### LiTS2017 datasets
- The LiTS2017 datasets can be downloaded here: {[LiTS2017](https://competitions.codalab.org/competitions/17094)}.
- The Synapse dataset can be downloaded here: {[Synapse multi-organ](https://www.synapse.org/Synapse:syn3193805/wiki/217789)}.

- After downloading the datasets, you should run ./data_prepare/preprocess_lits2017_png.py to convert .nii files into .png files for training. (Save the downloaded LiTS2017 datasets in the data folder in the following format.)

- './data_prepare/'
  - preprocess_lits2017_png.py
- './data/'
  - LITS2017
    - ct
      - .nii
    - label
      - .nii
  - trainImage_lits2017_png
      - .png
  - trainImage_lits2017_png
      - .png
### Other datasets
- Other datasets just similar to LiTS2017

  
## 2. Prepare the pre_trained weights
- The weights of the pre-trained ASSNet could be downloaded [here](). After that, the pre-trained weights should be stored in './pretrained_weights/'.
- To use pre-trained file, you should change 2 places in './train/train_lits2017_png.py'
  - 1. Change 'default=True' in 'parser.add_argument('--pretrained', default=False, type=str2bool)'
  - 2. Change 'pretrained_path= "./your_pretrained_file_path"' after 'if args.model_name == 'ASSNet':'

## 3. Train the ASSNet
```bash
cd ./train/
python train_lits2017_png.py 
```

## 4. Obtain the outputs
- After trianing, you could obtain the results in './trained_models/LiTS_ASSNet/'

  



# 🧧 Acknowledgement

This work was supported in part by the National Key R\&D Project of China (2018YFA0704102, 2018YFA0704104), in part by Natural Science Foundation of Guangdong Province (No. 2023A1515010673), and in part by Shenzhen Technology Innovation Commission (No. JSGG20220831110400001), in part by Shenzhen Development and Reform Commission (No. XMHT20220104009), in part by the Science and Technology Development Fund, Macau SAR, under Grant 0141/2023/RIA2 and 0193/2023/RIA3.



