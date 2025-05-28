# SHADeS
This is the official PyTorch implementation for training and testing depth estimation models using the method described in
> [**SHADeS: self-supervised monocular depth estimation through non-Lambertian image decomposition**](https://rdcu.be/eob6R)

> Rema Daher, Francisco Vasconcelos and Danail Stoyanov 

## Overview 
![image](figure/flowchart.png)
## 📄 Citation
If you find our work useful in your research please consider citing our paper:
```
@article{daher2025shades,
  title={SHADeS: self-supervised monocular depth estimation through non-Lambertian image decomposition},
  author={Daher, Rema and Vasconcelos, Francisco and Stoyanov, Danail},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--9},
  year={2025},
  publisher={Springer}
}
```

## 📦 Requirements
Use Python 3.7 and install the following packages:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8
pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image
pip3 install mmcv-full==1.3.0 mmsegmentation==0.11.0  
pip install timm einops IPython
pip install protobuf==3.20.0
pip install gdown
pip install scikit-learn
pip install tensorboard
```

## 💾 Datasets
For the Dataset setup, refer to [prepare_data/README.md](prepare_data/README.md).


### Split
The train/test/validation split for C3VD and Hyper Kvasir dataset used in our works is defined in the  [splits](splits) folder. We train our model on Hyper Kvasir and test on both C3VD and Hyper Kvasir datasets. 

## 🖼️ Inference & Evaluation
You can download our [pretrained models](https://liveuclac-my.sharepoint.com/my?id=%2Fpersonal%2Fucabrd0%5Fucl%5Fac%5Fuk%2FDocuments%2FPretrainedModel%2DEndoSTTN%2FSHADeS%2FModels&ga=1).


For C3VD dataset:
```
python test_simple.py --image_path <your_image_or_folder_path> --model_name <depth_model_name> --method <method_name> --model_basepath <depth_model_base_path> --output_path <path_to_save_results> --save_triplet --eval --maxing --seq all
```
For Hyper Kvasir dataset:
```
python test_simple.py --split_path splits/hk/test_files.txt --image_path <your_image_or_folder_path> --model_name <depth_model_name> --method <method_name> --model_basepath <depth_model_base_path> --output_path <path_to_save_results> --save_depth --eval --maxing

```

- `<method_name>` can be `monovit`, `monodepth2`, or `IID` for MonoViT, Monodepth2, and SHADeS respectively. When using MonoViT or Monodepth2, set `--max_depth` to 100.

- If testing on MonoViT, pLease download the ImageNet-1K pretrained MPViT [model](https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth) to `./ckpt/`.

- Add `--decompose` to decompose the images into albedo and shading.


### Additional scripts
We also include some helpful scripts to get tables with: scores, depth figures, decomposition figures, specular masks, or reconstruction figures for different methods. They should be edited to set the correct paths and method names. You can find these scripts in the [scripts](scripts) folder.


## ⏳ Training
For initializing the model, you can use the pretrained weights of [IID-SfmLearner](https://github.com/bobo909/IID-SfmLearner). However, they only provide `depth.pth` and `encoder.pth`. Thus, use the `pose.pth` and `pose_encoder.pth` from [Monodepth2 mono_640x192 model](https://github.com/nianticlabs/monodepth2). After downloading the pretrained weights, you can set their directory in the configuration file. We provide a sample configuration file in [configs](configs). You can modify the parameters in this file according to your needs.

You can train a model by running the following command:
```
python train.py --config <config_file_path> 
```



## ✏️Acknowledgement
Our code is based on the implementation of [IID-SfmLearner](https://github.com/bobo909/IID-SfmLearner), [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner), [Monodepth2](https://github.com/nianticlabs/monodepth2) and [MonoViT](https://github.com/zxcqlf/MonoViT). We thank these authors for their excellent work and repository.
