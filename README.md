# SDE-Net
This repo contains our code for paper: [SDE-Net: Equipping Deep Neural Network with Uncertainty Estimates (ICML2020)] 
![SDE-Net](figure/illustration.png)


## Training & Evaluation

#### MNIST
```
cd MNIST
```
Training vanilla DNN:
```
python resnet_mnist.py 
```
Evaluation:
```
python test_detection.py --pre_trained_net save_resnet_mnist --network resnet --dataset mnist --out_dataset svhn
```
Training SDE-Net:
```
python sdenet_mnist.py 
```
Evaluation:
```
python test_detection.py --pre_trained_net save_sdenet_mnist --network sdenet --dataset mnist --out_dataset svhn
```

#### SVHN
```
cd SVHN
```
Training vanilla DNN:
```
python resnet_svhn.py 
```
Evaluation:
```
python test_detection.py --pre_trained_net save_resnet_svhn --network resnet --dataset svhn --out_dataset cifar10
```
Training SDE-Net:
```
python sdenet_mnist.py 
```
Evaluation:
```
python test_detection.py --pre_trained_net save_sdenet_svhn --network sdenet --dataset svhn --out_dataset cifar10
```




## Citation

Please cite the following paper if you find this repo helpful. Thanks!
```
@inproceedings{kong2020sdenet,
  title={SDE-Net: Equipping Deep Neural Network with Uncertainty Estimates},
  author={Lingkai Kong, Jimeng Sun and Zhang, Chao},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2020}
}
```