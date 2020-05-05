# DeepFovea++ for Video Reconstruction and Super Resolution

This repository tries to solve the task of fovea sampled reconstruction and video super resolution with partly based on the
architecture of the [Deep Fovea](https://research.fb.com/wp-content/uploads/2019/11/DeepFovea-Neural-Reconstruction-for-Foveated-Rendering-and-Video-Compression-using-Learned-Statistics-of-Natural-Videos.pdf?) 
paper by Anton S. Kaplanyan et al. ([facebook research](https://research.fb.com/)). [1]

## Model Architecture

![Generator model](img/g_model.png)
![Losses](img/losses.png)
[Source](https://github.com/facebookresearch/DeepFovea) [1]

To reach the desired super-resolution (4 times higher than the input) two additional blocks are used, in the end of the 
generator network. This so called super-resolution blocks are based on two (for the final block three) 
[deformable convolutions](https://arxiv.org/abs/1811.11168) and an bilinear upsampling. [1]

## Losses

We applied the same losses as the paper, but we also added a supervised loss. For this supervised loss, the adaptive 
robust loss function by Jonathan T. Barron was chosen. [1, 4]

<img src="https://render.githubusercontent.com/render/math?math=L_{G}=w_{sv}\cdot L_{sv} %2B w_{LPIPS}\cdot L_{LPIPS} %2B w_{flow}\cdot L_{flow} %2B w_{adv}\cdot L_{adv}">

We compute each loss independent, since otherwise we run into a VRAM issue (Tesla V100 16GB).

## Dependencies

This implementation used the [adaptive robust loss](https://arxiv.org/abs/1701.03077) 
[implementation](https://github.com/jonbarron/robust_loss_pytorch) 
by [Jonathan T. Barron](https://github.com/jonbarron/robust_loss_pytorch). Furthermore, 
[deformable convolutions V2](https://arxiv.org/abs/1811.11168) are used in the generator network. 
Therefor the [implementation](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0) 
of [Dazhi Cheng](https://github.com/chengdazhi) is utilized.
For the [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/PyTorch) 
the implementation and pre-trained weights of [Nvidia Research](https://github.com/NVlabs) is used. 
Additionally the PWC-Net and the flow loss implementation depends on the 
[correlation](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package), and 
[resample](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/resample2d_package) package 
of the PyTorch [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks) 
implementation by Nvidia. To install the packages run `python setup.py install` for each package. The setup.py file
is located in the corresponding folder. [4, 2, 6, 5]

All additional required packages can be found in [requirements.txt](requirements.txt).
To install the additional required packages simply run `pip install -r requirements.txt`.

### Full installation

```shell script
git clone https://github.com/ChristophReich1996/Deep_Fovea_Architecture_for_Video_Super_Resolution
cd Deep_Fovea_Architecture_for_Video_Super_Resolution
pip install -r requirements.txt
python correlation/setup.py install
python resample/setup.py install
```

## Results

We sampled approximately **19.7%** of the low resolution (192 X 256) input image, when apply the fovea sampling. This 19.7% 
corresponds to **1.2%** when compared to the high resolution (768 X 1024) label.

Results of the training run started at the 02.05.2020. For this training run the recurrent tensor of each temporal block 
was reset after each full video.

![plots02052020input](results/2020-05-02/plots/input_220_2020-05-04%2011_17_59.593499.png)
Low resolution (192 X 256) fovea sampled input image
![plots02052020pred](results/2020-05-02/plots/prediction_220_2020-05-04%2011_17_55.343509.png)
High resolution (768 X 1024) reconstructed prediction of the generator
![plots02052020label](results/2020-05-02/plots/label_220_2020-05-04%2011_17_57.695080.png)
High resolution (768 X 1024) label [3]

The corresponding pre-trained models, additional plots and all metrics can be found in the folder 
[results/2020-05-02](results/2020-05-02).

## References

```
[1] @article{deepfovea,
    title={DeepFovea: neural reconstruction for foveated rendering and video compression using learned statistics of natural videos},
    author={Kaplanyan, Anton S and Sochenov, Anton and Leimk{\"u}hler, Thomas and Okunev, Mikhail and Goodall, Todd and Rufo, Gizem},
    journal={ACM Transactions on Graphics (TOG)},
    volume={38},
    number={6},
    pages={1--13},
    year={2019},
    publisher={ACM New York, NY, USA}
}
```

```
[2] @inproceedings{deformableconv2,
    title={Deformable convnets v2: More deformable, better results},
    author={Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={9308--9316},
    year={2019}
}
```

```
[3] @InProceedings{reds,
    author = {Nah, Seungjun and Baik, Sungyong and Hong, Seokil and Moon, Gyeongsik and Son, Sanghyun and Timofte, Radu and Lee, Kyoung Mu},
    title = {NTIRE 2019 Challenge on Video Deblurring and Super-Resolution: Dataset and Study},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2019}
}
```

```
[4] @inproceedings{adaptiveroubustloss,
    title={A general and adaptive robust loss function},
    author={Barron, Jonathan T},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={4331--4339},
    year={2019}
}
```

```
[5] @inproceedings{flownet2,
    title={Flownet 2.0: Evolution of optical flow estimation with deep networks},
    author={Ilg, Eddy and Mayer, Nikolaus and Saikia, Tonmoy and Keuper, Margret and Dosovitskiy, Alexey and Brox, Thomas},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={2462--2470},
    year={2017}
}
```

```
[6] @inproceedings{pwcnet,
    title={Pwc-net: Cnns for optical flow using pyramid, warping, and cost volume},
    author={Sun, Deqing and Yang, Xiaodong and Liu, Ming-Yu and Kautz, Jan},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={8934--8943},
    year={2018}
}
```
