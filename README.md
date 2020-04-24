# Deep Fovea Architecture for Video Super Resolution

This repository tries to solve the task of video super resolution with main architecture of the 
[Deep Fovea](https://research.fb.com/wp-content/uploads/2019/11/DeepFovea-Neural-Reconstruction-for-Foveated-Rendering-and-Video-Compression-using-Learned-Statistics-of-Natural-Videos.pdf?) 
paper by Anton S. Kaplanyan et al. (facebook research).

## TODO

* Find dataset ([youtube 8M](https://research.google.com/youtube8m/) or [cityscapes](https://www.cityscapes-dataset.com/))
* Implement dataset class for chosen dataset
* Implement [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) and [SSIM](https://en.wikipedia.org/wiki/Structural_similarity)
* Implement validation method (L1, L2, PSNR and SSIM for validation?)
* Implement test method
* Implement inference method
* Analyse results and HPO if needed ;)

## Model Architecture
![Generator model](img/g_model.png)
![Losses](img/losses.png)
[Source](https://github.com/facebookresearch/DeepFovea)
