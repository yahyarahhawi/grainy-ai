
## Initial ReadME for deep learning class grading

## Acknowledgements

 - [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
 - [Shrodinger Bridge Style Transfer](https://arxiv.org/html/2305.15086v3)


For running inference, pretrained weights can be obtained [here](https://drive.google.com/drive/folders/1quxKnOvxVHPVaXo-K-gNmRQ1sFGfDtFX?usp=drive_link)

Please review the following code that I have written: 
1. /Filmic-ai/pytorch-CycleGAN-and-pix2pix/filmic.py contains all the image processing code I wrote to process the model output. This library's use is demonstrated in the inference
2. /Filmic-ai/pytorch-CycleGAN-and-pix2pix/inference.ipynb contains the inference of cycleGAN and can be used to generate images with the help of filmic.py

3. /Filmic-ai/UNSB/inference02.ipynb contains inference for Schrodinger Bridge.

Additionally, there are multiple util files created, such as pytorch-to-coreML conversion, and other eperiments I have done related to simulating film.

please refer to the original implementations of models to learn how to train