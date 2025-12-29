# Universal Retinal Vessel Segmentation Model

## Universal Vessel Segmentation for Multi-Modality Retinal Images (TIP 2025)

- IEEE Xplore: [link](https://ieeexplore.ieee.org/document/11218739); arXiv: [link](https://arxiv.org/abs/2502.06987); 

- The three new multi-modal retinal vessel segmentation datasets are available at [Zenodo](https://zenodo.org/records/17874693).

- The revised ground truth annotation (in resolution 768x768) with improved quality (especially topological accuracy) for the DRIVE dataset is available at `./DRIVE/gt_revised_768x768`



## Get Started

Recommended dependencies:
<pre>
python==3.9.18
torch==2.1.0
numpy==1.26.0
scikit-image==0.24.0
scikit-learn==1.3.0
opencv-python==4.8.1.78
torch-topological==0.1.7
pillow==10.0.1
</pre>

### To use the model to segment retinal vessels (output segmentation only): 
1. Create a new directory in data (recommended) and copy the images (any format PIL.Image supports reading, see dataloader.py for details) in the folder
<pre>
./data/dataset_name
</pre>

2. run
<pre>
python segment.py --datapath ./data/dataset_name --note note_name
</pre>

3. The results, by default are saved to
<pre>
./result/note_name/segmentation
</pre>

### To use the model and compute the segmentation scores (ground truth available)
1. Create a new directory in data (recommended) with two folders `image` and `label`, copy the raw images and ground truth segmentation into the two folders, respectively.
<pre>
-data
  -dataset_name
    -image
      -img_01.png
      -img_02.png
      ...
    -label
      -gt_01.png
      -gt_02.png
      ...
</pre>

2. run
<pre>
python evaluate.py --datapath ./data/dataset_name --note note_name
</pre>

3. The metrics are printed as output. The segmented images, by default are saved to
<pre>
./result/note_name/segmentation
</pre>

### Training
For image translation, please refer to [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

For topological segmentation, please refer to [SATLoss](https://github.com/JRC-VPLab/SATLoss)


## 📄 Citation

If you find this work useful or use the dataset in your work, please cite:

```bibtex
@article{URVSM_TIP_2025,
  title        = {Universal Vessel Segmentation for Multi-Modality Retinal Images},
  author       = {Wen, Bo and Heinke, Anna and Agnihotri, Akshay and Bartsch, Dirk-Uwe and Freeman, William and Nguyen, Truong and An, Cheolhong},
  journal      = {IEEE Transactions on Image Processing},
  year         = {2025},
  volume       = {34},
  pages        = {7903-7918},
  doi          = {10.1109/TIP.2025.3623893}
}
```
