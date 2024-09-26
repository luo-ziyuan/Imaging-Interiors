# Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems

This is the repository that contains source code for the project page of [Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems](https://luo-ziyuan.github.io/Imaging-Interiors/).

## Quick Start
To run a case of the Austria profile with a size of 64*64, just execute the following command:
``` bash
python run_isp_double_MLP_L1.py --config configs/circle_double_MLP_Au_L1.txt
```
The data is generated based on [this GitHub repository](https://github.com/eleweiz/Solving-full-wave-nonlinear-inverse-scattering-problems-with-back-propagation-scheme).

If you find our paper useful for your work please cite:
```
@inproceedings{luo2024imaging,
  author    = {Ziyuan Luo and Boxin Shi and Haoliang Li and Renjie Wan},
  title     = {Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems},
  booktitle   = {European Conference on Computer Vision},
  year      = {2024},
}
```
