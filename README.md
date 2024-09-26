# Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems

This is the repository that contains source code for the project page of [Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems](https://luo-ziyuan.github.io/Imaging-Interiors/).

## Quick Start
To run a case of the Austria profile with a size of 64x64, just follow the following instructions.

### Data preparation
The data is generated based on [this GitHub repository](https://github.com/eleweiz/Solving-full-wave-nonlinear-inverse-scattering-problems-with-back-propagation-scheme). You can download the data for the Austria profile from [this Google Drive link](https://drive.google.com/file/d/11pSTFg9QkP1Zjnl6Usc2dwJNMl4_8VcD/view?usp=sharing) and place the data under the `./data/` folder.

### Run the code
Execute the following command:
```bash
python run_isp_double_MLP_L1.py --config configs/circle_double_MLP_Au_L1.txt
```

If you find our paper useful for your work please cite:
```
@inproceedings{luo2024imaging,
  author    = {Ziyuan Luo and Boxin Shi and Haoliang Li and Renjie Wan},
  title     = {Imaging Interiors: An Implicit Solution to Electromagnetic Inverse Scattering Problems},
  booktitle   = {European Conference on Computer Vision},
  year      = {2024},
}
```
