# TFRegNCI

## Introduction

TFRegNCI is a tool for predicting Non-covalent interaction(NCI). This program package includes NCI prediction models, 2D (TFRegNCI) and 3D (TFRegNCI_3D), and a visualization module, Grad RAM, for feature visual analyses. The corresponding codes for utility are as follows.

## Programs and Files:

1.TFRegNCI: A 2D multimodal correction model.
- TFRegNCI.py: main running file.
- TFRegNCI_model.py: 2D TFRegNCI model.
- util: Transformer utiliy package.

2.TFRegNCI-3D: A 3D Interpretable correction multimodal model.
- TFRegNCI_3D_grad_ram.py: main running file.
- TFRegNCI_3D_model.py: 3D TFRegNCI model.
- grad_ram: Grad RAM implementation package.
- util: Transformer utiliy package.

3.Model parameters: The parameter file for TFRegNCI and TFRegNCI-3D.
- TFRegNCI_para.pkl: TFRegNCI parameters
- TFRegNCI-3D_para.pkl: TFRegNCI-3D parameters

## Environment

* Python: 3.7
* Platform: pytorch 1.10

## Citation
If you find the code useful in your research, please consider citing:
@article{wang2023tfregnci,
  title={TFRegNCI: Interpretable Noncovalent Interaction Correction Multimodal Based on Transformer Encoder Fusion},
  author={Wang, Donghan and Li, Wenze and Dong, Xu and Li, Hongzhi and Hu, LiHong},
  journal={Journal of Chemical Information and Modeling},
  year={2023},
  publisher={ACS Publications}
}
