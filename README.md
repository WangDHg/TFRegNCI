# TFRegNCI
Code of TFRegNCI and TFRegNCI-3D

## Introduction

TFRegNCI is a tool for predicting Non-covalent interaction(NCI), which mainly use RegNet and MLP. Because of the electron density can be location the meaningful region by Grad RAM, We combine the TFRegNCI-3D and Grad RAM to available the Interpretability model for visualize the important features in 3D feature space.

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
