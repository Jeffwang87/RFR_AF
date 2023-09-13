
### Introduction
This repository supports the ICLR 2023 paper titled "Optimal Activation Functions for the Random Features".

### Prerequisite
Mathlab, Mathematica, Pytorch, and Sklearn

### Code for Figure 1, Figure 4 and Figure 5
To generate Figure 1 please run RunMeToGenerateFigure1.nb. It requires Wolfram Mathematica v12.	

To generate Figure 4 please run RunMeToGenerateFigure4.m. It requires Matlab 2020b.	

To generate Figure 5 please run RunMeToGenerateFigure5.m. It requires Matlab 2020b.	

### Code for Theorems Checking
it contains code to symbolically check the formulas in our Theorems 9, 10 and 11.
These are files RunMeToCheckProofOfTheorem9.nb, RunMeToCheckProofOfTheorem10.nb, and RunMeToCheckProofOfTheorem11.nb.

### Code for Figure 6
To run figure for regime 1, run
`python RFR_LR_r1.py --F_1 {} --d {} --F_star {} --psi_2 {} --tau {}`

To run figure for regime 2, run
`python RFR_LR_r2.py --F_1 {} --d {} --F_star {} --psi_1 {} --tau {} --lambda_i {}`

To run figure for regime 3, run
`python RFR_LR_r3.py --F_1 {} --d {} --F_star {} --psi_2 {} --tau {} --lambda_i {}`

{} corresponds to custom numerical settings

All code is distributed under an MIT License. If you find our work or this codebase useful for your research, please cite

```
@inproceedings{
wang2023optimal,
 title={Optimal Activation Functions for the Random Features Regression Model},
 author={Jianxin Wang and Jos{\'e} Bento},
 booktitle={The Eleventh International Conference on Learning Representations },
 year={2023},
}
```
If you have any other questions, please raise the issue
