# Position-induced Transformer

The code in this repository presents the implementation of **Position-induced Transformer (PiT)** and the numerical experiments of using PiT for learing operators in partial differential equations. It is built upon the position-attention mechanism, proposed in the paper *Positional Knowledge is All You Need: Position-induced Transformer (PiT) for Operator Learning*. The paper can be accessed <a href="https://www.alphaxiv.org/pdf/2405.09285">here</a>.

## Updates May 2025
- PiT is now part of **<a href="https://github.com/AI4Equations/due">DUE</a>**, our open-source toolkit for data-driven equation modeling with modern deep learning methods. For installation and examples, visit the <a href="https://github.com/AI4Equations/due">GitHub repository</a>. 
- Added a numerical example showcasing PiT for learning the unsteady flow past a cylinder.
## Contents
- `train_burgers`: One-dimensional inviscid Burgers' equation.
- `train_sod`: One-dimensional compressible Euler equations.
- `train_darcy`: Two-dimensional Darcy flow problem.
- `train_vorticity`: Two-dimensional incompressible Navier&ndash;Stokes equations with periodic boundary conditions.
- `train_elasticity`: Two-dimensional hyper-elastic problem.
- `train_naca`: Two-dimensional transonic flow over airfoils.
- `train_cylinder`: Two-dimensional flow past cylinder at Reynolds number equal to 100.

## Datasets
The raw data required to reproduce the main results can be obtained from some of the baseline methods selected in our paper.
- For InviscidBurgers and ShockTube, datasets are provided in <a href="https://openreview.net/pdf?id=CrfhZAsJDsZ">Lanthaler et al.</a> They can be downloaded <a href="https://zenodo.org/records/7118642">here</a>.
- For Darcy2D and Vorticity, datasets are provided by <a href="https://openreview.net/pdf?id=c8P9NQVtmnO">Li et al</a>. They can be downloaded <a href="https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-">here</a>.
- For Elasticity and NACA, datasets are provided by <a href="https://www.jmlr.org/papers/volume24/23-0064/23-0064.pdf">Li et al</a>. They can be downloaded <a href="https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8">here</a>.
- For Cylinder, the dataset is generated using <a href="https://fenicsproject.org/">FEniCS</a>. It can be downloaded <a href="https://drive.google.com/drive/folders/1efL-RR_H43Pe6P5BLtcEPFgz7ZmXnl5a">here</a>.

## Requirements
- This code is primarily based on PyTorch. We have observed significant improvements in PiT's training speed with PyTorch 2.x, especially when using `torch.compile`. Therefore, we highly recommend using PyTorch 2.x with `torch.compile` enabled for optimal performance.
- If any issues arise with `torch.compile` that cannot be resolved, the code is also compatible with recent versions of PyTorch 1.x. In such cases, simply comment out the line `model = torch.compile(model)` in the scripts.
- Matplotlib and Scipy are also required.

## Citations
```
@inproceedings{chen2024positional,
               title={Positional Knowledge is All You Need: Position-induced Transformer (PiT) for Operator Learning},
               author={Junfeng Chen and Kailiang Wu},
               booktitle={International Conference on Machine Learning},
               year={2024},
               organization={PMLR}
}
```
