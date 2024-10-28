# Position-induced Transformer

The code in this repository presents six numerical experiments of using Position-induced Transformer (PiT) for learing operators in partial differential equations. PiT is built upon the position-attention mechanism, proposed in the paper *Positional Knowledge is All You Need: Position-induced Transformer (PiT) for Operator Learning*. The paper can be downloaded <a href="https://arxiv.org/pdf/2405.09285">here</a>.

## Contents
- The numerical experiment on the one-dimensional inviscid Burgers' equation.
- The numerical experiment on the one-dimensional compressible Euler equations.
- The numerical experiment on the two-dimensional Darcy flow problem.
- The numerical experiment on the two-dimensional incompressible Navier&ndash;Stokes equations.
- The numerical experiment on the two-dimensional hyper-elastic problem.
- The numerical experiment on the two-dimensional compressible Euler equations.

## Datasets
The raw data required to reproduce the main results can be obtained from some of the baseline methods selected in our paper.
- For InviscidBurgers and ShockTube, data sets are provided in <a href="https://openreview.net/pdf?id=CrfhZAsJDsZ">Lanthaler et al.</a> They can be downloaded <a href="https://zenodo.org/records/7118642">here</a>.
- For Darcy2D and Vorticity, data sets are provided by <a href="https://openreview.net/pdf?id=c8P9NQVtmnO">Li et al</a>. They can be downloaded <a href="https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-">here</a>.
- For Elasticity and NACA, data sets are provided by <a href="https://www.jmlr.org/papers/volume24/23-0064/23-0064.pdf">Li et al</a>. They can be downloaded <a href="https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8">here</a>.

## Requirements
- This code is primarily based on PyTorch. We have observed significant improvements in PiT's training speed with PyTorch 2.X, especially when using `torch.compile`. Therefore, we highly recommend using PyTorch 2.X with `torch.compile` enabled for optimal performance.
- If any issues arise with `torch.compile` that cannot be resolved, the code is also compatible with recent versions of PyTorch 1.X. In such cases, simply comment out the line `model = torch.compile(model)` in the scripts.

## Citations
```
@inproceedings{chen2024positional,
               title={Positional Knowledge is All You Need: Position-induced Transformer (PiT) for Operator Learning},
               author={Junfeng Chen and Kailiang Wu},
               booktitle={International conference on machine learning},
               year={2024},
               organization={PMLR}
}
```
