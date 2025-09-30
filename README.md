# Learning KAN-based Implicit Neural Representations for Deformable Image Registration

This repository contains the source code for our paper:
> **Learning KAN-based Implicit Neural Representations for Deformable Image Registration**<br>
> Nikita Drozdov,
> Marat Zinovev,
> Dmitry Sorokin
> <br>
> https://arxiv.org/abs/2509.22874

**Updates:**

- :fire: **September 2025** — Preprint of our paper is available on [arXiv](https://arxiv.org/abs/2509.22874)!

### Installation

#### 1. Install PyTorch + CUDA

Follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/previous-versions) to install PyTorch with CUDA support. We used PyTorch `2.3.0` with CUDA 12.1 and Python `3.8.10`.

#### 2. Install Other Dependencies

Run the following command:

```
pip install -r requirements.txt
```

### Data Preparation

To prepare the data, follow the instructions in the [DIR-Lab](./data/dirlab), [OASIS-1](./data/oasis), and [ACDC](./data/acdc) folders (detailed instructions are coming soon).

### Running the Code

To reproduce the results from our paper, run the corresponding script with additional arguments:

```
python run_<dataset>.py --model model_name --runs N_runs
```

Where:
- `dataset` is one of: `dirlab`, `oasis`, or `acdc`
- `model_name` is one of: `kan` (for KAN-IDIR), `rand_kan` (for RandKAN-IDIR), `a_kan` (for A-KAN-IDIR), or `idir` (for IDIR)
- `N_runs` specifies the number of runs with different random seeds

To reproduce our results, we recommend 10 runs for DIR-Lab and 3-5 runs for OASIS/ACDC.

For example, to replicate the results of the RandKAN-IDIR model on the DIR-Lab dataset, run:

```
python run_dirlab.py --model rand_kan --runs 10
```

Configurations for all models are stored in [configs/config.py](./configs/config.py).

**Note:** By default, all metrics are computed on the GPU, and memory consumption is higher during the evaluation phase than during training. To mitigate this, you can reduce the batch size for the model during validation by adjusting the `SEG_BS` and `NJD_BS` constants in [this file](./utils/eval_utils.py).

### Acknowledgments

Our work builds upon the [IDIR](https://github.com/MIAGroupUT/IDIR) codebase. [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN) repo was also very useful for implementing the KAN-based models.

### Citation

```bibtex
@misc{drozdov2025learningkanbasedimplicitneural,
      title={Learning KAN-based Implicit Neural Representations for Deformable Image Registration}, 
      author={Nikita Drozdov and Marat Zinovev and Dmitry Sorokin},
      year={2025},
      eprint={2509.22874},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.22874}, 
}
