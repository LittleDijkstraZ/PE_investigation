# How does Transformers without PE Learn Positional Information?

## Overview
---
This repository contains the code for our investigation of positional encoding in Transformers.

## Important files
- [trian.py](./train.py) contains the training loop.
- [pe_info](./pe_info) contains our implementation of the experiments
- [pe_info/model_nope.py](./pe_info/model_nope.py) this is the one and only file for the model, which has modifiable PE, SC, and more.
- [pe_info_experiments](./pe_info_experiments) contains the experiments we have run.
- [pe_info_experiments/tuning.py](./pe_info_experiments/tuning.py) contains the code for tuning the hyperparameters.

### Notes on the implementation 
---
This codebase forked from [teaching arithmetic](https://github.com/lee-ny/teaching_arithmetic)
- [Here](teaching_arithmetic_pe/pe_info) contains our modifications.
- NOPE has been implemented.
- Control on Layerwise positional encoding has been added.