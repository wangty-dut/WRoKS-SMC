# 🔬 A Weak Robust Modeling and Knowledge-driven Solution Approach for Scheduling Margin Calculation of Hot Rolling Process
This repository is the code implementation of the paper **"A Weak Robust Modeling and Knowledge-driven Solution Approach for Scheduling Margin Calculation of Hot Rolling Process"**.
## Project Introduction
This project includes experimental code for one dataset, using production data from the 2250 continuous casting and hot rolling line of a domestic steel company.   
The experiment mainly includes the following modules:

- **NEH-based expert trajectory generation** (Includes Python code)
- **MH-GCN state encoding**  
- **Multi-head attention (MHA) decoding**  
- **Knowledge-driven imitation learning training**  
- **test**

## file structure
```plaintext
project_root/
│
│── data/                               # Input datasets / instances for HSMSP
│
│── hot_strip_mill_scheduling_problem/  # WRoKS-SMC implementation for HSMSP
│   │
│   ├── logs/
│   │   └── HSMSP/                      # Training logs and model checkpoints
│   │
│   ├── nets/                           # Neural network definitions
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   └── MHGCN_encoder.py        # MH-GCN encoder for graph-based state representation
│   │   │
│   │   └── attention_model.py          # Multi-head attention decoder (policy network)
│   │
│   ├── outputs/
│   │   └── HSMSP/                      # Saved schedules, evaluation metrics, and result files
│   │
│   ├── problems/                       # Problem formulation and expert policy
│   │   └── HSMSP/
│   │       ├── __init__.py
│   │       ├── neh_run.py              # NEH heuristic: generate expert schedules and trajectories
│   │       ├── penalty_func.py         # Penalty / reward functions for the WRO objective
│   │       ├── problem_hsmsp.py        # HSMSP weak-robust optimization problem definition
│   │       └── state_hsmsp.py          # MDP state construction (U_t, V_t, time-margin features, etc.)
│   │
│   ├── utils/                          # Utility functions and helper modules
│   │   ├── __init__.py
│   │   ├── beam_search.py              # Beam search and inference helpers
│   │   ├── boolmask.py                 # Boolean masking utilities (e.g., mask for scheduled slabs)
│   │   ├── data_utils.py               # Data loading, preprocessing, batch construction
│   │   ├── functions.py                # General-purpose helper functions
│   │   ├── lexsor.py                   # Sorting / permutation utilities
│   │   ├── log_utils.py                # Logging and experiment tracking
│   │   └── tensor_functions.py         # Tensor operations and numerical utilities
│   │
│   ├── main.py                         # Main entry point: parse options and launch training / testing
│   ├── options.py                      # Command-line options and experiment configuration
│   └── train.py                        # Knowledge-driven imitation learning (NEH + MH-GCN + MHA)
│── README.md
└── requirements.txt
```

## Environmental Requirements
- Python 3.7
- matplotlib==3.5.2
- numpy==1.21.6
- pandas==1.3.4
- torch==1.9.0+cu111
- tensorboard==1.14.0
- scipy==1.6.0
- tqdm==4.64.1

Other dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## run steps
**HSMSP experiment**

1.End-to-End Training & Testing (Full Pipeline)
```bash
python main.py
```
Description: Main entry point for complete training, validation, and testing workflow (full WRoKS-SMC pipeline)

2.Generate Expert Data (NEH Heuristic)
```bash
python -m problems.HSMSP.neh_run
```
Description: Run NEH heuristic to generate expert schedules and state-action pairs for imitation learning

3.Continue training from a checkpoint (optional)
```bash
python main.py --problem HSMSP --resume path/to/checkpoint.pt
```

4.Test only (evaluate a trained model)
```bash
python main.py --problem HSMSP --eval_only --load_path path/to/model.pt
```

## Contact Information
If there are any questions about the codes and datasets, please don't hesitate to contact us. Thanks!
