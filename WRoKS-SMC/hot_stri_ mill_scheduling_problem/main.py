#!/usr/bin/env Python
# coding=utf-8
import os
import json
import pprint as pp
import numpy as np

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from train import train_epoch_sl, validate, get_inner_model

from nets.attention_model import AttentionModel
from nets.encoders.MHGCN_encoder import GraphAttentionEncoder

from utils import torch_load_cpu, load_problem
import time
import pickle


def run(opts):  # Define run function with options(opts) as parameter

    pp.pprint(vars(opts))  # Print all running parameters

    torch.manual_seed(opts.seed)  # Set PyTorch random seed
    np.random.seed(opts.seed)  # Set NumPy random seed

    tb_logger = None  # Initialize TensorBoard logger as None
    if not opts.no_tensorboard:  # If TensorBoard is not disabled
        tb_logger = TbLogger(os.path.join(  # Configure TensorBoard logger
            opts.log_dir, "{}".format(opts.problem), opts.run_name))

    os.makedirs(opts.save_dir)  # Create save directory

    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:  # Save parameters to args.json file
        json.dump(vars(opts), f, indent=True)

    # Set device to GPU if available
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")  # Set computation device to GPU or CPU

    # Start loading model from here
    # ####################################################################################################################
    # Specify which problem is being solved
    problem = load_problem(opts.problem)  # Load problem definition
    assert opts.problem == 'HSMSP', "Supervised learning only supports imitation learning"  # Assert problem type is hot strip mill scheduling

    # Load data from data list
    load_data = {}  # Initialize loaded data as empty
    assert opts.load_path is None or opts.resume is None, "Only one of load_path and resume should be provided"  # Ensure load_path and resume are not both provided
    load_path = opts.load_path if opts.load_path is not None else opts.resume  # Determine load path (this is the path to load trained model data)

    if opts.eval_only:  # If only evaluating
        load_path = opts.model  # Load path is the model path
        opts.model = 'attention'  # Model type set to attention
    if load_path is not None:  # If load path is not None
        print('\nLoading data from {}'.format(load_path))  # Print load path
        load_data = torch_load_cpu(load_path)  # Load data from CPU

    # Initialize model
    model_class = {  # Define model type mapping
        'attention': AttentionModel
    }.get(opts.model, None)  # Get model class
    assert model_class is not None, "Unknown model: {}".format(model_class)  # Assert model class is not None
    encoder_class = {  # Define encoder type mapping
        'gat': GraphAttentionEncoder
    }.get(opts.encoder, None)  # Get encoder class
    assert encoder_class is not None, "Unknown encoder: {}".format(encoder_class)  # Assert encoder class is not None

    # Initialize model
    model = model_class(
        problem=problem,
        machine=opts.machine,
        embedding_dim=opts.embedding_dim,
        encoder_class=encoder_class,
        n_encode_layers=opts.n_encode_layers,
        aggregation=opts.aggregation,
        aggregation_graph=opts.aggregation_graph,
        normalization=opts.normalization,
        learn_norm=opts.learn_norm,
        track_norm=opts.track_norm,
        gated=opts.gated,
        n_heads=opts.n_heads,
        tanh_clipping=opts.tanh_clipping,
        mask_inner=True,
        mask_logits=True,
        mask_graph=False,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)  # Move model to computation device

    # Calculate network parameters
    nb_param = 0  # Initialize parameter count
    for param in model.parameters():  # Iterate through model parameters
        nb_param += np.prod(list(param.data.size()))  # Calculate parameter count

    # Override model parameters with parameters to load
    model_ = get_inner_model(model)  # Get inner model object
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})  # Load model state dictionary

    # Initialize optimizer
    optimizer = optim.Adam([{'params': model.parameters(), 'lr': opts.lr_model}])  # Initialize Adam optimizer

    # Load optimizer state
    if 'optimizer' in load_data:  # If loaded data contains optimizer state
        optimizer.load_state_dict(load_data['optimizer'])  # Load optimizer state
        for state in optimizer.state.values():  # Iterate through optimizer states
            for k, v in state.items():  # Iterate through state dictionary items
                if torch.is_tensor(v):  # If value is a tensor
                    state[k] = v.to(opts.device)  # Move tensor to computation device

    # Initialize learning rate scheduler, decays once per epoch via lr_decay
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda
        epoch: opts.lr_decay ** epoch)  # Initialize learning rate scheduler

    # # ## Load/generate validation data
    val_datasets = problem.make_train_dataset(filename=r'G:\data_boshi_paper\data_boshi_paper\WRoKS-SMC\data\one_day_test.xlsx',
            batch_size=opts.batch_size, neighbors=opts.neighbors, knn_strat=opts.knn_strat, supervised=True, nar=True)

    # Save validation set computation results
    with open(r'G:\data_boshi_paper\data_boshi_paper\WRoKS-SMC\data\val_datasets.pkl', 'wb') as f:
        pickle.dump(val_datasets, f)

    # Load validation set data
    with open(r'G:\data_boshi_paper\data_boshi_paper\WRoKS-SMC\data\val_datasets.pkl', 'rb') as f:
        val_datasets = pickle.load(f)

    ####################################################################################################################
    train_dataset = problem.make_train_dataset(filename=r'G:\data_boshi_paper\data_boshi_paper\data\A_small_big\two_mo_train.xlsx',
        batch_size=opts.batch_size, neighbors=opts.neighbors, knn_strat=opts.knn_strat, supervised=True, nar=True)

    # Save training set computation results
    with open(r'G:\data_boshi_paper\data_boshi_paper\WRoKS-SMC\data\train_datasets.pkl', 'wb') as f:

        pickle.dump(train_dataset, f)
    # Load training set data
    with open(r'G:\data_boshi_paper\data_boshi_paper\WRoKS-SMC\data\train_datasets.pkl', 'rb') as f:
        train_dataset = pickle.load(f)

    print('Number of batches per epoch during training:', train_dataset.size)
    start_time = time.time()  # Record start time
    opts.epoch_size = train_dataset.size  # Update epoch_size to training set size, 38

    # Start training
    log_path = os.path.join(opts.save_dir, 'alog.txt')  # Log file path
    f = open(log_path, 'w')  # Open log file
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):  # Training loop
        start = time.time()  # Record start time
        train_epoch_sl(  # Train one epoch
            model,
            optimizer,
            lr_scheduler,
            epoch,
            train_dataset,
            val_datasets,
            problem,
            tb_logger,
            opts, f
        )
        end = time.time()  # Record end time
        print(f"Training time cost {end - start}")  # Print training time cost
    f.close()  # Close log file

    end_time = time.time()  # Record end time
    total_time = end_time - start_time  # Calculate total running time
    print(f"Total running time: {total_time:.2f} seconds")  # Print total running time

if __name__ == "__main__":
    run(get_options())