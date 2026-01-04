import os
import time
from tqdm import tqdm
import torch
import math
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, RandomSampler
from torch.nn import DataParallel

from utils.log_utils import log_values, log_values_sl
from utils.data_utils import BatchedRandomSampler
from utils import move_to
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import pickle


def get_inner_model(model):  # Define get_inner_model function
    return model.module if isinstance(model, DataParallel) else model  # If model is DataParallel instance, return its inner model; otherwise return the model itself


def set_decode_type(model, decode_type):  # Define set_decode_type function
    if isinstance(model, DataParallel):  # If model is DataParallel instance
        model = model.module  # Get its inner model
    model.set_decode_type(decode_type)  # Set decode type


def move_to(obj, device):  # Define move_to function
    if torch.is_tensor(obj):  # If object is a tensor
        return obj.to(device)  # Move it to specified device
    elif isinstance(obj, dict):  # If object is a dictionary
        return {k: move_to(v, device) for k, v in obj.items()}  # Recursively move each value in dictionary to specified device
    elif isinstance(obj, list):  # If object is a list
        return [move_to(v, device) for v in obj]  # Recursively move each element in list to specified device
    else:  # If object is other type
        return obj  # Return the object itself


# Define collate_fn function, used to collate multiple samples into a batch
def collate_fn(batch):
    nodes = [torch.tensor(item['nodes'], dtype=torch.float) for item in batch]
    graph = [torch.tensor(item['graph'], dtype=torch.float) for item in batch]
    tour_nodes = [torch.tensor(item['tour_nodes'], dtype=torch.long) for item in batch]
    tour_edges = [torch.tensor(item['tour_edges'], dtype=torch.long) for item in batch]

    max_graph_dim = max(g.size(0) for g in graph)
    max_node_dim = max(n.size(0) for n in nodes)
    max_edge_dim = max(e.size(0) for e in tour_edges)

    graph_padded = torch.zeros((len(graph), max_graph_dim, max_graph_dim), dtype=torch.float)
    for i, g in enumerate(graph):
        graph_padded[i, :g.size(0), :g.size(1)] = g

    nodes_padded = torch.zeros((len(nodes), max_node_dim, nodes[0].size(1)), dtype=torch.float)
    for i, n in enumerate(nodes):
        nodes_padded[i, :n.size(0), :] = n

    tour_nodes_padded = pad_sequence(tour_nodes, batch_first=True, padding_value=-1)

    tour_edges_padded = torch.zeros((len(tour_edges), max_edge_dim, max_edge_dim), dtype=torch.long)
    for i, e in enumerate(tour_edges):
        tour_edges_padded[i, :e.size(0), :e.size(1)] = e

    nodes_mask = torch.tensor(
        [[1] * len(item['nodes']) + [0] * (nodes_padded.size(1) - len(item['nodes'])) for item in batch],
        dtype=torch.bool)
    graph_mask = torch.tensor(
        [[1] * len(item['graph']) + [0] * (graph_padded.size(1) - len(item['graph'])) for item in batch],
        dtype=torch.bool)

    return {
        'nodes': nodes_padded,
        'graph': graph_padded,
        'tour_nodes': tour_nodes_padded,
        'tour_edges': tour_edges_padded,
        'nodes_mask': nodes_mask,
        'graph_mask': graph_mask,
    }


def validate(model, dataset, problem, opts):

    gt_cost = rollout_groundtruth(problem, dataset, opts)  # Given standard scheduling
    cost = rollout_model(model, dataset, opts)[0]
    pi = rollout_model(model, dataset, opts)[1]

    opt_gap = ((cost / gt_cost - 1) * 100)  # Clarify the optimization logic: gt_cost should be larger than cost, optimization aims for smaller cost

    print('Validation groundtruth cost: {:.3f} +- {:.3f}'.format(
        gt_cost.mean(), torch.std(gt_cost)))
    print('Validation average cost: {:.3f} +- {:.3f}'.format(
        cost.mean(), torch.std(cost)))
    print('Validation optimality gap: {:.3f}% +- {:.3f}'.format(
        opt_gap.mean(), torch.std(opt_gap)))

    return cost.mean(), opt_gap.mean()


def rollout_groundtruth(problem, dataset, opts):  # Define baseline inference function
    batch_size = len(dataset)

    # Use DataLoader to iterate through batch data
    for bat in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=opts.num_workers):
        # # Print bat content
        # print("Batch content:", bat)  # Print entire batch
        # print("Nodes:", bat['nodes'])  # Print 'nodes' part
        # print("Tour nodes:", bat['tour_nodes'])  # Print 'tour_nodes' part
        return torch.cat([
            problem.get_costs(bat['nodes'], bat['tour_nodes'])[0]
        ], 0)  # 0 means stacking along row direction, 1 means stacking along column direction


def rollout_model(model, dataset, opts):
    # Define model inference function
    set_decode_type(model, "greedy")  # Set decode type to greedy algorithm

    model.eval()  # Set model to evaluation mode
    all_costs = []  # Used to store all costs
    all_pi, all_loss, all_margin = [], [], []     # Used to store all pi, loss, margin

    def eval_model_bat(bat):  # Define evaluate model batch function
        class_weights = None  # Otherwise class weights are empty
        targets = move_to(bat['tour_nodes'], opts.device)  # Move target nodes to specified device
        x = move_to(bat['nodes'], opts.device)  # Move nodes to specified device
        graph = move_to(bat['graph'], opts.device)  # Move graph to specified device
        with torch.no_grad():  # Turn off gradient calculation
            # Execute model inference and return cost, sequence data
            cost, loss, pi, expected_margin = model(x, graph, targets, class_weights, supervised=True)
        return cost.data.cpu(), loss.cpu(), pi.cpu(), expected_margin.cpu()

    for bat in tqdm(DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers),
                    disable=opts.no_progress_bar, ascii=True):
        cost, loss, pi, expected_margin = eval_model_bat(bat)
        print('Test loss:', loss)
        all_costs.append(cost)  # Collect cost
        all_pi.append(pi)       # Collect pi
        all_margin.append(expected_margin)  # Collect expected_margin

        # Prepare to save data to CSV file
        batch_results = {
            'cost': cost.tolist(),  # Convert cost to list
            'loss': loss.tolist(),  # Convert loss to list
            'pi': [p.tolist() for p in pi],  # Convert each element of pi to list
            'expected_margin': expected_margin.tolist(),  # Convert expected_margin to list
            'nodes': [node.tolist() for node in bat['nodes']]  # Adapt to multi-dimensional node data
        }

        # Convert dictionary to DataFrame
        batch_df = pd.DataFrame(batch_results)

        # Append data to CSV file
        header = not os.path.exists('test_1129_results.csv')  # Add header if file doesn't exist
        batch_df.to_csv('test_1129_results.csv', mode='a', index=False, header=header)

    # Concatenate all costs and sequence data respectively cost, loss, pi, expected_margin
    all_costs = torch.cat(all_costs, 0)
    all_pi = torch.cat(all_pi, 0)
    all_margin = torch.cat(all_margin, 0)

    return all_costs, all_pi, all_margin

def clip_grad_norms(param_groups, max_norm=math.inf):  # Define clip gradient norms function
    grad_norms = [  # Calculate gradient norm for each parameter group and clip
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms  # Clip gradient norm if max_norm is set
    return grad_norms, grad_norms_clipped  # Return gradient norms and clipped gradient norms


def train_epoch_sl(model, optimizer, lr_scheduler, epoch, train_dataset, val_datasets, problem, tb_logger, opts, f):
    # Print information about current epoch start, including learning rate and run name
    print("\nStart train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    start_time = time.time()  # Record training start time

    # If TensorBoard logging is enabled, record learning rate
    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], epoch)  # Record learning rate

    # Create training data loader with specified batch size and random sampler
    train_dataloader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                  sampler=BatchedRandomSampler(train_dataset, opts.batch_size))

    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear optimizer gradients
    set_decode_type(model, "greedy")  # Set model decode type to greedy algorithm

    best_gap = np.inf  # Initialize optimality gap to infinity

    # Create an empty DataFrame to store results from all epochs
    columns = ['epoch', 'batch_id', 'cost', 'loss', 'pi']
    all_results = pd.DataFrame(columns=columns)

    # Use tqdm progress bar to iterate through training dataset
    for batch_id, batch in enumerate(tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=False)):
        # Train one batch
        epoch, batch_id, cost, loss, pi, expected_margin = train_batch_sl(model, optimizer, epoch, batch_id, batch, tb_logger, opts)

        # Get training process data
        batch_results = {
            'epoch': epoch,
            'batch_id': batch_id,
            'cost': cost.tolist(),  # Convert cost to scalar
            'loss': loss.item(),  # Convert loss to scalar
            'pi': pi.tolist(),  # Convert pi to list format
            'expected_margin': expected_margin.tolist()
        }

        # Convert dictionary to DataFrame and append to all_results
        all_results = all_results.append(batch_results, ignore_index=True)

    # Write data to CSV file after each epoch, using 'a' mode to append data
    csv_file_path = 'training_1129_results.csv'  # CSV file path
    all_results.to_csv(csv_file_path, mode='a', index=False, header=not os.path.exists(csv_file_path))

    lr_scheduler.step(epoch)  # Update learning rate scheduler
    epoch_duration = time.time() - start_time  # Calculate epoch duration
    # Print epoch completion information and time taken
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # Perform model validation and get average reward and optimality gap
    avg_reward, avg_opt_gap = validate(model, val_datasets, problem, opts)
    # Write validation results to file
    f.write(f"{epoch},{avg_reward},{avg_opt_gap}\n")

    # If current optimality gap is less than best gap, update best gap and save model
    if best_gap > avg_opt_gap:
        best_gap = avg_opt_gap  # Update best gap
        print('Saving model and state...')  # Print model saving information
        # Save model and its state, including optimizer state and random number generator state
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),  # Model parameters
                'optimizer': optimizer.state_dict(),  # Optimizer state
                'rng_state': torch.get_rng_state(),  # CPU random number generator state
                'cuda_rng_state': torch.cuda.get_rng_state_all()  # CUDA random number generator state
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))  # Save path
        )


def train_batch_sl(model, optimizer, epoch, batch_id, batch, tb_logger, opts):
    # Move batch node data to specified device (e.g., GPU)
    x = move_to(batch['nodes'], opts.device)
    # Move batch graph data to specified device
    graph = move_to(batch['graph'], opts.device)

    # If model type is NAR, need to compute class weights
    if opts.model == 'nar':
        targets = move_to(batch['tour_edges'], opts.device)  # Move target edges to specified device
        _targets = batch['tour_edges'].numpy().flatten()  # Flatten target edges into NumPy array
        # Compute class weights to balance class distribution
        class_weights = compute_class_weight("balanced", classes=np.unique(_targets), y=_targets)
        class_weights = move_to(torch.FloatTensor(class_weights), opts.device)  # Convert class weights to tensor and move to device
    else:
        class_weights = None  # If not NAR model, class weights are empty
        targets = move_to(batch['tour_nodes'], opts.device)  # Move target edges to specified device

    # Execute model forward pass, compute cost, loss, and path
    cost, loss, pi, expected_margin = model(x, graph, targets, class_weights, supervised=True)

    loss = loss / opts.accumulation_steps  # Accumulate loss, compute in batches
    loss.sum().backward()  # Backpropagation to compute gradients
    # Clip gradient norms to prevent gradient explosion
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)

    # Update model parameters every accumulation_steps
    optimizer.step()  # Update model parameters
    optimizer.zero_grad()  # Clear optimizer gradients

    # If current batch is multiple of log step, record log
    if batch_id % int(opts.log_step) == 0:
        # Record cost, gradient norms, loss, etc.
        log_values_sl(cost, grad_norms, epoch, batch_id, None, loss, tb_logger, opts)

    return epoch, batch_id, cost, loss, pi, expected_margin