import datetime
#!/usr/bin/env Python
# coding=utf-8
from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from problems.HSMSP.state_hsmsp import StateHSMSP
from utils.beam_search import beam_search
from problems.HSMSP.neh_run import neh_Rollingunit, neh_text, compute_robust_objective

import math
import random
import copy
import time
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from itertools import groupby

import logging
import colorlog
from problems.HSMSP.penalty_func import get_p_hard, get_p_thick, get_p_width
import torch.nn.utils.rnn as rnn_utils

# Set up logging color configuration
log_colors_config = {
    'DEBUG': 'cyan',  # Debug information uses cyan
    'INFO': 'green',  # General information uses green
    'WARNING': 'yellow',  # Warning information uses yellow
    'ERROR': 'red',  # Error information uses red
    'CRITICAL': 'bold_red',  # Critical error information uses bold red
}

# Create logger and set log level to DEBUG
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

# Create a stream handler (output logs to console) and set log level to INFO
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)

# Create a file handler (output logs to file) and set log level to DEBUG
fh = logging.FileHandler(filename='run_file.log', mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)

# Set log format for stream handler and apply colors
stream_fmt = colorlog.ColoredFormatter(
    fmt="%(log_color)s[%(asctime)s] - %(filename)-8s - %(levelname)-7s - line %(lineno)s - %(message)s",
    log_colors=log_colors_config)
file_fmt = logging.Formatter(
    fmt="[%(asctime)s] - %(name)s - %(levelname)-5s - %(filename)-8s : line %(lineno)s - %(funcName)s - %(message)s"
    , datefmt="%Y/%m/%d %H:%M:%S")

# Apply format to handlers
sh.setFormatter(stream_fmt)
fh.setFormatter(file_fmt)

# Add handlers to logger
logger.addHandler(sh)
logger.addHandler(fh)

# Close handlers
sh.close()
fh.close()

np.random.seed(1234)


def get_c_mat(width, hard, thick) -> np.ndarray:
    """ Get cost matrix between slabs """
    num_of_slab = len(width)
    c_mat = np.zeros((num_of_slab, num_of_slab), order='C')

    for i in range(num_of_slab):
        for j in range(num_of_slab):
            c_mat[i, j] = get_p_width(width[i] - width[j]) + \
                          get_p_hard(hard[i] - hard[j]) + \
                          get_p_thick(thick[i] - thick[j])
    return c_mat


def nearest_neighbor_graph(nodes, neighbors):  # Needs modification
    """Return k-nearest neighbor graph as a **NEGATIVE** adjacency matrix"""
    ####################################################################################################
    num_nodes = len(nodes)

    neighbors = min(neighbors, num_nodes - 1)
    #     # Calculate distance matrix
    #     #[[0, 1550, 5.95, 5, 20000, 90.0], [1, 1550, 6.29, 5, 20000, 90.0], [2, 1550, 6.54, 5, 20000, 90.0],
    #     # [3, 1550, 6.29, 5, 22000, 99.0], [4, 1550, 5.95, 5, 28000, 126.0], [5, 1550, 6.54, 5, 28000, 126.0],
    #     # [6, 1550, 5.95, 5, 29000, 130.5], [10, 1530, 6.37, 4, 20000, 90.0], [11, 1530, 6.54, 4, 21000, 94.5],
    #     # [12, 1530, 5.95, 5, 22000, 99.0], [13, 1530, 6.37, 4, 25000, 112.5], [14, 1530, 5.95, 5, 25000, 112.5],
    #     # [15, 1530, 5.95, 5, 25000, 112.5], [16, 1530, 6.54, 4, 35000, 157.5], [20, 1500, 6.37, 4, 21000, 94.5],
    #     # [21, 1500, 6.37, 4, 21000, 94.5], [22, 1500, 6.37, 4, 21000, 94.5], [23, 1500, 6.54, 5, 21000, 94.5], [24, 1500, 6.37, 5, 22000, 99.0], [25, 1500, 6.37, 4, 25000, 112.5], [26, 1500, 6.37, 4, 25000, 112.5], [27, 1500, 6.37, 5, 25000, 112.5], [40, 1280, 6.37, 4, 21000, 94.5], [41, 1280, 6.37, 4, 35000, 157.5], [42, 1280, 6.37, 4, 35000, 157.5], [43, 1280, 5.95, 4, 36000, 162.0], [44, 1280, 6.37, 4, 36000, 162.0], [45, 1250, 6.29, 3, 20000, 90.0], [46, 1250, 5.95, 3, 20000, 90.0],
    #     # [47, 1250, 5.95, 3, 22000, 99.0], [48, 1250, 5.95, 3, 28000, 126.0], [54, 1200, 5.49, 3, 22000, 99.0]]
    #     # W_val = squareform(pdist(nodes, metric='euclidean'))
    #
    # Calculate penalty cost matrix
    width = [sublist[0] for sublist in nodes]
    hard = [sublist[2] for sublist in nodes]
    thick = [sublist[1] for sublist in nodes]
    # [[0.   7.   7.... 105. 105. 179.], [14.   0.   3.... 119. 119. 189.], [14.   6.   0.... 119. 119. 201.], ...,
    #  [inf  inf  inf...   0.   0.  16.], [inf  inf  inf...   0.   0.  16.], [inf  inf  inf...inf  inf   0.]]
    # This distance matrix needs to be recalculated
    W_val = get_c_mat(width, hard, thick)

    W = np.ones((num_nodes, num_nodes))

    # Determine k nearest neighbors for each node
    knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]

    # Define directed graph for graph structure
    for idx in range(num_nodes):
        for nn in knns[idx]:
            if width[idx] > width[nn]:  # Only connect in width decreasing direction
                W[idx][nn] = 0

    for idx in range(num_nodes):
        W[idx][knns[idx]] = 0

    # Ensure no self-connection in slab connections
    np.fill_diagonal(W, 1)
    return W

# The output data is an adjacency matrix where each element indicates whether two nodes in the path are connected (1 means connected, 0 means not connected).
# For example, input tour_nodes = [0, 2, 3, 1], representing visiting nodes in order 0 -> 2 -> 3 -> 1, then returning to node 0.

# Corresponding adjacency matrix:
# tour_nodes = [0, 2, 3, 1]
# Output adjacency matrix tour_edges:
# [[0. 1. 0. 1.]   # Node 0 connected to nodes 1 and 2
#  [1. 0. 1. 0.]   # Node 1 connected to nodes 0 and 3
#  [1. 0. 0. 1.]   # Node 2 connected to nodes 0 and 3
#  [0. 1. 1. 0.]]  # Node 3 connected to nodes 1 and 2
def tour_nodes_to_W(tour_nodes):    # Define a function to convert a path of nodes into an adjacency matrix
    """Compute tour edge adjacency matrix representation"""
    num_nodes = len(tour_nodes)    # Get number of nodes in the path
    tour_edges = np.zeros((num_nodes, num_nodes))  # Initialize a num_nodes x num_nodes all-zero adjacency matrix to represent connectivity between nodes
    for idx in range(len(tour_nodes) - 1):    # Traverse all adjacent node pairs in the path
        i = tour_nodes[idx]                   # Current node i
        j = tour_nodes[idx + 1]                # Next node j
        tour_edges[i][j] = 1                         # Set edge from node i to node j, set corresponding position in adjacency matrix to 1
        tour_edges[j][i] = 1                         # Set edge from node j to node i (symmetric matrix), set corresponding position in adjacency matrix to 1
    # Add final connection
    tour_edges[j][tour_nodes[0]] = 1                 # Connect last node back to first node to form a cycle
    tour_edges[tour_nodes[0]][j] = 1                  # Also connect reverse edge of cycle, maintain symmetry
    return tour_edges                                  # Return computed adjacency matrix

# Read data from file
def read_in(path):
    rename_dict = {
        '序号': 'id',
        '钢卷宽度': 'width',
        '原料宽度': 'ingredients_width',
        '目标厚度': 'thick',
        '硬度': 'hard',
        '钢卷重量': 'weight',
        '生产时间': 'time',
        '轧制长度': 'length',
        '轧机出口钢种': 'roll_grade',
        '冶炼钢种': 'smelt_grade',
        '发货时间': 'deliver_time'
    }

    df = pd.read_excel(path, engine='openpyxl')  # Read data from excel
    df.rename(columns=rename_dict, inplace=True)

    return df


class TSP(object):
    # Class representing traveling salesman problem, this problem needs to be modified to solve the first stage of optimization problem
    NAME = 'tsp'  # Class name attribute

    @staticmethod
    def get_costs(dataset, pi):
        """
        Return TSP tour length and custom cost for given graph nodes and tour permutation
        Args:
            dataset: Graph node data (torch.Tensor), shape (batch_size, num_nodes, num_features)
            pi: Node permutation representation (torch.Tensor), shape (batch_size, num_nodes)

        Returns:
            Return TSP tour length, and calculated custom makespan
        """
        # Check if pi contains all nodes
        assert (torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) == pi.data.sort(1)[
            0]).all(), "Invalid tour:\n{}\n{}".format(dataset, pi)
        # Ensure tensors are on CPU
        data_cpu = dataset.cpu()
        pi_cpu = pi.cpu()

        # Convert tensors to lists
        data_list = data_cpu.tolist()
        pi_list = pi_cpu.tolist()

        makespan, time_margin = [], []

        global alpha, beta, w
        w = 0.9
        alpha = 0.2  # Initial value
        beta = 100  # Given initial value for negative time margin

        # Traverse each batch
        for i in range(len(data_list)):  # Traverse first dimension (batch_size)
            # Directly use model output pi sequence corresponding sorted data
            total_penalty, f1, f2, expected_time = compute_robust_objective(data_list[i], pi_list[i], w,
                                                                            alpha, beta)

            makespan.append(total_penalty)  # Add cost of each batch to makespan list
            time_margin.append(expected_time)

        # Convert list to tensor
        cost = torch.tensor(makespan)
        expected_margin = torch.tensor(time_margin)

        return cost, expected_margin, None

    @staticmethod
    def make_train_dataset(*args, **kwargs):
        return HSMSPTrainDataset(*args, **kwargs)  # Call TSPDataset class to create dataset

    @staticmethod
    def make_text_dataset(*args, **kwargs):
        return HSMSPTextDataset(*args, **kwargs)  # Call TSPDataset class to create dataset

    @staticmethod
    def make_state(*args, **kwargs):
        return StateHSMSP.initialize(*args, **kwargs)  # Call StateTSP class initialize method to create state

    @staticmethod
    def beam_search(nodes, graph, beam_size, expand_size=None,  # Define beam_search static method
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        # Given TSP sample and model, call beam search method

        assert model is not None, "Provide model"  # Assert model is not None

        fixed = model.precompute_fixed(nodes, graph)  # Precompute fixed nodes and graph

        def propose_expansions(beam):
            return model.propose_expansions(  # Call model's propose_expansions method
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(  # Create initial state
            nodes, graph, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)  # Return beam search result

class HSMSP(TSP):
    """Class representing the Travelling Salesman Problem, trained with Supervised Learning
    """
    NAME = 'HSMSP'


class HSMSPTrainDataset(Dataset):  # Define HSMSPDataset class, inheriting from Dataset class

    def __init__(self, filename=None,  batch_size=128, offset=0, distribution=None, neighbors=20,
                 knn_strat=None, supervised=True, nar=True):  # Continue initialization method parameter list
        """Class representing PyTorch dataset of HSMSP instances that are fed to data loaders
        Args:
            filename: File path, used when reading from file (for supervised learning).  # File path parameter
            batch_size: Batch size for data loading and batch processing.  # Batch processing size parameter
            offset: Offset when loading data from file.  # Data offset parameter
            distribution: Data generation distribution (not used).  # Data distribution parameter
            neighbors: Number of neighbors when computing k-nearest neighbor graph.  # Number of neighbors parameter
            knn_strat: Strategy for computing k-nearest neighbor graph ('percentage' or 'standard').  # k-nearest neighbor strategy parameter
            supervised: Flag to enable supervised learning.  # Supervised learning flag parameter
            nar: Flag indicating use of non-autoregressive decoding scheme that uses edge-level ground truth.  # Non-autoregressive decoding flag parameter
        Notes:
            batch_size is an important parameter fixed in dataset and data loader because we deal with variable-sized HSMSP graphs.
            To achieve efficient training without DGL/PyG-style sparse graph libraries, we ensure each batch contains dense graphs of same size.  # Note explanation
        """
        super(HSMSPTrainDataset, self).__init__()  # Call parent class initialization method

        self.filename = filename
        self.batch_size = batch_size  # Initialize batch_size attribute

        # self.num_of_slab = num_of_slab  # Initialize num_of_slab attribute
        self.offset = offset  # Initialize offset attribute

        self.distribution = distribution  # Initialize distribution attribute

        self.neighbors = neighbors  # Initialize neighbors attribute
        self.knn_strat = knn_strat  # Initialize knn_strat attribute
        self.supervised = supervised  # Initialize supervised attribute
        self.nar = nar  # Initialize nar attribute

        data = read_in(self.filename)  # Initialize filename attribute

        # Get data parameters from data table
        num_of_slab = len(data['id'])
        id = list(data['id'])
        width = list(data['width'])

        ingredients_width = list(data['ingredients_width'])

        thick = list(data['thick'])

        time = list(data['time'])

        hard = list(data['hard'])
        weight = list(data['weight'])
        length = list(data['length'])

        roll_grade = list(data['roll_grade'])

        smelt_grade = list(data['smelt_grade'])

        deliver_time = list(data['deliver_time'])

        self.nodes_coords, self.tour_nodes, self.time_margin, self.cost = [], [], [], []  # Initialize node coordinate list main rolling material
        nodes_coords_tg, nodes_coords_zz = [], []  # , warm-up material node coordinate list

        # Calculate based on time parameters for front and back slabs
        # Convert list to Pandas Series
        time_series = pd.Series(time)

        # Calculate difference between adjacent timestamps
        time_diffs = time_series.diff()

        # Convert difference to seconds
        time_diffs_in_seconds = time_diffs.dt.total_seconds()

        # Count number of time differences greater than 900 seconds (15 minutes)
        batch_Rollingunit = (time_diffs_in_seconds > 900).sum()  # Number of rolling units

        # Remove NaN values
        time_diffs_in_seconds = time_diffs_in_seconds.dropna()

        # Convert to list
        time_diffs_list = time_diffs_in_seconds.tolist()

        # Add value same as previous difference
        time_diffs_list.append(time_diffs_list[-1])

        # Read data from file, slab yard entry time and furnace residence time,
        with open(r'G:\data_boshi_paper\data_boshi_paper\WRoKS-SMC\data\time_diffs_data.pkl','rb') as f:
            # Read three data objects
            clustered_data, cut_furnace_results = pickle.load(f)

        # Define yard entry time min, yard entry time max, yard entry time average: furnace residence min, max, average, rolling min time, max time, average time, total 9 dimensions of data to add
        slabyard_min, slabyard_max, slabyard_rel, furnace_min, furnace_max, furnace_rel, roll_min, roll_max, roll_rel = [], [], [], [], [], [], [], [], []

        # Calculate yard entry time min, yard entry time max, yard entry time average: furnace residence min, max, average
        for i in range(len(id)):

            for j, temp_data in enumerate(clustered_data):
                if smelt_grade[i] == temp_data[4] and ingredients_width[i] == temp_data[1]:
                    # print(f"Matching data for i={i}, j={j}")

                    temp_slabyard_min = cut_furnace_results[j]['cut_to_furnace_min']
                    temp_slabyard_max = cut_furnace_results[j]['cut_to_furnace_max']
                    temp_slabyard_avg = cut_furnace_results[j]['cut_to_furnace_avg']
                    temp_furnace_min = cut_furnace_results[j]['furnace_to_out_min']
                    temp_furnace_max = cut_furnace_results[j]['furnace_to_out_max']
                    temp_furnace_avg = cut_furnace_results[j]['furnace_to_out_avg']

                    slabyard_min.append(temp_slabyard_min)
                    slabyard_max.append(temp_slabyard_max)
                    slabyard_rel.append(temp_slabyard_avg)
                    furnace_min.append(temp_furnace_min)
                    furnace_max.append(temp_furnace_max)
                    furnace_rel.append(temp_furnace_avg)
                # If no match then assign
                else:
                    slabyard_min.append(0)
                    slabyard_max.append(86400)
                    slabyard_rel.append(11880)
                    furnace_min.append(9000)
                    furnace_max.append(18000)
                    furnace_rel.append(11880)

        # Initialize nested list
        grouped_timestamps = []
        current_timegroup = [time[0]]  # Start with the first timestamp

        # Traverse timestamps and differences, group timestamps with difference greater than 15 minutes  and self.width[i+1] - self.width[i] <= 300
        for i in range(1, len(time_diffs_in_seconds)):
            if time_diffs_in_seconds[i] > 540:
                grouped_timestamps.append(current_timegroup)
                current_timegroup = [time[i]]  # Start new cluster
            else:
                current_timegroup.append(time[i])

        # Add last group if exists
        if current_timegroup:
            grouped_timestamps.append(current_timegroup)

        # Data format timestamp grouping:
        # ['2022-01-01 00:01:11', '2022-01-01 00:02:41', '2022-01-01 00:04:15']
        # ['2022-01-01 00:20:43', '2022-01-01 00:21:43', '2022-01-01 00:22:43']
        # ['2022-01-01 00:40:00', '2022-01-01 00:42:00']
        # Print timestamp grouping
        # print("Timestamp grouping:")

        for group in grouped_timestamps:
            count = 0
            batch_tour_nodes, batch_tour_nodes_tg, batch_tour_nodes_zz, batch_tour_id = [], [], [], []
            # Traverse each group's timestamps
            for timestamp in group:

                # Find data matching timestamp
                for k in range(num_of_slab):
                    if timestamp == time[k]:
                        node_data = [
                            width[k],  # Width
                            thick[k],  # Thickness
                            hard[k],  # Hardness
                            length[k],  # Length
                            slabyard_min[k],  # Yard entry time min
                            slabyard_max[k],  # Yard entry time max
                            slabyard_rel[k],  # Yard entry time average
                            furnace_min[k],  # Furnace residence time min
                            furnace_max[k],  # Furnace residence time max
                            furnace_rel[k],  # Furnace residence time average
                            time_diffs_list[k]  # Time difference
                        ]

                        count += 1
                        batch_tour_nodes.append(node_data)
                        batch_tour_id.append(count+1)

            # First 6 as warm-up material, store in batch_tour_nodes_zz list
            batch_tour_nodes_tg = batch_tour_nodes[:6]

            # Sort batch_tour_nodes by self.width[k] (i.e., node_data[1]) from small to large
            batch_tour_nodes_tg.sort(key=lambda x: x[0], reverse=False)

            # Remaining as main rolling, store in batch_tour_nodes_tg list
            batch_tour_nodes_zz = batch_tour_nodes[6:]

            batch_tour_nodes_zz.sort(key=lambda x: x[0], reverse=True)

            # Add entire batch of nodes to nodes_coords

            if len(batch_tour_nodes) >= 25:
                nodes_coords_zz.append(batch_tour_nodes_zz)
                nodes_coords_tg.append(batch_tour_nodes_tg)

        # Need to preprocess data, expand slabs of different groups
        # Step 1: Calculate maximum nodes per group
        max_nodes_per_group = max(len(group) for group in nodes_coords_zz)

        # Step 2: Construct new NumPy array, initialized to 0, shape (number of groups, max_nodes_per_group, 11)
        num_groups = len(nodes_coords_zz)
        num_attributes = len(nodes_coords_zz[0][0])  # Calculate dimension of each node data, representing slab mathematics
        self.nodes_coords = np.zeros((num_groups, max_nodes_per_group, num_attributes))

        # Step 3: Fill data into NumPy array, using existing data, fill missing parts
        for i, group in enumerate(nodes_coords_zz):
            for j, node in enumerate(group):
                self.nodes_coords[i, j, :len(node)] = node  # Fill original data, number of slabs

        # Expert-driven heuristic algorithm
        for i in tqdm(range(len(self.nodes_coords))):
            batch_data = self.nodes_coords[i]
            ms, seq, time_margin_min = neh_Rollingunit(batch_data)
            self.tour_nodes.append(seq)
            self.time_margin.append(time_margin_min)
            self.cost.append(ms)

        self.size = len(self.nodes_coords)  # Set dataset size

    def __len__(self):  # Define __len__ method
        return self.size  # Return dataset size

    def __getitem__(self, idx):  # Define __getitem__ method

        nodes = self.nodes_coords[idx]  # Get node coordinates corresponding to index  list[32*6]

        item = {  # Create dictionary
            'nodes': torch.FloatTensor(nodes),  # Convert node coordinates to FloatTensor
            'graph': torch.ByteTensor(nearest_neighbor_graph(nodes, self.neighbors))  # Compute and convert nearest neighbor graph
        }
        if self.supervised:  # If supervised learning enabled
            # # Add supervised learning ground truth labels
            tour_nodes = self.tour_nodes[idx]  # Get corresponding node indices
            item['tour_nodes'] = torch.LongTensor(tour_nodes)  # Convert tour nodes to LongTensor
            if self.nar:  # If non-autoregressive decoding enabled
                # # Non-autoregressive decoding ground truth is HSMSP in adjacency matrix format
                item['tour_edges'] = torch.LongTensor(tour_nodes_to_W(tour_nodes))  # Convert tour nodes to adjacency matrix and convert to LongTensor

        return item  # Return dictionary


class HSMSPTextDataset(Dataset):  # Define HSMSPDataset class, inheriting from Dataset class

    def __init__(self, filename=None, min_size=25, machine=11, max_size=200, batch_size=128,
                 num_samples=128000, offset=0, distribution=None, neighbors=20,
                 knn_strat=None, supervised=True, nar=True):  # Continue initialization method parameter list
        """Class representing PyTorch dataset of HSMSP instances that are fed to data loaders
        Args:
            filename: File path, used when reading from file (for supervised learning).  # File path parameter
            min_size: Minimum size for generating HSMSP problems (for reinforcement learning).  # Minimum size parameter
            max_size: Maximum size for generating HSMSP problems (for reinforcement learning).  # Maximum size parameter
            batch_size: Batch size for data loading and batch processing.  # Batch processing size parameter
            num_samples: Total number of samples in dataset.  # Total sample count parameter
            offset: Offset when loading data from file.  # Data offset parameter
            distribution: Data generation distribution (not used).  # Data distribution parameter
            neighbors: Number of neighbors when computing k-nearest neighbor graph.  # Number of neighbors parameter
            knn_strat: Strategy for computing k-nearest neighbor graph ('percentage' or 'standard').  # k-nearest neighbor strategy parameter
            supervised: Flag to enable supervised learning.  # Supervised learning flag parameter
            nar: Flag indicating use of non-autoregressive decoding scheme that uses edge-level ground truth.  # Non-autoregressive decoding flag parameter
        Notes:
            batch_size is an important parameter fixed in dataset and data loader because we deal with variable-sized HSMSP graphs.
            To achieve efficient training without DGL/PyG-style sparse graph libraries, we ensure each batch contains dense graphs of same size.  # Note explanation
        """
        super(HSMSPTextDataset, self).__init__()  # Call parent class initialization method

        self.filename = filename
        self.batch_size = batch_size  # Initialize batch_size attribute

        # self.num_of_slab = num_of_slab  # Initialize num_of_slab attribute
        self.offset = offset  # Initialize offset attribute

        self.distribution = distribution  # Initialize distribution attribute

        self.neighbors = neighbors  # Initialize neighbors attribute
        self.knn_strat = knn_strat  # Initialize knn_strat attribute
        self.supervised = supervised  # Initialize supervised attribute
        self.nar = nar  # Initialize nar attribute

        data = read_in(self.filename)  # Initialize filename attribute
        max_batch_len = 800000  # Maximum rolling kilometers for same rolling unit, here variable for each steel grade
        max_same_width_len = 350000  # Maximum rolling kilometers for same width, cannot exceed 35km
        temp = 800  # Starting temperature 800
        temp_dec_rate = 0.99  # Cooling rate 0.99

        # Get data parameters from data table
        num_of_slab = len(data['id'])
        id = list(data['id'])
        width = list(data['width'])

        ingredients_width = list(data['ingredients_width'])

        thick = list(data['thick'])

        time = list(data['time'])

        hard = list(data['hard'])
        weight = list(data['weight'])
        length = list(data['length'])

        roll_grade = list(data['roll_grade'])

        smelt_grade = list(data['smelt_grade'])

        deliver_time = list(data['deliver_time'])

        self.nodes_coords, self.tour_nodes, self.time_margin = [], [], []  # Initialize node coordinate list main rolling material
        nodes_coords_tg, nodes_coords_zz = [], []  # , warm-up material node coordinate list

        # Calculate based on time parameters for front and back slabs
        # Convert list to Pandas Series
        time_series = pd.Series(time)

        # Calculate difference between adjacent timestamps
        time_diffs = time_series.diff()

        # Convert difference to seconds
        time_diffs_in_seconds = time_diffs.dt.total_seconds()

        # Count number of time differences greater than 900 seconds (15 minutes)
        batch_Rollingunit = (time_diffs_in_seconds > 900).sum()  # Number of rolling units

        # Remove NaN values
        time_diffs_in_seconds = time_diffs_in_seconds.dropna()

        # Convert to list
        time_diffs_list = time_diffs_in_seconds.tolist()

        # Add value same as previous difference
        time_diffs_list.append(time_diffs_list[-1])

        # Read data from file, slab yard entry time and furnace residence time,
        with open(r'data\time_diffs_data.pkl','rb') as f:
            # Read three data objects
            clustered_data, cut_furnace_results = pickle.load(f)

        # # Read data from file, slab rolling time
        # with open(r'C:\wyj\博士小论文\one\热轧不确定性鲁棒优化\64强化学习+图神经网络\IL_PFSS_job_plans\learning-pfss\data\rolltime_results.pkl',
        #           'rb') as f:
        #     # Read 2 data objects
        #     self.rollcluster_data, self.rolltime_results = pickle.load(f)

        # Define yard entry time min, yard entry time max, yard entry time average: furnace residence min, max, average, rolling min time, max time, average time, total 9 dimensions of data to add
        slabyard_min, slabyard_max, slabyard_rel, furnace_min, furnace_max, furnace_rel, roll_min, roll_max, roll_rel = [], [], [], [], [], [], [], [], []

        # Calculate yard entry time min, yard entry time max, yard entry time average: furnace residence min, max, average
        for i in range(len(id)):
            temp_slabyard_min, temp_slabyard_max, temp_slabyard_avg = None, None, None
            temp_furnace_min, temp_furnace_max, temp_furnace_avg = None, None, None
            for j, temp_data in enumerate(clustered_data):
                if smelt_grade[i] == temp_data[4] and ingredients_width[i] == temp_data[1]:
                    # print(f"Matching data for i={i}, j={j}")

                    temp_slabyard_min = cut_furnace_results[j]['cut_to_furnace_min']
                    temp_slabyard_max = cut_furnace_results[j]['cut_to_furnace_max']
                    temp_slabyard_avg = cut_furnace_results[j]['cut_to_furnace_avg']
                    temp_furnace_min = cut_furnace_results[j]['furnace_to_out_min']
                    temp_furnace_max = cut_furnace_results[j]['furnace_to_out_max']
                    temp_furnace_avg = cut_furnace_results[j]['furnace_to_out_avg']

                    slabyard_min.append(temp_slabyard_min)
                    slabyard_max.append(temp_slabyard_max)
                    slabyard_rel.append(temp_slabyard_avg)
                    furnace_min.append(temp_furnace_min)
                    furnace_max.append(temp_furnace_max)
                    furnace_rel.append(temp_furnace_avg)
                # If no match then assign
                else:
                    slabyard_min.append(0)
                    slabyard_max.append(86400)
                    slabyard_rel.append(11880)
                    furnace_min.append(9000)
                    furnace_max.append(18000)
                    furnace_rel.append(11880)

        # Initialize nested list
        grouped_timestamps = []
        current_timegroup = [time[0]]  # Start with the first timestamp

        # Traverse timestamps and differences, group timestamps with difference greater than 15 minutes  and self.width[i+1] - self.width[i] <= 300
        for i in range(1, len(time_diffs_in_seconds)):
            if time_diffs_in_seconds[i] > 540:
                grouped_timestamps.append(current_timegroup)
                current_timegroup = [time[i]]  # Start new cluster
            else:
                current_timegroup.append(time[i])

        # Add last group if exists
        if current_timegroup:
            grouped_timestamps.append(current_timegroup)

        # Data format timestamp grouping:
        # ['2022-01-01 00:01:11', '2022-01-01 00:02:41', '2022-01-01 00:04:15']
        # ['2022-01-01 00:20:43', '2022-01-01 00:21:43', '2022-01-01 00:22:43']
        # ['2022-01-01 00:40:00', '2022-01-01 00:42:00']
        # Print timestamp grouping
        # print("Timestamp grouping:")

        for group in grouped_timestamps:
            count = 0
            batch_tour_nodes, batch_tour_nodes_tg, batch_tour_nodes_zz, batch_tour_id = [], [], [], []
            # Traverse each group's timestamps
            for timestamp in group:

                # Find data matching timestamp
                for k in range(num_of_slab):
                    if timestamp == time[k]:
                        node_data = [
                            width[k],  # Width
                            thick[k],  # Thickness
                            hard[k],  # Hardness
                            length[k],  # Length
                            slabyard_min[k],
                            slabyard_max[k],
                            slabyard_rel[k],
                            furnace_min[k],
                            furnace_max[k],
                            furnace_rel[k],
                            time_diffs_list[k]
                        ]
                        count += 1
                        batch_tour_nodes.append(node_data)
                        batch_tour_id.append(count+1)

            # First 6 as warm-up material, store in batch_tour_nodes_zz list
            batch_tour_nodes_tg = batch_tour_nodes[:6]
            # Sort batch_tour_nodes by self.width[k] (i.e., node_data[1]) from small to large
            batch_tour_nodes_tg.sort(key=lambda x: x[0], reverse=False)

            # Remaining as main rolling, store in batch_tour_nodes_tg list
            batch_tour_nodes_zz = batch_tour_nodes[6:]

            batch_tour_nodes_zz.sort(key=lambda x: x[0], reverse=True)

            # Add entire batch of nodes to nodes_coords

            if len(batch_tour_nodes) >= 25:
                nodes_coords_zz.append(batch_tour_nodes_zz)
                nodes_coords_tg.append(batch_tour_nodes_tg)

        # Need to preprocess data, expand slabs of different groups
        # Step 1: Calculate maximum nodes per group
        max_nodes_per_group = max(len(group) for group in nodes_coords_zz)

        # Step 2: Construct new NumPy array, initialized to 0, shape (number of groups, max_nodes_per_group, 11)
        num_groups = len(nodes_coords_zz)
        num_attributes = len(nodes_coords_zz[0][0])  # Calculate dimension of each node data, representing slab mathematics
        self.nodes_coords = np.zeros((num_groups, max_nodes_per_group, num_attributes))

        # Step 3: Fill data into NumPy array, using existing data, fill missing parts
        for i, group in enumerate(nodes_coords_zz):
            for j, node in enumerate(group):
                self.nodes_coords[i, j, :len(node)] = node  # Fill original data, number of slabs

        # # Print output array shape, ensure it meets requirements
        # print("Output array shape:", output_array.shape)
        # print("Output data:\n", output_array)

        # Expert-driven heuristic algorithm
        for i in tqdm(range(len(self.nodes_coords))):
            batch_data = self.nodes_coords[i]
            ms, seq, time_margin_min = neh_text(batch_data)
            self.tour_nodes.append(seq)
            self.time_margin.append(time_margin_min)

        self.size = len(self.nodes_coords)  # Set dataset size

    def __len__(self):  # Define __len__ method
        return self.size  # Return dataset size

    def __getitem__(self, idx):  # Define __getitem__ method

        # Print debug information
        # print(f"Index requested: {idx}")
        # print(f"Length of nodes_coords: {len(self.nodes_coords)}")

        # [[0, 1550, 5.95, 5, 20000, 90.0], [1, 1550, 6.29, 5, 20000, 90.0], [2, 1550, 6.54, 5, 20000, 90.0],
        # [3, 1550, 6.29, 5, 22000, 99.0], [4, 1550, 5.95, 5, 28000, 126.0], [5, 1550, 6.54, 5, 28000, 126.0],
        # [6, 1550, 5.95, 5, 29000, 130.5], [10, 1530, 6.37, 4, 20000, 90.0], [11, 1530, 6.54, 4, 21000, 94.5],
        # [12, 1530, 5.95, 5, 22000, 99.0], [13, 1530, 6.37, 4, 25000, 112.5], [14, 1530, 5.95, 5, 25000, 112.5],
        # [15, 1530, 5.95, 5, 25000, 112.5], [16, 1530, 6.54, 4, 35000, 157.5], [20, 1500, 6.37, 4, 21000, 94.5],
        # [21, 1500, 6.37, 4, 21000, 94.5], [22, 1500, 6.37, 4, 21000, 94.5], [23, 1500, 6.54, 5, 21000, 94.5],
        # [24, 1500, 6.37, 5, 22000, 99.0], [25, 1500, 6.37, 4, 25000, 112.5], [26, 1500, 6.37, 4, 25000, 112.5],
        # [27, 1500, 6.37, 5, 25000, 112.5], [40, 1280, 6.37, 4, 21000, 94.5], [41, 1280, 6.37, 4, 35000, 157.5], [42, 1280, 6.37, 4, 35000, 157.5], [43, 1280, 5.95, 4, 36000, 162.0], [44, 1280, 6.37, 4, 36000, 162.0], [45, 1250, 6.29, 3, 20000, 90.0], [46, 1250, 5.95, 3, 20000, 90.0],
        # [47, 1250, 5.95, 3, 22000, 99.0], [48, 1250, 5.95, 3, 28000, 126.0], [54, 1200, 5.49, 3, 22000, 99.0]]
        nodes = self.nodes_coords[idx]  # Get node coordinates corresponding to index  list[32*6]

        item = {  # Create dictionary
            'nodes': torch.FloatTensor(nodes),  # Convert node coordinates to FloatTensor
            'graph': torch.ByteTensor(nearest_neighbor_graph(nodes, self.neighbors))  # Compute and convert nearest neighbor graph
        }
        if self.supervised:  # If supervised learning enabled
            # # Add supervised learning ground truth labels
            tour_nodes = self.tour_nodes[idx]  # Get node corresponding to index
            item['tour_nodes'] = torch.LongTensor(tour_nodes)  # Convert tour nodes to LongTensor
            if self.nar:  # If non-autoregressive decoding enabled
                # # Non-autoregressive decoding ground truth is HSMSP tour path in adjacency matrix format
                # [[0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27, 40, 41, 42, 43, 44, 45, 46, 47, 48, 54],
                # [7, 8, 9, 17, 18, 19, 28, 29, 30, 31, 32, 33, 49, 50, 51, 52, 53, 55, 56, 57, 58, 59, 60, 62, 63],
                # [34, 35, 36, 37, 38, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
                # [39, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]]
                item['tour_edges'] = torch.LongTensor(tour_nodes_to_W(tour_nodes))  # Convert tour nodes to adjacency matrix and convert to LongTensor

        return item  # Return dictionary