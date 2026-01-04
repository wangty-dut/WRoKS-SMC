import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter
import numpy as np
import os
from problems.HSMSP.penalty_func import get_p_hard, get_p_thick, get_p_width
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class StateHSMSP(NamedTuple):  # HSMSP state class, used to track HSMSP state during beam search

    # Fixed inputs
    loc: torch.Tensor  # Node coordinates
    dist: torch.Tensor  # Distance matrix between nodes

    # If this state contains multiple copies of the same instance (i.e., beam search), for memory efficiency
    # the loc and dist tensors are not saved multiple times, so we need to use ids to index the correct rows
    ids: torch.Tensor  # Index to track original fixed data rows

    # State
    first_a: torch.Tensor  # First visited node
    prev_a: torch.Tensor  # Previously visited node
    visited_: torch.Tensor  # Track visited nodes
    lengths: torch.Tensor  # Total length of visited path
    cur_coord: torch.Tensor  # Current node coordinates
    i: torch.Tensor  # Track step count
    graph: torch.Tensor  # Graph representation

    # @property is a built-in decorator used to convert a class method into a read-only property.
    # This means you can call this method like an attribute without explicitly calling it
    @property
    def visited(self):  # Visited nodes property
        if self.visited_.dtype == torch.uint8:  # If data type is uint8
            return self.visited_  # Return original visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))  # Convert long integer visited_ to boolean type

    def __getitem__(self, key):  # Method to get item
        if torch.is_tensor(key) or isinstance(key, slice):  # If key is tensor or slice
            return self._replace(
                ids=self.ids[key],  # Index ids by key
                first_a=self.first_a[key],  # Index first_a by key
                prev_a=self.prev_a[key],  # Index prev_a by key
                visited_=self.visited_[key],  # Index visited_ by key
                lengths=self.lengths[key],  # Index lengths by key
                cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,  # Index cur_coord by key
            )
        return super(StateHSMSP, self).__getitem__(key)  # Otherwise call parent's __getitem__ method

    @staticmethod
    def get_c_mat(loc) -> list:
        """ Get cost matrix between slabs """
        widths = loc[:, :, 0].tolist()
        thicknesses = loc[:, :, 1].tolist()
        hardnesses = loc[:, :, 2].tolist()

        penalty_matrix = []
        for unit, rolling_unit in enumerate(loc):
            width = widths[unit]
            thick = thicknesses[unit]
            hard = hardnesses[unit]
            num_of_slab = sum(1 for i in range(len(rolling_unit)) if not (
                    width[i] == 0 and thick[i] == 0 and hard[i] == 0))  # Only consider number of valid nodes
            c_mat = np.full((len(rolling_unit), len(rolling_unit)), np.inf)  # Initialize to infinity
            for i in range(len(rolling_unit)):
                if width[i] == 0 and thick[i] == 0 and hard[i] == 0:  # Mask invalid points
                    continue
                for j in range(len(rolling_unit)):
                    if width[j] == 0 and thick[j] == 0 and hard[j] == 0:  # Mask invalid points
                        continue
                    c_mat[i, j] = get_p_width(width[i] - width[j]) + \
                                  get_p_hard(hard[i] - hard[j]) + \
                                  get_p_thick(thick[i] - thick[j])
            penalty_matrix.append(c_mat)

        return penalty_matrix

    @staticmethod
    # Define initialize function, input parameters include loc, graph, nodes_mask and visited_dtype
    def initialize(loc, graph, visited_dtype=torch.uint8):
        batch_size, n_loc, _ = loc.size()  # Get loc dimension info: batch_size represents batch size, n_loc represents node count
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long,
                             device=loc.device)  # Initialize prev_a as all-zero tensor, representing previous step operation, each batch corresponds to one zero value

        # Calculate penalty matrix by calling StateHSMSP's get_c_mat method and convert to numpy array
        penalty_matrix = StateHSMSP.get_c_mat(loc.cpu().numpy())

        # My original state initialization was like this, according to your idea of margin calculation, I've already added time data into nodes, how should I modify this state
        penalty_tensor = torch.stack([torch.tensor(penalty, device=loc.device) for penalty in
                                      penalty_matrix])  # Convert penalty_matrix to tensor and create penalty_tensor on loc.device

        return StateHSMSP(  # Return an instance of StateHSMSP, initialize its various attributes
            loc=loc,  # Position loc
            dist=penalty_tensor,  # Penalty matrix penalty_tensor
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Batch index ids, use torch.arange to create a sequence
            first_a=prev_a,  # First operation first_a, set to prev_a+
            prev_a=prev_a,  # Previous step operation prev_a
            visited_=(torch.zeros( batch_size, 1, n_loc,dtype=torch.uint8, device=loc.device)
                if visited_dtype == torch.uint8  # If visited_dtype is torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)
                # Otherwise initialize with torch.int64 type
            ),
            lengths=torch.zeros(batch_size, 1, device=loc.device),  # Initialize path length lengths as all-zero tensor, this part can be modified to reward function
            cur_coord=None,  # Current coordinate cur_coord set to None
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),  # Initialize step counter i to zero
            graph=graph  # Graph structure graph
        )

    def get_final_cost(self):  # Get final cost
        assert self.all_finished()  # Ensure all steps completed
        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)  # Calculate total length

    def update(self, selected):  # Update state method
        prev_a = selected[:, None]  # Add dimension to selected node as previous node

        cur_coord = self.loc[self.ids, prev_a]  # Get current node coordinates
        lengths = self.lengths  # Initialize length
        if self.cur_coord is not None:  # If current coordinate not None
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # Update path length

        first_a = prev_a if self.i.item() == 0 else self.first_a  # Update first_a if this is first step

        if self.visited_.dtype == torch.uint8:  # Update visited nodes
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)  # Use scatter to update visited_
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)  # Use mask_long_scatter to update visited_

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1)  # Return updated state

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):  # Get current node
        return self.prev_a  # Return previously visited node

    def get_mask(self):  # Get visited nodes mask
        return self.visited  # Return visited nodes

    def get_graph_mask(self):  # Get graph mask
        batch_size, n_loc, _ = self.loc.size()  # Get batch size and node count
        if self.i.item() == 0:  # If first step
            return torch.zeros(batch_size, 1, n_loc, dtype=torch.uint8, device=self.loc.device)  # Return all-zero mask
        else:
            return self.graph.gather(1, self.prev_a.unsqueeze(-1).expand(-1, -1, n_loc))  # Return graph mask

    def get_graph(self):  # Get graph
        return self.graph  # Return graph representation

    def get_nn(self, k=None):  # Get nearest neighbor nodes
        if k is None:  # If k is None
            k = self.loc.size(-2) - self.i.item()  # Set k to remaining node count
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[
            1]  # Get nearest neighbor node indices

    def get_nn_current(self, k=None):  # Get nearest neighbor nodes for current node
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"  # Currently not implemented
        if k is None:  # If k is None
            k = self.loc.size(-2)  # Set k to node count
        k = min(k, self.loc.size(-2) - self.i.item())  # Set k to remaining node count
        return (
                self.dist[
                    self.ids,
                    self.prev_a
                ] +
                self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]  # Get nearest neighbor node indices

    def construct_solutions(self, actions):  # Construct solutions
        return actions  # Return actions