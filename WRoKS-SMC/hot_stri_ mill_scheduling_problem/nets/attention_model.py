import math
import numpy as np
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel

from utils.tensor_functions import compute_in_batches
from utils.beam_search import CachedLookup
from utils.functions import sample_many


class AttentionModelFixed(NamedTuple):  # Define a class named AttentionModelFixed, inheriting from NamedTuple
    """
    AttentionModel decoder context fixed during decoding, so it can be precomputed/cached
    This class allows efficient indexing of multiple tensors at once
    """
    node_embeddings: torch.Tensor  # Tensor of node embeddings
    context_node_projected: torch.Tensor  # Tensor of projected context nodes
    glimpse_key: torch.Tensor  # Glimpse key tensor
    glimpse_val: torch.Tensor  # Glimpse value tensor
    logit_key: torch.Tensor  # Logit key tensor

    def __getitem__(self, key):  # Override getitem method
        if torch.is_tensor(key) or isinstance(key, slice):  # If key is tensor or slice
            return AttentionModelFixed(  # Return a new AttentionModelFixed instance
                node_embeddings=self.node_embeddings[key],  # Index node embeddings by key
                context_node_projected=self.context_node_projected[key],  # Index projected context nodes by key
                glimpse_key=self.glimpse_key[:, key],  # Index glimpse key tensor by key, dim 0 is heads
                glimpse_val=self.glimpse_val[:, key],  # Index glimpse value tensor by key, dim 0 is heads
                logit_key=self.logit_key[key]  # Index logit key tensor by key
            )
        return super(AttentionModelFixed, self).__getitem__(key)  # Otherwise call parent's getitem method


class AttentionModel(nn.Module):  # Define a class named AttentionModel, inheriting from nn.Module

    def __init__(self,  # Initialization method
                 problem,  # Problem type (e.g.,modeling of slabs)
                 machine,  # Number of slab attributes, there is an issue with subsequent calculation here
                 embedding_dim,  # Hidden dimension of encoder/decoder
                 encoder_class,  # Encoder class (e.g., GNN/Transformer/MLP)
                 n_encode_layers,  # Number of encoder layers
                 aggregation="sum",  # Aggregation function for GNN encoder
                 aggregation_graph="mean",  # Graph aggregation function
                 normalization="batch",  # Normalization scheme ('batch'/'layer'/'none')
                 learn_norm=True,  # Whether to enable learning affine transformations in normalization
                 track_norm=False,  # Whether to track training dataset statistics during normalization using batch statistics
                 gated=True,  # Whether to enable anisotropic GNN aggregation
                 n_heads=8,  # Number of attention heads for Transformer encoder/MHA decoder
                 tanh_clipping=10.0,  # Constant value for tanh clipping of decoder logits
                 mask_inner=True,  # Whether to use visit mask in decoder's inner function
                 mask_logits=True,  # Whether to use visit mask in decoder's logit calculation
                 mask_graph=False,  # Whether to use graph mask during decoding
                 checkpoint_encoder=False,  # Whether to use checkpointing in encoder embedding
                 shrink_size=None,  # Set to 20% of batch size or 16, whichever is larger, # Not applicable
                 extra_logging=False,  # Whether to perform additional logging for plotting histograms of embeddings
                 *args, **kwargs):  # Other parameters

        super(AttentionModel, self).__init__()  # Call parent class initialization method

        self.problem = problem  # Initialize problem type
        self.embedding_dim = embedding_dim  # Initialize embedding dimension
        self.encoder_class = encoder_class  # Initialize encoder class
        self.n_encode_layers = n_encode_layers  # Initialize number of encoder layers
        self.aggregation = aggregation  # Initialize aggregation function
        self.aggregation_graph = aggregation_graph  # Initialize graph aggregation function
        self.normalization = normalization  # Initialize normalization scheme
        self.learn_norm = learn_norm  # Initialize whether to learn affine transformations in normalization
        self.track_norm = track_norm  # Initialize whether to track training dataset statistics
        self.gated = gated  # Initialize whether to enable anisotropic GNN aggregation
        self.n_heads = n_heads  # Initialize number of attention heads
        self.tanh_clipping = tanh_clipping  # Initialize tanh clipping constant
        self.mask_inner = mask_inner  # Initialize whether to use visit mask in inner function
        self.mask_logits = mask_logits  # Initialize whether to use visit mask in logit calculation
        self.mask_graph = mask_graph  # Initialize whether to use graph mask during decoding
        self.checkpoint_encoder = checkpoint_encoder  # Initialize whether to use checkpointing in encoder embedding
        self.shrink_size = shrink_size  # Initialize shrink_size (not used)

        self.extra_logging = extra_logging  # Initialize whether to perform additional logging

        self.decode_type = None  # Initialize decode type
        self.temp = 800.0  # Initialize temperature

        self.allow_partial = problem.NAME == 'sdvrp'  # Determine if partial delivery is allowed
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'  # Determine if it's vehicle routing problem
        self.is_orienteering = problem.NAME == 'op'  # Determine if it's orienteering problem
        self.is_pctsp = problem.NAME == 'pctsp'  # Determine if it's prize collecting traveling salesman problem

        # Problem-specific context parameters (placeholder and step context dimension)
        # Not used because we only handle TSP
        if self.is_vrp or self.is_orienteering or self.is_pctsp:  # If VRP, orienteering, or PCTSP problem
            # Last node embedding + remaining capacity/remaining length/remaining collected reward
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:  # If PCTSP problem
                node_dim = 4  # Node dimension is 4: x, y, expected reward, penalty
            else:  # Otherwise
                node_dim = 3  # Node dimension is 3: x, y, demand/reward

            # Special depot node embedding projection
            self.init_embed_depot = nn.Linear(5, embedding_dim)  # Initialize depot node embedding projection

            if self.is_vrp and self.allow_partial:  # If VRP problem and partial delivery allowed
                # Need to include demand if split delivery is allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)  # Initialize node step projection

        else:
            assert problem.NAME in ("HSMSP"), "Unsupported problem: {}".format(problem.NAME)
            # step_context_dim is a context dimension used in sequence models (especially when solving traveling salesma or similar problems).
            # It combines context information of the current step with node embedding information,
            # helping the model better understand and handle the current state during decoding
            step_context_dim = 2 * embedding_dim  # Step context dimension is 2 times embedding dimension,
            node_dim = machine  # Node dimension is machine parameter = 11, considered as slab attributes

            # Learned input symbol for first action
            # nn.Parameter is a subclass of torch.Tensor that is automatically registered as a parameter of the module.
            # This method converts an ordinary torch.Tensor into a trainable parameter of the model, so these parameters
            # will be updated during optimization. Specifically:
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))  # Initialize placeholder parameter
            self.W_placeholder.data.uniform_(-1, 1)  # Initialize placeholder parameter with uniform distribution between -1 and 1

        # Input embedding layer # Here node_dim is 6 representing 6 attributes nn.Linear fully connected layer defines a linear layer (fully connected)
        self.init_embed = nn.Linear(node_dim, embedding_dim, bias=True)  # Initialize input embedding layer

        # Encoder model
        self.embedder = self.encoder_class(n_layers=n_encode_layers,  # Initialize encoder model 3 layers
                                           n_heads=n_heads,          # 8 heads
                                           hidden_dim=embedding_dim,   # Hidden layer dimension 128
                                           aggregation=aggregation,  # Specify feature aggregation method used in model. Aggregation method determines how to combine outputs of multi-head attention or other multi-node features. Common aggregation methods include weighted average, pooling, etc.
                                           norm=normalization,  # Specify which normalization technique to use
                                           learn_norm=learn_norm,  # Boolean indicating whether to learn parameters in normalization layer
                                           track_norm=track_norm,  # Boolean indicating whether to track normalization statistics during training
                                           gated=gated)

        # For each node, compute (glimpse key, glimpse value, logit key), so 3 times embedding dimension
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)  # Initialize node embedding projection layer

        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)  # Initialize fixed context projection layer

        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)  # Initialize step context projection layer

        assert embedding_dim % n_heads == 0  # Ensure embedding dimension is divisible by number of attention heads 128, 8

        # Note n_heads * val_dim == embedding_dim, so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)  # Initialize output projection layer

    def set_decode_type(self, decode_type, temp=None):  # Method to set decode type
        self.decode_type = decode_type  # Set decode type
        if temp is not None:  # If temperature parameter provided, set temperature
            self.temp = temp  # Set temperature parameter

    def forward(self, nodes, graph, targets, class_weights, supervised=True, return_pi=True):  # Forward propagation method
        """x, graph, targets, class_weights, supervised=True
        Args:
            nodes_mask:
            nodes: Input graph nodes (B x V x 2)  # Input graph nodes (B x V x 2)    B represents number of graphs in input data
            graph: Graph negative adjacency matrix (B x V x V)  # Graph negative adjacency matrix (B x V x V)
            supervised: Whether to use supervised learning, enables teacher forcing and negative log likelihood loss calculation  # Whether to use supervised learning
            targets: Targets for teacher forcing and negative log likelihood loss  # Targets
            return_pi: Whether to return output sequence  # Whether to return output sequence
                       (Incompatible with DataParallel because results on different GPUs may have different lengths)  # Incompatible with DataParallel
        """
        # Use GNN to embed input graph batch (B x V x H)  Program set to False, save intermediate computation graph
        if self.checkpoint_encoder:  # If checkpoint encoder enabled # Save GPU, don't save intermediate computation graph, slower to recompute during backpropagation
            embeddings = checkpoint(self.embedder, self._init_embed(nodes), graph)  # Use checkpoint encoder embedding
        else:  # Otherwise  This paper chooses
            embeddings = self.embedder(self._init_embed(nodes), graph)  # Direct embedding, save intermediate computation graph, faster backpropagation without recomputation

        if self.extra_logging:  # If additional logging enabled
            self.embeddings_batch = embeddings  # Record embedding batch

        # Imitation learning
        if self.problem.NAME == 'HSMSP' and supervised:  # Imitation learning
            assert targets is not None, "Pass targets during training in supervised mode"

            # Model forward propagation
            _log_p, pi = self._inner(nodes, graph, embeddings, targets, supervised=supervised)

            # Get predicted path cost
            cost, expected_margin, mask = self.problem.get_costs(nodes.cpu(), pi.cpu())
            cost = cost.cuda()

            # Dynamic normalization of cost
            if not hasattr(self, 'cost_max') or not hasattr(self, 'cost_min'):
                self.cost_max = cost.max().item()
                self.cost_min = cost.min().item()
            else:
                alpha = 0.9  # EMA weight
                self.cost_max = max(self.cost_max, cost.max().item() * alpha + self.cost_max * (1 - alpha))
                self.cost_min = min(self.cost_min, cost.min().item() * alpha + self.cost_min * (1 - alpha))

            if self.cost_max - self.cost_min < 1e-8:
                normalized_cost = torch.zeros_like(cost)
            else:
                normalized_cost = (cost - self.cost_min) / (self.cost_max - self.cost_min)
                normalized_cost = torch.clamp(normalized_cost, 0, 1)

            # Imitation learning loss
            logits = _log_p.permute(0, 2, 1)
            logits = torch.where(logits == -float(np.inf), torch.full_like(logits, -1000), logits)
            nll_loss = nn.NLLLoss(reduction='mean')(logits, targets)
            print('Imitation learning loss:', nll_loss)

            # Policy gradient loss (for cost optimization)
            log_prob = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1).sum(1)
            baseline = normalized_cost.mean()
            advantage = normalized_cost - baseline
            rl_loss = -(advantage.detach() * log_prob).mean()
            print('Policy gradient loss:', rl_loss)

            # Total loss = imitation learning loss + cost-related loss
            cost_weight = 0.9  # Weight for cost-related loss
            loss = nll_loss + (1 - cost_weight) * rl_loss

            if return_pi:
                return cost, loss, pi, expected_margin
            return cost, loss

        # Reinforcement learning and inference
        else:
            # Run inner function
            _log_p, pi = self._inner(nodes, graph, embeddings)  # Get log probability and path

            if self.extra_logging:  # If additional logging enabled

                self.log_p_batch = _log_p  # Record log probability batch

                self.log_p_sel_batch = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)  # Record selected log probability batch

            # Get predicted cost
            cost, expected_margin, mask = self.problem.get_costs(nodes.cpu(), pi.cpu())  # Get path cost and mask
            cost = cost.cuda()  # Move cost to GPU
            # Calculate log likelihood in model because each action returns log likelihood incompatible with DataParallel
            # (because sequences may have different lengths on different GPUs)
            ll = self._calc_log_likelihood(_log_p, pi, mask)  # Calculate log likelihood

            if return_pi:  # If return output sequence
                return cost, ll, pi, expected_margin  # Return cost, log likelihood, and path
            return cost, ll  # Return cost and log likelihood

    def beam_search(self, *args, **kwargs):  # Beam search method
        """ # Helper method to call beam search
        """
        return self.problem.beam_search(*args, **kwargs, model=self)  # Call beam search method of problem class, passing model instance

    def _calc_log_likelihood(self, _log_p, a, mask):  # Method to calculate log likelihood

        # # Get log probability corresponding to selected action
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)  # Get log probability of selected action

        # # Optional: mask actions unrelated to target
        if mask is not None:  # If mask is not None
            log_p[mask] = 0  # Mask actions unrelated to target

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"  # Assert log probability greater than -1000

        # # Calculate log likelihood
        return log_p.sum(1)  # Return sum of log likelihood

    def _init_embed(self, nodes):  # Method to initialize embedding
        if self.is_vrp or self.is_orienteering or self.is_pctsp:  # If VRP, orienteering, or PCTSP problem
            if self.is_vrp:  # If VRP problem
                features = ('demand',)  # Feature is demand
            elif self.is_orienteering:  # If orienteering problem
                features = ('prize',)  # Feature is prize
            else:  # Otherwise
                assert self.is_pctsp  # Assert PCTSP problem
                features = ('deterministic_prize', 'penalty')  # Features are deterministic prize and penalty
            return torch.cat(  # Return concatenated tensor
                (
                    self.init_embed_depot(nodes['depot'])[:, None, :],  # Initial embed depot node
                    self.init_embed(torch.cat((  # Initial embed
                        nodes['loc'],  # Location
                        *(nodes[feat][:, :, None] for feat in features)  # Features
                    ), -1))  # Concatenate features
                ),
                1  # Concatenation dimension
            )

        return self.init_embed(nodes)  # Return initial embed nodes

    def _inner(self, nodes, graph, embeddings, targets, supervised=None):  # Inner method
        outputs = []  # Output list
        sequences = []  # Sequence list

        # # Create problem state for masking (track which nodes have been visited)
        # Finally found the core modification place, first generate an initial state
        state = self.problem.make_state(nodes, graph)  # Generate initial state

        #   # Calculate glimpse keys and values and logit keys for reuse\\
        # The main purpose of this code is to precompute and cache fixed parameters that don't need to be
        # recalculated at each step, such as projected node embeddings and fixed context of graph embeddings,
        # thereby improving model efficiency during decoding
        fixed = self._precompute(embeddings)  # Precompute fixed parameters

        batch_size, num_nodes, _ = nodes.shape  # Get batch size and number of nodes _ is data dimension 2, 109, 11

        #   # Execute decoding steps
        i, count = 0, 1  # Initialize step counter
        # Loop continues as long as self.shrink_size is not None, or state.all_finished() returns False
        # Loop only terminates when self.shrink_size is None and state.all_finished() returns True  shrink_size=None
        while not (self.shrink_size is None and state.all_finished()):  # If decoding not finished or shrink size limit not reached

            # self.shrink_size=None
            if self.shrink_size is not None:  # If shrink size limit exists
                unfinished = torch.nonzero(state.get_finished() == 0)  # Get indices of unfinished states
                if len(unfinished) == 0:  # If no unfinished states
                    break  # Break loop
                unfinished = unfinished[:, 0]  # Get index values of unfinished states
                # # Check if can shrink by at least shrink_size and keep at least 16
                # # Otherwise batch normalization will not work properly and is inefficient
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:  # If unfinished count within reasonable range
                    # # Filter state
                    state = state[unfinished]  # Update state
                    fixed = fixed[unfinished]  # Update fixed parameters

            # # Get selection probability for next action
            log_p, mask = self._get_log_p(fixed, state)  # Get log probability and mask

            #   # Select index of next node in sequence
            # Teacher forcing mechanism refers to using true target values (rather than model predictions) to guide model learning
            # during training at certain steps. This helps the model converge to the correct solution faster
            if self.problem.NAME == 'HSMSP' and supervised:
                # # Teacher forcing in supervised mode   tensor([0], device='cuda:0')
                t_idx = torch.LongTensor([i]).to(nodes.device)  # Create current step index
                # This line selects target action for current step from targets. targets is a tensor containing target sequences
                # index_select method selects target actions based on indices in t_idx. dim = -1 indicates index selection on last dimension.
                # Selected result is reshaped with view(batch_size) to fit current batch size
                selected = targets.index_select(dim=-1, index=t_idx).squeeze(-1)

                # selected = selected.squeeze(1)
                selected = selected.long()

            else:  # Otherwise
                ## Select node Calculate selection action probability and mask invalid actions
                selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            # # Update problem state
            state = state.update(selected)  # Update state based on selected action

            #   # Make log_p and selected action conform to expected output size by "unshrinking"
            if self.shrink_size is not None and state.ids.size(0) < batch_size:  # If shrink size limit exists and current batch size less than initial batch size
                log_p_, selected_ = log_p, selected  # Save current log_p and selected action
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])  # Initialize new log_p
                selected = selected_.new_zeros(batch_size)  # Initialize new selected action

                log_p[state.ids[:, 0]] = log_p_  # Restore log_p
                selected[state.ids[:, 0]] = selected_  # Restore selected action

            sequences.append(selected)  # Add selected action to sequence list
            # # Collect step output
            outputs.append(log_p[:, 0, :])  # Add log_p to output list
            i += 1  # Increment step counter

        return torch.stack(outputs, 1), torch.stack(sequences, 1)  # Return tensors of outputs and sequences

    def _select_node(self, probs, mask):  # Method to select node
        assert (probs == probs).all(), "Probs should not contain any NaNs"  # Assert probabilities contain no NaNs

        if self.decode_type == "greedy":  # If decode type is greedy
            _, selected = probs.max(1)  # Select index with maximum probability
            assert not mask.gather(1, selected.unsqueeze(  # Assert selected index not in mask
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":  # If decode type is sampling
            selected = probs.multinomial(1).squeeze(1)  # Multinomial sampling

            #   # Check if sampling is correct, may fail due to GPU errors
            #   # Reference link
            while mask.gather(1, selected.unsqueeze(-1)).data.any():  # While selected index in mask
                print('Sampled bad values, resampling!')  # Print error message, resample
                selected = probs.multinomial(1).squeeze(1)  # Multinomial resampling

        else:  # Otherwise
            assert False, "Unknown decode type"  # Assert error, unknown decode type

        return selected  # Return selected index

    def _precompute(self, embeddings, num_steps=1):  # Precompute method
        #   # Fixed context projection calculated only once for efficiency
        if self.aggregation_graph == "sum":  # If aggregation method is sum
            graph_embed = embeddings.sum(1)  # Calculate sum of node embeddings
        elif self.aggregation_graph == "max":  # If aggregation method is max
            graph_embed = embeddings.max(1)[0]  # Calculate max of node embeddings
        elif self.aggregation_graph == "mean":  # If aggregation method is mean    This paper's method
            graph_embed = embeddings.mean(1)  # Calculate mean of node embeddings
        else:  # Default: disable graph embedding  # Default: disable graph embedding
            graph_embed = embeddings.sum(1) * 0.0  # Set graph embedding to 0

        # # Fixed context shape (batch_size, 1, embedding_dim) for broadcasting with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]  # Project fixed context

        #   # Node embedding projection for attention calculated only once
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)  # Split projected node embeddings

        #   # No need to rearrange logit key because there's only one head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),  # Create glimpse key heads
            self._make_heads(glimpse_val_fixed, num_steps),  # Create glimpse value heads
            logit_key_fixed.contiguous()  # Get contiguous logit key
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)  # Return fixed attention model data

    def _get_log_p(self, fixed, state,  normalize=False):  # Method to get sorting probability

        # # Calculate query = context node embedding
        # Note: there's an issue with this part of the program, state calculation has problems    self.project_step_context embedding context network, is neural network layer
        # Here query is obtained by adding global context information (fixed.context_node_projected) with dynamic context information of current step (self.project_step_context(...))

        # This addition operation combines global information with local information of current step, making query contain both types of information,
        # which is very useful for subsequent attention mechanism or other operations requiring combination of global and local information
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))  # Calculate query

        # # Calculate node keys and values, that's this part and the following _one_to_many_logits
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)  # Get attention node data

        #   # Calculate mask, mask next actions based on previous actions
        mask = state.get_mask()  # Get mask

        graph_mask = None
        if self.mask_graph:  # If graph mask needed
            # # Calculate graph mask, mask next actions based on graph structure
            graph_mask = state.get_graph_mask()  # Get graph mask

        # # Calculate logits (unnormalized log probabilities)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask)  # Calculate logits and glimpse

        if normalize:  # If normalization needed
            log_p = F.log_softmax(log_p / self.temp, dim=-1)  # Normalize log probability

        assert not torch.isnan(log_p).any()  # Assert log probability contains no NaNs

        return log_p, mask  # Return log probability and mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):  # Return context for each step, supports simultaneous calculation of multiple steps (improving model evaluation efficiency)
        current_node = state.get_current_node()  # Get current node  # Return previously visited node
        batch_size, num_steps = current_node.size()  # Get batch size and number of steps

        if num_steps == 1:  # If only one step, special handling needed, may be first step or not
            if state.i.item() == 0:
                # First step and only one step, ignore prev_a (this is placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(1, torch.cat((state.first_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)

        # More than one step, assume always starts with first step
        embeddings_per_step = embeddings.gather(1, current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
        )
        return torch.cat((
            # First step placeholder, concatenate on dimension 1 (timestep)
            self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
            # Second step, concatenate embedding of first step and embedding of current/previous step (on dimension 2, context dimension)
            torch.cat((
                embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                embeddings_per_step
            ), 2)
        ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None):  # Calculate one-to-many logits
        batch_size, num_steps, embed_dim = query.size()  # Get batch size, number of steps, and embedding dimension
        key_size = val_size = embed_dim // self.n_heads  # Calculate key and value size

        # Calculate glimpse, rearrange dimensions to (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to calculate compatibility (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -1e10  # Mask

            if self.mask_graph:
                compatibility[graph_mask[None, :, :, None, :].expand_as(compatibility)] = -1e10  # Mask

        # Batch matrix multiplication to calculate heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projection of glimpse is not needed because this can be absorbed into project_out
        final_Q = glimpse
        # Batch matrix multiplication to calculate logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # Calculate probability by masking graph, clipping and masking visited nodes
        if self.mask_logits and self.mask_graph:
            logits[graph_mask] = -1e10
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -1e10

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):  # Get attention node data
        if self.is_vrp and self.allow_partial:  # If vehicle routing problem and partial delivery allowed
            # Need to provide information on how much each node has been served
            # Clone demand because they are needed during backpropagation and will be updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Concatenated projection equivalent to additive projection, but more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )


        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):  # Create heads in multi-head attention mechanism
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps  # Ensure number of steps matches

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
                .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
                .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )