import numpy as np
import time
from problems.HSMSP.penalty_func import get_p_hard, get_p_thick, get_p_width

np.random.seed(1234)

max_batch_len = 800000  # Maximum rolling kilometers for the same rolling unit
max_same_width_len = 350000  # Maximum rolling kilometers for the same width


def calculated_penalty(M, j_seq):  # Define function to calculate single-machine scheduling time j_seq:[2, 0, 1]
    num_of_slab = len(j_seq)
    reward = 0
    for i in range(1, num_of_slab):
        reward += get_p_width(M[j_seq[i-1]][0] - M[j_seq[i]][0]) + \
                  get_p_hard(M[j_seq[i-1]][2] - M[j_seq[i]][2]) + \
                  get_p_thick(M[j_seq[i-1]][1] - M[j_seq[i]][1])
    return reward  # Return total penalty


def calculated_max_penalty(M, j_seq):
    """Calculate maximum possible jump penalty"""
    # Get max and min values for width, thickness, and hardness
    width_min = min([M[j_seq[i]][0] for i in range(len(j_seq))])
    width_max = max([M[j_seq[i]][0] for i in range(len(j_seq))])
    thick_min = min([M[j_seq[i]][1] for i in range(len(j_seq))])
    thick_max = max([M[j_seq[i]][1] for i in range(len(j_seq))])
    hard_min = min([M[j_seq[i]][2] for i in range(len(j_seq))])
    hard_max = max([M[j_seq[i]][2] for i in range(len(j_seq))])

    # Calculate maximum difference for each attribute
    max_width_diff = abs(width_max - width_min)
    max_thick_diff = abs(thick_max - thick_min)
    max_hard_diff = abs(hard_max - hard_min)

    # Calculate maximum penalty value
    max_penalty = 0
    num_of_slab = len(j_seq)
    for i in range(1, num_of_slab):
        max_penalty += get_p_width(max_width_diff) + \
                       get_p_thick(max_thick_diff) + \
                       get_p_hard(max_hard_diff)

    return max_penalty


def calculate_time_margin_with_uncertainty(pt, tmp, omega_L, omega_F):
    """Calculate time margin considering uncertainty"""
    total_time_margin_max = 0
    total_time_margin_min = 0
    time_margin_max = []
    time_margin_min = []

    for i in tmp:
        # Build uncertain datasets P for heating furnace and insulation pit

        # Calculate maximum and minimum actual yard residence time
        actual_slabyard_time_max = pt[i][6] + omega_L[i] * (pt[i][5] - pt[i][4])  # Use maximum deviation
        actual_slabyard_time_min = pt[i][6] - (1 - omega_L[i]) * (pt[i][5] - pt[i][4])  # Use minimum deviation

        # Calculate maximum and minimum actual furnace residence time
        actual_furnace_time_max = pt[i][9] + omega_F[i] * (pt[i][8] - pt[i][7])  # Use maximum deviation
        actual_furnace_time_min = pt[i][9] - (1 - omega_F[i]) * (pt[i][8] - pt[i][7])  # Use minimum deviation

        # Calculate heating furnace and insulation pit time margin
        slabyard_time_margin_max = actual_slabyard_time_max - pt[i][4]  # Maximum yard residence time margin
        # Ensure minimum time margin is not less than 0
        slabyard_time_margin_min = max(0, actual_slabyard_time_min - pt[i][4])  # Minimum yard residence time margin

        furnace_time_margin_max = actual_furnace_time_max - pt[i][7]  # Maximum furnace residence time margin
        furnace_time_margin_min = max(0, actual_furnace_time_min - pt[i][7])  # Minimum furnace residence time margin

        # Calculate total time margin
        total_time_margin_max += (slabyard_time_margin_max + furnace_time_margin_max)
        total_time_margin_min += (slabyard_time_margin_min + furnace_time_margin_min)

        # Store time margin for single rolling unit
        time_margin_max.append(slabyard_time_margin_max + furnace_time_margin_max)
        time_margin_min.append(slabyard_time_margin_min + furnace_time_margin_min)

    # Return maximum and minimum total time margin
    return total_time_margin_max, total_time_margin_min, time_margin_max, time_margin_min


def generate_uncertainty_samples(n, num_samples):
    """Generate uncertainty samples"""
    omega_L_samples = np.random.uniform(0, 1, size=(num_samples, n))
    omega_F_samples = np.random.uniform(0, 1, size=(num_samples, n))
    return omega_L_samples, omega_F_samples


def compute_robust_objective(pt, tmp, w, alpha, beta):
    """Compute robust optimization objective function"""
    # num_samples = 1000  # Generate large number of uncertainty samples to simulate worst case
    omega_L_samples, omega_F_samples = generate_uncertainty_samples(len(tmp), len(tmp))

    max_time_margin = float('-inf')  # Maximum time margin
    min_time_margin = float('inf')  # Minimum time margin
    total_expected_time = 0  # For calculating expected time

    # Traverse all samples, calculate maximum and minimum time margin
    for omega_L, omega_F in zip(omega_L_samples, omega_F_samples):
        total_time_margin_max, total_time_margin_min, time_margin_max, time_margin_min = calculate_time_margin_with_uncertainty(pt, tmp, omega_L, omega_F)

        # Update worst-case maximum time margin
        max_time_margin = max(max_time_margin, min(time_margin_max))

        # Update worst-case minimum time margin, including negative value handling
        min_time_margin = min(min_time_margin, max(time_margin_min))

        # Accumulate to calculate expected total time
        total_expected_time += (total_time_margin_max + total_time_margin_min) / 2

    # Calculate average expected time
    expected_time = total_expected_time / len(tmp)

    # Calculate first objective function f1 (jump penalty)
    f1_value = calculated_penalty(pt, tmp)  # Calculate jump penalty

    f2_value = ((max_time_margin + min_time_margin) / 2 + alpha * expected_time-min_time_margin) / 60  # Maximum possible value of second objective function

    total_penalty = w * f1_value + (1 - w) * f2_value

    return total_penalty, f1_value, f2_value, expected_time

def neh_Rollingunit(pt):
    seq_tmp = [i for i in range(len(pt))]  # Create index list for all jobs
    global alpha, beta, w  # Declare global variables alpha, beta and w
    w = 0.9  # Set value of w, possibly for adjusting weights in multi-objective optimization
    alpha = 0.2  # Set initial value of alpha, a parameter for calculation
    beta = 100  # Set initial value of beta, another parameter for calculation

    # Filter valid sequences, i.e., indices where jobs are not all zeros
    valid_seq = [i for i in seq_tmp if not (pt[i][0] == 0 and pt[i][1] == 0 and pt[i][2] == 0)]
    # Filter invalid sequences, i.e., indices where all three job parameters are zero
    invalid_seq = [i for i in seq_tmp if pt[i][0] == 0 and pt[i][1] == 0 and pt[i][2] == 0]

    length_list = 0  # Initialize length accumulator
    same_width_list = [0, 0]  # Initialize list for tracking jobs with same width

    # Sort first two elements of valid sequence in descending order based on first parameter
    seq = sorted(valid_seq[:2], key=lambda x: pt[x][0], reverse=True)

    total_cost, expected_margin = 0, 0
    # Loop through remaining valid sequences
    for i in range(2, len(valid_seq)):
        # Check if current accumulated length plus new job length exceeds maximum batch length
        if length_list + pt[valid_seq[i]][3] <= max_batch_len:
            # Check if new job has same width as previous jobs and accumulated length exceeds maximum same width limit
            if pt[valid_seq[i]][0] == same_width_list[0] and same_width_list[1] + pt[valid_seq[i]][3] > max_same_width_len:
                continue
            else:
                new_job_index = i  # Determine index of new job
                new_job = valid_seq[new_job_index]  # Get new job
                det, time_margin, f2_value = [], [], []  # Initialize cost and time margin lists

                # Try inserting new job into all possible positions of current sequence and calculate cost
                for k in range(len(seq) + 1):
                    tmp = seq[:]
                    tmp.insert(k, new_job)
                    total_penalty, f1, f2, expected_time = compute_robust_objective(pt, tmp, w, alpha, beta)   # Calculate multi-objective cost
                    det.append(total_penalty)
                    time_margin.append(expected_time)
                    f2_value.append(f2)
                # Select insertion position that minimizes cost
                min_det_value = min(det)
                det_index = len(det) - 1 - det[::-1].index(min_det_value)
                seq.insert(det_index, new_job)
                total_cost = det[det_index]
                expected_margin = time_margin[det_index]

                # Update accumulated length and width tracking
                length_list += pt[new_job][3]
                if same_width_list[0] == pt[new_job][0]:
                    same_width_list[1] += pt[new_job][3]
                else:
                    same_width_list[0], same_width_list[1] = pt[new_job][0], pt[new_job][3]

    # Add invalid sequences to the end
    seq.extend(invalid_seq)

    # Convert sequence to NumPy array
    seq = np.array(seq)

    # Return final total cost, sequence, and two objective values
    return total_cost, seq, expected_margin


# Function to calculate time margin
def time_margin(pt, idx):
    """Calculate time margin"""
    total_time_margin = 0
    # Calculate yard time margin
    slabyard_time_margin = pt[idx][5] - pt[idx][6]  # Maximum yard time - actual yard time
    # Calculate furnace residence time margin
    furnace_time_margin = pt[idx][7] - pt[idx][8]  # Maximum furnace residence time - actual furnace residence time
    # Accumulate time margin
    total_time_margin += (slabyard_time_margin + furnace_time_margin)

    return total_time_margin


def neh_text(pt):  # Define NEH slab scheduling algorithm
    # Initialize initial sequence
    seq_tmp = [i for i in range(len(pt))]
    seq = np.array(seq_tmp)
    reward = calculated_penalty(pt, seq)
    # Calculate minimum time margin
    # Filter out slabs with all-zero data
    valid_seq = [idx for idx in seq if not (pt[idx][0] == 0 and pt[idx][1] == 0 and pt[idx][2] == 0)]

    # Calculate time margin for valid slabs
    time_margins = [time_margin(pt, idx) for idx in valid_seq]
    time_margin_min = min(time_margins) if time_margins else 0  # Ensure time margin is not empty

    return reward, seq, time_margin_min  # Return final scheduling sequence