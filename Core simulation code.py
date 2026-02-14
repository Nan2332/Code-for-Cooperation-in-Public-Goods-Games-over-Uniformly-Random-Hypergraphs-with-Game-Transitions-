from tqdm import tqdm
import networkx as nx
import random
import math

# Configurable initial ratio parameters
coop_init_ratio = 0.5
g1_init_ratio = 0.5

# Parameter settings
N = 1000 # Network scale
g = 5 # Hyperedge order
delta = 0.05 # Resource advantage of high-value game (delta=r1-r2)
alpha = 5 # Defection sensitivity of hyperedges
r2 = 0.85 # Synergy factor of low-value game g2

monte_steps = 10000  # Total Monte Carlo steps
basic_steps_per_monte = N  # Basic steps per Monte Carlo step

# Naming output files according to specified format
file_suffix = f"g={g}_alpha={alpha}_r2={r2:.3f}_delta={delta:.3f}"
cooperator_ratio_output_file = f"fc_average_{file_suffix}.txt"  # Final result output file
fc_time_series_file = f"fc_step_{file_suffix}.txt"  # Time series data file of fc with steps


def URH_initialize(N, g):
    G = nx.Graph()
    # Create nodes
    for i in range(N):
        G.add_node(i)

    L = N / g * math.log(N, math.e)
    L = int(L // 1 + 1)
    print(f"Number of hyperedges L: {L}")

    nodes = {}
    for node in G:
        nodes[node] = []

    edges = {}
    for i in range(L):
        edge_i = random.sample(range(N), g)
        for node in edge_i:
            nodes[node].append(i)
        edges[i] = edge_i
    for key in nodes:
        if len(nodes[key]) == 0:
            G.remove_node(key)

    return G, edges, nodes, L


G, edges, nodes, L = URH_initialize(N, g)

# Actual number of nodes participating in simulation (filter out nodes without hyperedges)
num_nodes = len(nodes)
print(f"Number of nodes actually participating in the simulation: {num_nodes}\n")


def calculate_delta(g, r1, r2, c=1):
    if r2 < 1 and r1 < 1:
        return r1 * (g - 1) - (r2 - 1)
    elif r2 <= 1 and r1 >= 1:
        return r1 * g - r2
    elif r2 > 1 and r1 > 1:
        return r1 * g - 1
    return 0


def update_contribution(node, old_strategy, new_strategy, contribution, nodes):
    for edge in nodes[node]:
        contribution[edge] += new_strategy - old_strategy


def update_r_values(ni, strategy, alpha, r_values, nodes, edges, g):
    for edge in nodes[ni]:
        edge_nodes = edges[edge]
        contrib_li = sum(strategy[node] for node in edge_nodes)
        p = (contrib_li / g) ** alpha
        r_values[edge] = 1 if random.random() < p else 0


def initialize_r_values(strategy, alpha, edges, g):
    r_vals = {}
    # Calculate the number of hyperedges to be initialized as g1
    num_g1_edges = int(len(edges) * g1_init_ratio)
    # Randomly select the specified number of hyperedges as initial g1
    g1_edge_ids = random.sample(list(edges.keys()), num_g1_edges)

    for edge in edges.keys():
        r_vals[edge] = 1 if edge in g1_edge_ids else 0

    # Verify initial g1 ratio
    init_g1_count = sum(1 for v in r_vals.values() if v == 1)
    init_g1_actual_ratio = init_g1_count / len(edges)
    print(f"Initial number of g1 hyperedges: {init_g1_count}/{len(edges)}, actual ratio: {init_g1_actual_ratio:.4f} (target: {g1_init_ratio})")
    return r_vals


# Clear the final result file and write the header
with open(cooperator_ratio_output_file, 'w') as f:
    f.write("delta\talpha\tr1\tr2\tcooperator_ratio\n")

# Clear the fc time series data file and write the header
with open(fc_time_series_file, 'w') as f:
    f.write("step\talpha\tr1\tr2\tfc\n")


r1 = delta + r2
delta_val = calculate_delta(g, r1, r2)
print(f"===== Start simulation: alpha={alpha}, r2={r2:.3f}, r1={r1:.3f}, delta={delta_val:.6f} =====")

# Initialize strategy
num_coop_nodes = int(num_nodes * coop_init_ratio)
coop_node_ids = random.sample(list(nodes.keys()), num_coop_nodes)
strategy = {}
for node in nodes:
    strategy[node] = 1 if node in coop_node_ids else 0
# Verify initial cooperator ratio
init_coop_count = sum(1 for v in strategy.values() if v == 1)
init_coop_actual_ratio = init_coop_count / num_nodes
print(f"Initial number of cooperators: {init_coop_count}/{num_nodes}, actual ratio: {init_coop_actual_ratio:.4f} (target: {coop_init_ratio})")

# Initialize contribution values
contribution = {edge: sum(strategy[node] for node in edge_nodes) for edge, edge_nodes in edges.items()}
degree = {node: len(edge_list) for node, edge_list in nodes.items()}
payoff = {node: 0 for node in nodes}
average_payoff = {n: 0 for n in nodes}

# Initialize r values
r_values = initialize_r_values(strategy, alpha, edges, g)

cooperator_ratios = []  # Store fc for each step
# Main simulation loop
for monte_step in tqdm(range(monte_steps), desc=f"Simulation progress"):
    # Reset payoff
    average_payoff = {n: 0 for n in nodes}
    for _ in range(basic_steps_per_monte):
        payoff = {node: 0 for node in nodes}
        ni = random.choice(list(nodes.keys()))
        if not nodes[ni]:
            continue
        li = random.choice(nodes[ni])
        hyperedge_nodes = edges[li]

        hk_set = set()
        for nj in hyperedge_nodes:
            hk_set.update(nodes[nj])

        for hk in hk_set:
            current_r = r1 if r_values[hk] == 1 else r2
            contrib = contribution[hk]
            edge_nodes = edges[hk]
            for n in edge_nodes:
                if n in hyperedge_nodes:
                    cost = strategy[n]
                    pay = (contrib * current_r) - cost
                    payoff[n] += pay

        # Calculate average payoff for each node in the current hyperedge
        for n in hyperedge_nodes:
            average_payoff[n] = payoff[n] / degree[n] if degree[n] > 0 else 0

        # Find the node with maximum payoff
        maxi = None
        pmax = -math.inf
        for n in hyperedge_nodes:
            ap = average_payoff[n]
            if ap > pmax:
                pmax = ap
                maxi = n

        # Strategy update
        if strategy[ni] != strategy[maxi]:
            old_strategy = strategy[ni]
            prob = (pmax - average_payoff[ni]) / delta_val
            if random.random() < prob:
                strategy[ni] = strategy[maxi]
                new_strategy = strategy[ni]
                update_contribution(ni, old_strategy, new_strategy, contribution, nodes)

        # Update r values of hyperedges where node ni is located
        update_r_values(ni, strategy, alpha, r_values, nodes, edges, g)

    # Count the cooperator ratio fc of the current step
    cooperator_count = sum(1 for s in strategy.values() if s == 1)
    cooperator_ratio = cooperator_count / num_nodes
    cooperator_ratios.append(cooperator_ratio)

    # Print real-time data every 100 steps
    current_step = monte_step + 1  # Convert to steps starting from 1
    if current_step % 100 == 0:
        print(f"[Real-time data] Step: {current_step:4d} | alpha={alpha} | r2={r2:.3f} | fc={cooperator_ratio:.6f}")


    with open(fc_time_series_file, 'a') as f:
        f.write(f"{current_step}\t{alpha}\t{r1:.3f}\t{r2:.3f}\t{cooperator_ratio:.6f}\n")

# Take the cooperator ratio of the last 4000 steps as the final result
final_cooperator_ratio = sum(cooperator_ratios[-4000:]) / 4000

print(f"\n[Final result] alpha={alpha}, r2={r2:.3f}: final fc = {final_cooperator_ratio:.6f}\n")


with open(cooperator_ratio_output_file, 'a') as f:
    f.write(f"{delta}\t{alpha}\t{r1:.3f}\t{r2:.3f}\t{final_cooperator_ratio:.6f}\n")


print(f"\nFinal cooperator ratio results saved to: {cooperator_ratio_output_file}")
print(f"fc time series data (per step) saved to: {fc_time_series_file}")