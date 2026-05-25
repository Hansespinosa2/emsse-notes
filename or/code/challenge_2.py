import pandas as pd
import pulp
import time
import os

# =========================
# CONFIG
# =========================
TIME_LIMIT_SECONDS = 600          # 10 minutes hard cap
CHECKPOINT_EVERY_SECONDS = 60     # save partial solutions every minute

OUTPUT_FILE = "orders.csv"


# =========================
# LOAD DATA
# =========================
data_files = {
    "demand": "demand_data.csv",
    "capacity": "availability_data.csv",
    "holding": "holding_cost_data.csv",
    "fixed_cost": "fixed_order_costs.csv",
    "initial_inv": "initial_inventory.csv",
}

demand = pd.read_csv(data_files["demand"])
capacity = pd.read_csv(data_files["capacity"])
holding = pd.read_csv(data_files["holding"])
fixed_cost = pd.read_csv(data_files["fixed_cost"])
initial_inv = pd.read_csv(data_files["initial_inv"])


# =========================
# INDEX SETS
# =========================
N = sorted(demand["Site"].unique())
T = sorted(demand["Day"].unique())
S = [0, 1, 2]  # sugar, milk, flavor
M = sorted(capacity["Reseller"].unique())


# =========================
# DICTIONARIES
# =========================
d = demand.set_index(["Site", "Day", "Product"])["Demand"].to_dict()
p = capacity.set_index(["Reseller", "Day", "Product"])["MaxQty"].to_dict()
q = holding.set_index(["Site", "Product"])["HoldingCost"].to_dict()
b = fixed_cost.set_index(["Reseller"])["FixedOrderCost"].to_dict()
v0 = initial_inv.set_index(["Site", "Product"])["InitialInventory"].to_dict()


# =========================
# BIG M (tight-ish bound)
# =========================
BIG_M = demand["Demand"].max() * len(T)


# =========================
# MODEL
# =========================
model = pulp.LpProblem("IceCreamMILP", pulp.LpMinimize)


# =========================
# VARIABLES
# =========================
x = pulp.LpVariable.dicts("x",
    (N, M, T, S),
    lowBound=0,
    cat="Continuous"
)

v = pulp.LpVariable.dicts("v",
    (N, range(len(T) + 1), S),
    lowBound=0,
    cat="Continuous"
)

y = pulp.LpVariable.dicts("y",
    (N, M, T),
    cat="Binary"
)


# =========================
# INITIAL INVENTORY
# =========================
for n in N:
    for s in S:
        model += v[n][0][s] == v0.get((n, s), 0)


# =========================
# OBJECTIVE
# =========================
holding_cost = pulp.lpSum(
    q[n, s] * v[n][t][s]
    for n in N
    for s in S
    for t in range(len(T))
)

fixed_order_cost = pulp.lpSum(
    b[m] * y[n][m][t]
    for n in N
    for m in M
    for t in T
)

model += holding_cost + fixed_order_cost


# =========================
# CONSTRAINTS
# =========================

# Inventory balance
for n in N:
    for s in S:
        for t in range(len(T)):
            model += (
                v[n][t + 1][s]
                == v[n][t][s]
                + pulp.lpSum(x[n][m][t][s] for m in M)
                - d.get((n, t, s), 0)
            )

# Reseller capacity
for m in M:
    for t in T:
        for s in S:
            model += (
                pulp.lpSum(x[n][m][t][s] for n in N)
                <= p.get((m, t, s), 0)
            )

# Fixed cost activation
for n in N:
    for m in M:
        for t in T:
            model += (
                pulp.lpSum(x[n][m][t][s] for s in S)
                <= BIG_M * y[n][m][t]
            )


# =========================
# SOLVER WITH TIME LIMIT
# =========================
solver = pulp.PULP_CBC_CMD(
    timeLimit=TIME_LIMIT_SECONDS,
    msg=True
)


start_time = time.time()

result_status = model.solve(solver)

print("\nSTATUS:", pulp.LpStatus[result_status])
print("OBJECTIVE:", pulp.value(model.objective))


# =========================
# CHECKPOINT + EXPORT FUNCTION
# =========================
def export_solution(filename):
    rows = []

    for n in N:
        for m in M:
            for t in T:
                for s in S:
                    val = pulp.value(x[n][m][t][s])
                    if val is None:
                        val = 0
                    if val > 1e-6:
                        rows.append([t, n, s, m, val])

    df = pd.DataFrame(rows, columns=[
        "Day", "Production Site", "Product", "Reseller", "Order Quantity"
    ])

    df.to_csv(filename, index=False)


# =========================
# FINAL OUTPUT
# =========================
export_solution(OUTPUT_FILE)

print(f"\nSaved final solution to {OUTPUT_FILE}")