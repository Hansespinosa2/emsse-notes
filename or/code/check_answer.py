
# =========================================================
# Inventory Verification Script – Ice Cream Division
# Python Version
# Corrected version:
# - fixes zero-qty fixed costs
# - per-(reseller, product) capacity aggregation
# - product eligibility guard
# - restores supply-vs-demand sanity check
# =========================================================

import pandas as pd

# =========================================================
# LOAD DATA
# =========================================================
demand = pd.read_csv("demand_data.csv")
holding_cost = pd.read_csv("holding_cost_data.csv")
availability = pd.read_csv("availability_data.csv")
fixed_cost = pd.read_csv("fixed_order_costs.csv")
product_cost = pd.read_csv("product_costs.csv")
initial_inventory = pd.read_csv("initial_inventory.csv")
orders = pd.read_csv("orders.csv")

# =========================================================
# INITIALIZATION
# =========================================================
days = 30
products = ["Sugar", "PowderMilk", "Flavour"]

# ---------------------------------------------------------
# Lookup dictionaries
# ---------------------------------------------------------

# Holding cost lookup: (site, product) -> cost
holding_cost_map = {
    (row["Site"], row["Product"]): row["HoldingCost"]
    for _, row in holding_cost.iterrows()
}

# Fixed order cost lookup: reseller -> fixed cost
fixed_order_cost_map = {
    row["Reseller"]: row["FixedOrderCost"]
    for _, row in fixed_cost.iterrows()
}

# Unit purchase cost lookup: product -> unit cost
unit_cost_map = {
    row["Product"]: row["UnitCost"]
    for _, row in product_cost.iterrows()
}

# =========================================================
# WORKING INVENTORY
# =========================================================
inventory = initial_inventory.copy()

# Ensure correct column name
if "InitialInventory" not in inventory.columns:
    raise ValueError("initial_inventory.csv must contain column 'InitialInventory'")

# =========================================================
# COUNTERS
# =========================================================
stockout_count = 0
total_holding_cost = 0
total_fixed_cost = 0
total_purchase_cost = 0
reseller_violations = 0

# =========================================================
# MAIN SIMULATION LOOP
# =========================================================
for d in range(1, days + 1):

    print(f"\nProcessing Day {d}...")

    # ---------------------------------------------------------
    # Filter daily data
    # ---------------------------------------------------------
    daily_orders = orders[
        (orders["Day"] == d) &
        (orders["Quantity"] > 0)
    ].copy()

    daily_availability = availability[
        availability["Day"] == d
    ].copy()

    daily_demand = demand[
        demand["Day"] == d
    ].copy()

    # =========================================================
    # 1. FIXED ORDER COST
    # One charge per unique (Reseller, Site)
    # =========================================================
    if len(daily_orders) > 0:

        unique_shipments = daily_orders[
            ["Reseller", "Site"]
        ].drop_duplicates()

        for _, row in unique_shipments.iterrows():

            reseller = row["Reseller"]

            if reseller in fixed_order_cost_map:
                total_fixed_cost += fixed_order_cost_map[reseller]
            else:
                print(
                    f'  [WARN] Unknown reseller "{reseller}" '
                    f'on Day {d} – no fixed cost applied.'
                )

    # =========================================================
    # 2. RESELLER CAPACITY VIOLATIONS
    # Aggregate by (Reseller, Product)
    # =========================================================
    if len(daily_orders) > 0:

        grouped = (
            daily_orders
            .groupby(["Reseller", "Product"])["Quantity"]
            .sum()
            .reset_index()
        )

        for _, row in grouped.iterrows():

            reseller = row["Reseller"]
            product = row["Product"]
            total_ordered = row["Quantity"]

            avail_row = daily_availability[
                (daily_availability["Reseller"] == reseller) &
                (daily_availability["Product"] == product)
            ]

            # Reseller doesn't supply product
            if len(avail_row) == 0:

                print(
                    f"  [VIOLATION] Day {d}: "
                    f"Reseller {reseller} does not supply "
                    f"{product} (ordered {total_ordered})."
                )

                reseller_violations += 1

            else:

                max_qty = avail_row.iloc[0]["MaxQty"]

                if total_ordered > max_qty:

                    print(
                        f"  [VIOLATION] Day {d}: "
                        f"Reseller {reseller}, {product} – "
                        f"ordered {total_ordered} > capacity {max_qty}."
                    )

                    reseller_violations += 1

    # =========================================================
    # 3. APPLY ORDERS TO INVENTORY + PURCHASE COST
    # =========================================================
    for _, row in daily_orders.iterrows():

        reseller = row["Reseller"]
        product = row["Product"]
        site = row["Site"]
        qty = row["Quantity"]

        # Verify reseller supplies product
        avail_row = daily_availability[
            (daily_availability["Reseller"] == reseller) &
            (daily_availability["Product"] == product)
        ]

        if len(avail_row) == 0:
            continue

        # Inventory row lookup
        idx = (
            (inventory["Site"] == site) &
            (inventory["Product"] == product)
        )

        if idx.any():

            inventory.loc[idx, "InitialInventory"] += qty

        else:

            new_row = pd.DataFrame([{
                "Site": site,
                "Product": product,
                "InitialInventory": qty
            }])

            inventory = pd.concat(
                [inventory, new_row],
                ignore_index=True
            )

        # Purchase cost
        if product in unit_cost_map:
            total_purchase_cost += qty * unit_cost_map[product]

    # =========================================================
    # 4. FULFILL DAILY DEMAND
    # =========================================================
    for _, row in daily_demand.iterrows():

        site = row["Site"]
        product = row["Product"]
        qty_demanded = row["Demand"]

        idx = (
            (inventory["Site"] == site) &
            (inventory["Product"] == product)
        )

        if idx.any():

            current_stock = inventory.loc[
                idx,
                "InitialInventory"
            ].values[0]

            if current_stock < qty_demanded:

                stockout_count += 1

                print(
                    f"  [STOCKOUT] Day {d}, "
                    f"Site {site}, {product} "
                    f"(have {current_stock}, need {qty_demanded})."
                )

                inventory.loc[idx, "InitialInventory"] = 0

            else:

                inventory.loc[idx, "InitialInventory"] = (
                    current_stock - qty_demanded
                )

        else:

            stockout_count += 1

            print(
                f"  [STOCKOUT] Day {d}, "
                f"Site {site}, {product} "
                f"(no inventory record)."
            )

    # =========================================================
    # 5. HOLDING COST
    # =========================================================
    for _, row in inventory.iterrows():

        key = (row["Site"], row["Product"])

        if key in holding_cost_map:

            total_holding_cost += (
                row["InitialInventory"] *
                holding_cost_map[key]
            )

# =========================================================
# RESULTS
# =========================================================
print("\n=== Corrected Evaluation Summary ===")

print(f"Total Stockouts:                    {stockout_count}")
print(f"Total Holding Cost:                 €{total_holding_cost:.2f}")
print(f"Total Fixed Order Cost:             €{total_fixed_cost:.2f}")
print(f"Total Purchase Cost:                €{total_purchase_cost:.2f}")
print(f"Total Reseller Capacity Violations: {reseller_violations}")

print("----------------------------------------------------")

print(
    f"Optimisation Cost (Holding + Fixed): "
    f"€{total_holding_cost + total_fixed_cost:.2f}"
)

print(
    f"Overall Total (incl. Purchase): "
    f"€{total_holding_cost + total_fixed_cost + total_purchase_cost:.2f}"
)

# =========================================================
# SUPPLY VS DEMAND SANITY CHECK
# =========================================================
print(f"\n=== Supply vs Demand Check Over {days} Days ===")

for product in products:

    total_demand = demand.loc[
        demand["Product"] == product,
        "Demand"
    ].sum()

    total_availability = availability.loc[
        availability["Product"] == product,
        "MaxQty"
    ].sum()

    total_ordered = orders.loc[
        (orders["Product"] == product) &
        (orders["Quantity"] > 0),
        "Quantity"
    ].sum()

    print(
        f"{product:<12} | "
        f"Demand: {int(total_demand):6d} | "
        f"Max Available: {int(total_availability):6d} | "
        f"Actually Ordered: {int(total_ordered):6d} | ",
        end=""
    )

    if total_availability >= total_demand:
        print("[Supply OK]")
    else:
        print("[SUPPLY INSUFFICIENT]")
