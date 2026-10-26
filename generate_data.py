"""
Synthetic micro-fulfilment dataset generator for the Inventory Integrity project.

Simulates a small dark store for N days: products, storage locations, inbound
deliveries (putaway) and picking events. Known integrity errors are injected on
purpose and written to a separate ground_truth/ folder, so the detection rules
in integrity_checks.py can be scored for precision and recall.

The simulation tracks two versions of stock for every (location, SKU, batch):
  * physical stock - what is actually on the shelf
  * system stock   - what the WMS believes is on the shelf
Errors make the two drift apart, which is exactly what happens in real stores.

Injected errors
  LOCATION_MISMATCH  picker takes stock from the location the app showed, but the
                     system deducts it from a different location. More likely at
                     peak hours and on a few "problem shelves" with confusing labels.
  FIFO_VIOLATION     picker takes a newer batch while an older one is physically
                     available. More likely at peak hours.
  SPLIT_STORAGE      one delivery batch (same SKU, same expiry) put away into two
                     locations. More likely in the chilled zone and the 06:00 rush.
  SHORT_DATED        supplier delivers stock with only 1-3 days of shelf life left,
                     which feeds the expiry-risk check.

Usage
  python generate_data.py                          # built-in product list
  python generate_data.py --days 28 --seed 7 --out data
  python generate_data.py --instacart-dir path/to/instacart   # real product names
"""

from __future__ import annotations

import argparse
import json
import math
import string
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CONFIG = {
    "n_skus": 500,
    "start_date": "2026-09-01",
    "shelves_per_aisle": 12,
    "levels_per_shelf": 5,
    "overflow_share": 0.25,          # extra locations per zone beyond SKU home bins
    # error rates (base, before multipliers)
    "p_mismatch": 0.018,
    "p_fifo": 0.008,
    "p_split": 0.03,
    "p_short_dated": 0.04,
    # multipliers that create learnable patterns
    "peak_multiplier_mismatch": 2.5,
    "peak_multiplier_fifo": 2.0,
    "problem_shelf_multiplier": 4.0,
    "chilled_split_multiplier": 2.0,
    "rush_split_multiplier": 1.5,
    # data-quality noise
    "p_scan_missing": 0.12,          # picker confirmed manually, no location scan
    "p_reason_recorded": 0.85,       # redirect reason code is entered most of the time
    "p_expiry_sweep": 0.9,           # chance an expired batch is found and written off each morning
    "expiry_risk_horizon_days": 3,
}

PEAK_HOURS = {11, 12, 13, 17, 18, 19, 20}
# relative pick volume per hour of day (store picks 07:00-23:00)
HOURLY_WEIGHTS = {
    7: 2, 8: 3, 9: 4, 10: 5, 11: 8, 12: 9, 13: 7, 14: 5,
    15: 5, 16: 6, 17: 9, 18: 10, 19: 9, 20: 7, 21: 5, 22: 3,
}

# storage type -> (shelf life range in days, case sizes)
STORAGE = {
    "AMB": ((60, 540), [6, 12, 24]),
    "CHL": ((5, 14), [4, 6, 8]),
    "FRZ": ((90, 365), [4, 6, 10]),
}

# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------

BUILTIN_PRODUCTS = {
    # category: (storage type, base products, variants)
    "Dairy": ("CHL", ["Whole Milk", "Skimmed Milk", "Oat Drink", "Greek Yogurt", "Natural Yogurt",
                      "Butter", "Cream Cheese", "Cheddar", "Mozzarella", "Feta", "Cottage Cheese",
                      "Sour Cream", "Kefir"], ["1L", "500ml", "200g", "400g", "Organic", "Lactose Free"]),
    "Meat & Fish": ("CHL", ["Chicken Breast", "Minced Beef", "Pork Chops", "Salmon Fillet", "Smoked Ham",
                            "Turkey Slices", "Bacon", "Chorizo"], ["300g", "500g", "Family Pack", "Organic"]),
    "Ready Meals": ("CHL", ["Pasta Salad", "Hummus", "Sushi Box", "Chicken Wrap", "Caesar Salad",
                            "Falafel Bowl"], ["Small", "Large", "Vegan"]),
    "Frozen": ("FRZ", ["Pizza Margherita", "Pizza Pepperoni", "Frozen Peas", "Fish Fingers", "Ice Cream Vanilla",
                       "Ice Cream Chocolate", "French Fries", "Mixed Berries", "Spinach", "Dumplings"],
               ["400g", "1kg", "Family Size", "Organic"]),
    "Bakery": ("AMB", ["Rye Bread", "Sourdough Loaf", "Croissants", "Tortilla Wraps", "Bagels",
                       "Crispbread"], ["4-pack", "6-pack", "Wholegrain"]),
    "Pantry": ("AMB", ["Spaghetti", "Penne", "Basmati Rice", "Jasmine Rice", "Rolled Oats", "Granola",
                       "Tomato Sauce", "Chopped Tomatoes", "Chickpeas", "Kidney Beans", "Olive Oil",
                       "Peanut Butter", "Honey", "Strawberry Jam", "Flour", "Sugar"],
               ["500g", "1kg", "Organic", "Value"]),
    "Snacks": ("AMB", ["Potato Chips", "Tortilla Chips", "Salted Peanuts", "Dark Chocolate", "Milk Chocolate",
                       "Protein Bar", "Rice Cakes", "Popcorn", "Gummy Bears", "Licorice"],
               ["Small", "Large", "Sea Salt", "Sharing Bag"]),
    "Beverages": ("AMB", ["Sparkling Water", "Still Water", "Orange Juice", "Apple Juice", "Cola", "Cola Zero",
                          "Energy Drink", "Ground Coffee", "Coffee Beans", "Green Tea", "Black Tea"],
                  ["330ml", "1.5L", "6-pack", "Organic"]),
    "Household": ("AMB", ["Dish Soap", "Laundry Detergent", "Paper Towels", "Toilet Paper", "Trash Bags",
                          "Sponges"], ["Small", "Large", "Eco"]),
    "Personal Care": ("AMB", ["Shampoo", "Conditioner", "Toothpaste", "Shower Gel", "Deodorant",
                              "Hand Soap"], ["Sensitive", "Fresh", "Travel Size"]),
}

BRANDS = ["Nordic Fields", "Daily Choice", "GreenLeaf", "Harbor & Co", "Polar Pantry", "Sunny Farm"]

# Instacart departments -> storage type
INSTACART_STORAGE = {
    "frozen": "FRZ",
    "dairy eggs": "CHL", "meat seafood": "CHL", "deli": "CHL", "produce": "CHL",
}


def builtin_products(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for category, (storage, bases, variants) in BUILTIN_PRODUCTS.items():
        for base in bases:
            for variant in variants:
                for brand in BRANDS:
                    rows.append((f"{brand} {base} {variant}", category, storage))
    df = pd.DataFrame(rows, columns=["product_name", "category", "storage_type"])
    if n > len(df):
        raise ValueError(f"built-in list only has {len(df)} unique products")
    return df.sample(n=n, random_state=int(rng.integers(1e9))).reset_index(drop=True)


def instacart_products(folder: Path, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Optional hook: sample real product names from the Instacart Market Basket
    Analysis dataset (Kaggle). Needs products.csv and departments.csv in `folder`."""
    products = pd.read_csv(folder / "products.csv")
    departments = pd.read_csv(folder / "departments.csv")
    df = products.merge(departments, on="department_id")
    df = df[~df["department"].isin(["missing", "other", "bulk", "pets", "babies", "alcohol"])]
    df["storage_type"] = df["department"].map(INSTACART_STORAGE).fillna("AMB")
    df = df.sample(n=n, random_state=int(rng.integers(1e9)))
    return (df.rename(columns={"department": "category"})
              [["product_name", "category", "storage_type"]].reset_index(drop=True))


def finish_products(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "sku", [f"SKU{100000 + i}" for i in range(len(df))])
    life, case = [], []
    for st in df["storage_type"]:
        (lo, hi), cases = STORAGE[st]
        life.append(int(rng.integers(lo, hi + 1)))
        case.append(int(rng.choice(cases)))
    df["shelf_life_days"] = life
    df["case_size"] = case
    # hidden true demand (units/day), long-tailed like real grocery demand
    df["true_daily_demand"] = np.round(rng.lognormal(mean=1.2, sigma=0.8, size=len(df)), 2).clip(0.3, 40)
    return df


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

def build_locations(products: pd.DataFrame, rng: np.random.Generator):
    per_aisle = CONFIG["shelves_per_aisle"] * CONFIG["levels_per_shelf"]
    letters = iter(string.ascii_uppercase)
    loc_rows, home = [], {}
    for zone in ["AMB", "CHL", "FRZ"]:
        skus = products.loc[products["storage_type"] == zone, "sku"].tolist()
        n_aisles = max(1, math.ceil(len(skus) * (1 + CONFIG["overflow_share"]) / per_aisle))
        zone_locs = []
        for _ in range(n_aisles):
            aisle = next(letters)
            for shelf in range(1, CONFIG["shelves_per_aisle"] + 1):
                for level in range(1, CONFIG["levels_per_shelf"] + 1):
                    zone_locs.append((f"{aisle}-{shelf:02d}-{level}", zone, aisle, shelf, level))
        order = rng.permutation(len(zone_locs))
        for i, sku in enumerate(skus):
            home[sku] = zone_locs[order[i]][0]
        homes = set(home.values())
        for loc in zone_locs:
            loc_rows.append(loc + ("HOME" if loc[0] in homes else "OVERFLOW",))
    locations = pd.DataFrame(loc_rows, columns=["location_id", "zone", "aisle", "shelf", "level", "location_role"])
    return locations, home


def pick_problem_shelves(locations: pd.DataFrame, rng: np.random.Generator):
    """A few shelf bays with confusing labels where mismatches cluster."""
    problems = set()
    for zone, count in [("AMB", 2), ("CHL", 1)]:
        aisles = locations.loc[locations["zone"] == zone, "aisle"].unique()
        for _ in range(count):
            aisle = rng.choice(aisles)
            start = int(rng.integers(1, CONFIG["shelves_per_aisle"] - 1))
            for s in range(start, start + 3):
                problems.add((aisle, s))
    return problems


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def simulate(products: pd.DataFrame, locations: pd.DataFrame, home: dict, days: int, rng: np.random.Generator):
    cfg = CONFIG
    start = datetime.fromisoformat(cfg["start_date"])
    prod = products.set_index("sku")
    loc_info = locations.set_index("location_id")
    overflow = {z: g["location_id"].tolist()
                for z, g in locations[locations["location_role"] == "OVERFLOW"].groupby("zone")}
    problem_shelves = pick_problem_shelves(locations, rng)

    # stock[sku][(loc, batch)] = qty
    sys_stock = defaultdict(lambda: defaultdict(int))
    phys_stock = defaultdict(lambda: defaultdict(int))
    batch_expiry = {}                       # batch_id -> date
    used_expiry = defaultdict(set)          # sku -> expiry dates used (kept unique per SKU)
    batch_seq = 0

    inbound_rows, split_truth, short_dated_truth = [], [], []
    pick_rows, pick_labels, adjustment_rows = [], [], []

    def new_batch(sku, arrival, short_dated=False):
        nonlocal batch_seq
        batch_seq += 1
        life = prod.at[sku, "shelf_life_days"]
        if short_dated:
            remaining = int(rng.integers(1, 4))
        else:
            remaining = max(2, int(life * rng.uniform(0.6, 1.0)))
        expiry = (arrival + timedelta(days=remaining)).date()
        while expiry in used_expiry[sku]:
            expiry += timedelta(days=1)
        used_expiry[sku].add(expiry)
        bid = f"B{batch_seq:06d}"
        batch_expiry[bid] = expiry
        return bid, expiry

    def putaway(sku, qty, ts, short_dated=False):
        bid, expiry = new_batch(sku, ts, short_dated)
        zone = prod.at[sku, "storage_type"]
        p_split = cfg["p_split"]
        if zone == "CHL":
            p_split *= cfg["chilled_split_multiplier"]
        if ts.hour == 6:
            p_split *= cfg["rush_split_multiplier"]
        targets = [(home[sku], qty)]
        if qty >= 2 and overflow.get(zone) and rng.random() < p_split:
            second = rng.choice(overflow[zone])
            part = int(rng.integers(1, qty))
            targets = [(home[sku], qty - part), (second, part)]
            split_truth.append({"sku": sku, "batch_id": bid, "expiry_date": expiry})
        if short_dated:
            short_dated_truth.append({"sku": sku, "batch_id": bid, "expiry_date": expiry})
        for loc, q in targets:
            sys_stock[sku][(loc, bid)] += q
            phys_stock[sku][(loc, bid)] += q
            inbound_rows.append({
                "inbound_id": f"IN{len(inbound_rows) + 1:06d}", "ts": ts, "sku": sku,
                "batch_id": bid, "expiry_date": expiry, "qty": q, "location_id": loc,
            })

    def round_case(sku, units):
        case = prod.at[sku, "case_size"]
        return int(max(1, math.ceil(units / case)) * case)

    # opening stock: two batches per SKU
    opening_ts = start - timedelta(hours=2)
    for sku in prod.index:
        rate = prod.at[sku, "true_daily_demand"]
        putaway(sku, round_case(sku, rate * 3), opening_ts - timedelta(days=2))
        putaway(sku, round_case(sku, rate * 3), opening_ts)

    def available(stock, sku, today):
        return [(loc, b, q) for (loc, b), q in stock[sku].items() if q > 0 and batch_expiry[b] >= today]

    def neighbour(loc):
        """Adjacent bin on the same shelf bay level: easy to confuse with the target."""
        a, s, lv = loc_info.at[loc, "aisle"], loc_info.at[loc, "shelf"], loc_info.at[loc, "level"]
        options = [f"{a}-{s + d:02d}-{lv}" for d in (-1, 1) if 1 <= s + d <= cfg["shelves_per_aisle"]]
        options += [f"{a}-{s:02d}-{lv + d}" for d in (-1, 1) if 1 <= lv + d <= cfg["levels_per_shelf"]]
        return rng.choice(options)

    hours = np.array(list(HOURLY_WEIGHTS))
    weights = np.array(list(HOURLY_WEIGHTS.values()), dtype=float)
    weights /= weights.sum()
    skus = prod.index.to_numpy()
    rates = prod["true_daily_demand"].to_numpy()
    order_seq = 0

    for d in range(days):
        day = start + timedelta(days=d)
        today = day.date()

        # 05:30 expiry sweep: staff remove expired batches, but miss some
        for sku in skus:
            expired = {b for (_, b) in list(phys_stock[sku]) + list(sys_stock[sku]) if batch_expiry[b] < today}
            for b in sorted(expired):
                if rng.random() >= cfg["p_expiry_sweep"]:
                    continue
                for stock in (phys_stock, sys_stock):
                    for key in [k for k in stock[sku] if k[1] == b]:
                        if stock is sys_stock and stock[sku][key] != 0:
                            adjustment_rows.append({
                                "adjustment_id": f"ADJ{len(adjustment_rows) + 1:06d}",
                                "ts": day + timedelta(hours=5, minutes=30), "sku": sku,
                                "location_id": key[0], "batch_id": b, "qty_change": -stock[sku][key],
                                "reason": "EXPIRED_WRITE_OFF"})
                        del stock[sku][key]

        # morning deliveries 06:00-08:59, ordered on system stock (so phantom stock hides real gaps)
        for sku in skus:
            rate = prod.at[sku, "true_daily_demand"]
            sys_avail = sum(q for _, _, q in available(sys_stock, sku, today))
            if sys_avail < rate * 2 + 1:
                ts = day + timedelta(hours=int(rng.integers(6, 9)), minutes=int(rng.integers(0, 60)))
                putaway(sku, round_case(sku, rate * 4), ts,
                        short_dated=rng.random() < cfg["p_short_dated"])

        # picking: Poisson demand per SKU per day, spread over the hourly profile
        daily_units = rng.poisson(rates)
        events = []
        for sku, units in zip(skus, daily_units):
            while units > 0:
                q = int(min(units, rng.choice([1, 1, 1, 2, 2, 3])))
                h = int(rng.choice(hours, p=weights))
                events.append((day + timedelta(hours=h, minutes=int(rng.integers(0, 60)),
                                               seconds=int(rng.integers(0, 60))), sku, q))
                units -= q
        events.sort()

        for ts, sku, qty in events:
            if rng.random() < 0.3 or order_seq == 0:
                order_seq += 1
            row = {"pick_id": f"PK{len(pick_rows) + 1:07d}", "ts": ts, "order_id": f"ORD{order_seq:07d}",
                   "sku": sku, "qty": qty, "location_shown": "", "batch_shown": "",
                   "location_scanned": "", "location_deducted": "", "batch_deducted": "",
                   "redirect_reason": "", "status": "PICKED"}
            label = {"pick_id": row["pick_id"], "gt_location_mismatch": 0, "gt_fifo_violation": 0,
                     "physical_location": "", "physical_batch": ""}
            peak = ts.hour in PEAK_HOURS

            sys_av = available(sys_stock, sku, today)
            if not sys_av:
                row["status"] = "OUT_OF_STOCK_SYSTEM"
                pick_rows.append(row); pick_labels.append(label)
                continue
            shown_loc, shown_b, _ = min(sys_av, key=lambda x: (batch_expiry[x[1]], x[0]))
            row["location_shown"], row["batch_shown"] = shown_loc, shown_b
            phys_av = available(phys_stock, sku, today)
            phys_at_shown = phys_stock[sku].get((shown_loc, shown_b), 0)

            p_fifo = cfg["p_fifo"] * (cfg["peak_multiplier_fifo"] if peak else 1)
            newer = [x for x in phys_av if batch_expiry[x[1]] > batch_expiry[shown_b]]

            if phys_at_shown > 0 and newer and rng.random() < p_fifo:
                # picker grabs a newer batch (front of shelf / easier reach); system follows the scan
                ploc, pb, pq = newer[int(rng.integers(len(newer)))]
                q = min(qty, pq)
                scanned, dloc, db = ploc, ploc, pb
                label["gt_fifo_violation"] = 1
            elif phys_at_shown > 0:
                ploc, pb = shown_loc, shown_b
                q = min(qty, phys_at_shown)
                scanned, dloc, db = shown_loc, shown_loc, shown_b
                aisle, shelf = loc_info.at[shown_loc, "aisle"], loc_info.at[shown_loc, "shelf"]
                p_mm = cfg["p_mismatch"]
                if peak:
                    p_mm *= cfg["peak_multiplier_mismatch"]
                if (aisle, shelf) in problem_shelves:
                    p_mm *= cfg["problem_shelf_multiplier"]
                if rng.random() < p_mm:
                    others = [(l, b) for (l, b), sq in sys_stock[sku].items() if l != shown_loc and sq > 0]
                    if others and rng.random() < 0.5:
                        dloc, db = others[int(rng.integers(len(others)))]
                    else:
                        dloc, db = neighbour(shown_loc), shown_b
                    label["gt_location_mismatch"] = 1
            else:
                # shown location is physically empty (often phantom stock from an earlier mismatch)
                if not phys_av:
                    row["status"] = "SHORT_PICK"
                    pick_rows.append(row); pick_labels.append(label)
                    continue
                ploc, pb, pq = min(phys_av, key=lambda x: (batch_expiry[x[1]], x[0]))
                q = min(qty, pq)
                scanned, dloc, db = ploc, ploc, pb
                if rng.random() < cfg["p_reason_recorded"]:
                    # reported empty: the WMS writes the phantom stock off with an adjustment
                    row["redirect_reason"] = "LOCATION_EMPTY"
                    phantom = sys_stock[sku][(shown_loc, shown_b)]
                    sys_stock[sku][(shown_loc, shown_b)] = 0
                    adjustment_rows.append({
                        "adjustment_id": f"ADJ{len(adjustment_rows) + 1:06d}", "ts": ts, "sku": sku,
                        "location_id": shown_loc, "batch_id": shown_b, "qty_change": -phantom,
                        "reason": "PICKER_REPORTED_EMPTY"})

            row["qty"] = q
            row["location_scanned"] = "" if rng.random() < cfg["p_scan_missing"] else scanned
            row["location_deducted"], row["batch_deducted"] = dloc, db
            label["physical_location"], label["physical_batch"] = ploc, pb
            phys_stock[sku][(ploc, pb)] -= q
            sys_stock[sku][(dloc, db)] -= q
            pick_rows.append(row); pick_labels.append(label)

    snapshot_ts = start + timedelta(days=days)
    return {
        "inbound": pd.DataFrame(inbound_rows),
        "picks": pd.DataFrame(pick_rows),
        "adjustments": pd.DataFrame(adjustment_rows),
        "pick_labels": pd.DataFrame(pick_labels),
        "split_truth": pd.DataFrame(split_truth),
        "short_dated_truth": pd.DataFrame(short_dated_truth),
        "sys_stock": sys_stock, "phys_stock": phys_stock, "batch_expiry": batch_expiry,
        "snapshot_ts": snapshot_ts, "problem_shelves": problem_shelves,
    }


def expiry_risk_truth(sim, products):
    """Ground truth expiry risk: physical stock that will expire (or has expired)
    before true demand can sell it, within the risk horizon."""
    horizon = CONFIG["expiry_risk_horizon_days"]
    snap = sim["snapshot_ts"].date()
    rate = products.set_index("sku")["true_daily_demand"]
    rows = []
    for sku, entries in sim["phys_stock"].items():
        per_batch = defaultdict(int)
        for (_, b), q in entries.items():
            per_batch[b] += q
        for b, q in per_batch.items():
            days_left = (sim["batch_expiry"][b] - snap).days
            if q > 0 and days_left <= horizon and q > rate[sku] * max(days_left, 0):
                rows.append({"sku": sku, "batch_id": b, "expiry_date": sim["batch_expiry"][b],
                             "physical_qty": q, "days_to_expiry": days_left})
    return pd.DataFrame(rows, columns=["sku", "batch_id", "expiry_date", "physical_qty", "days_to_expiry"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=28)
    ap.add_argument("--skus", type=int, default=CONFIG["n_skus"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--instacart-dir", type=Path, default=None,
                    help="folder with Instacart products.csv and departments.csv (optional)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = (instacart_products(args.instacart_dir, args.skus, rng) if args.instacart_dir
            else builtin_products(args.skus, rng))
    products = finish_products(base, rng)
    locations, home = build_locations(products, rng)
    products["home_location"] = products["sku"].map(home)

    sim = simulate(products, locations, home, args.days, rng)

    out, gt = args.out, args.out / "ground_truth"
    gt.mkdir(parents=True, exist_ok=True)
    products.drop(columns=["true_daily_demand"]).to_csv(out / "products.csv", index=False)
    locations.to_csv(out / "locations.csv", index=False)
    sim["inbound"].to_csv(out / "inbound.csv", index=False)
    sim["picks"].to_csv(out / "picks.csv", index=False)
    sim["adjustments"].to_csv(out / "adjustments.csv", index=False)

    sim["pick_labels"].to_csv(gt / "pick_labels.csv", index=False)
    sim["split_truth"].to_csv(gt / "split_storage_truth.csv", index=False)
    sim["short_dated_truth"].to_csv(gt / "short_dated_deliveries.csv", index=False)
    expiry_risk_truth(sim, products).to_csv(gt / "expiry_risk_truth.csv", index=False)
    products[["sku", "true_daily_demand"]].to_csv(gt / "true_demand.csv", index=False)

    meta = {
        "seed": args.seed, "days": args.days, "skus": args.skus,
        "start_date": CONFIG["start_date"], "snapshot_ts": sim["snapshot_ts"].isoformat(),
        "product_source": "instacart" if args.instacart_dir else "builtin",
        "config": CONFIG, "peak_hours": sorted(PEAK_HOURS),
        "problem_shelves": sorted(f"{a}-{s:02d}" for a, s in sim["problem_shelves"]),
    }
    (gt / "injection_metadata.json").write_text(json.dumps(meta, indent=2))
    public = {k: v for k, v in meta.items() if k in ("seed", "days", "skus", "start_date", "snapshot_ts",
                                                      "product_source")}
    public["config"] = {"expiry_risk_horizon_days": CONFIG["expiry_risk_horizon_days"]}
    (out / "metadata.json").write_text(json.dumps(public, indent=2))

    labels = sim["pick_labels"]
    print(f"products {len(products)}, locations {len(locations)}, "
          f"inbound lines {len(sim['inbound'])}, picks {len(sim['picks'])}")
    print(f"injected: {labels.gt_location_mismatch.sum()} location mismatches, "
          f"{labels.gt_fifo_violation.sum()} FIFO violations, {len(sim['split_truth'])} split batches, "
          f"{len(sim['short_dated_truth'])} short-dated deliveries")
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
