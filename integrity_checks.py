"""
Rule-based inventory integrity checks for the synthetic micro-fulfilment dataset.

Reads the operational tables produced by generate_data.py (it never looks at
ground_truth/ while detecting), raises alerts, computes an inventory
reliability score, and then scores every check against the injected ground
truth: precision, recall and F1.

Checks
  LOCATION_MISMATCH  stock deducted from a different location than the picker
                     scanned; when the scan is missing, deducted != shown with no
                     redirect reason recorded.
  FIFO_VIOLATION     replay of the system ledger: a pick deducted a batch while an
                     older, unexpired batch of the same SKU was in system stock.
                     Picks explained by a recorded redirect or a mismatch are skipped.
  SPLIT_STORAGE      the same SKU and expiry date put away into more than one location.
  EXPIRY_RISK        at snapshot time, stock that expires within the horizon and is
                     more than recent demand can sell before then (expired stock included).
  NEGATIVE_STOCK     the replayed system balance of a location/batch went below zero,
                     the downstream footprint of a mismatch. Reported, not scored.

Usage
  python integrity_checks.py                    # reads data/, writes output/
  python integrity_checks.py --data data --out output --no-eval
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEMAND_WINDOW_DAYS = 14


def load(data: Path):
    t = {name: pd.read_csv(data / f"{name}.csv", keep_default_na=False, na_values=[""])
         for name in ["products", "locations", "inbound", "picks", "adjustments"]}
    for name in ["inbound", "picks", "adjustments"]:
        t[name]["ts"] = pd.to_datetime(t[name]["ts"])
    t["inbound"]["expiry_date"] = pd.to_datetime(t["inbound"]["expiry_date"]).dt.date
    t["meta"] = json.loads((data / "metadata.json").read_text())
    return t


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_location_mismatch(picks: pd.DataFrame) -> pd.DataFrame:
    p = picks[picks["status"] == "PICKED"]
    scanned = p["location_scanned"].notna()
    by_scan = scanned & (p["location_scanned"] != p["location_deducted"])
    by_shown = (~scanned & (p["location_shown"] != p["location_deducted"])
                & p["redirect_reason"].isna())
    flagged = p[by_scan | by_shown].copy()
    flagged["confidence"] = by_scan[by_scan | by_shown].map({True: "high", False: "medium"})
    flagged["detail"] = ("picked at " + flagged["location_scanned"].fillna(flagged["location_shown"])
                         + ", deducted from " + flagged["location_deducted"])
    return flagged


def replay_ledger(t, mismatch_ids: set):
    """Replay inbound, adjustments and picks in time order on system stock.
    Returns FIFO violations, negative-stock events and the final system stock."""
    expiry = dict(zip(t["inbound"]["batch_id"], t["inbound"]["expiry_date"]))
    inbound = t["inbound"].assign(kind=0)
    adjust = t["adjustments"].assign(kind=1)
    picks = t["picks"][t["picks"]["status"] == "PICKED"].assign(kind=2)
    events = pd.concat([
        inbound[["ts", "kind", "sku", "location_id", "batch_id", "qty"]]
        .rename(columns={"location_id": "loc", "batch_id": "batch"}),
        adjust[["ts", "kind", "sku", "location_id", "batch_id", "qty_change"]]
        .rename(columns={"location_id": "loc", "batch_id": "batch", "qty_change": "qty"}),
        picks[["ts", "kind", "sku", "location_deducted", "batch_deducted", "qty", "pick_id", "redirect_reason"]]
        .rename(columns={"location_deducted": "loc", "batch_deducted": "batch"})
        .assign(qty=lambda d: -d["qty"]),
    ]).sort_values(["ts", "kind"], kind="stable")

    stock = defaultdict(lambda: defaultdict(int))     # sku -> (loc, batch) -> qty
    fifo, negative, negative_seen = [], [], set()
    for e in events.itertuples(index=False):
        if e.kind == 2:
            today = e.ts.date()
            available = [expiry[b] for (_, b), q in stock[e.sku].items() if q > 0 and expiry[b] >= today]
            if (available and expiry[e.batch] > min(available)
                    and e.redirect_reason != "LOCATION_EMPTY" and e.pick_id not in mismatch_ids):
                fifo.append({"pick_id": e.pick_id, "ts": e.ts, "sku": e.sku, "location_id": e.loc,
                             "detail": f"took batch expiring {expiry[e.batch]}, "
                                       f"older batch expiring {min(available)} in stock"})
        stock[e.sku][(e.loc, e.batch)] += e.qty
        key = (e.sku, e.loc, e.batch)
        if stock[e.sku][(e.loc, e.batch)] < 0 and key not in negative_seen:
            negative_seen.add(key)
            negative.append({"ts": e.ts, "sku": e.sku, "location_id": e.loc, "batch_id": e.batch,
                             "detail": f"system balance {stock[e.sku][(e.loc, e.batch)]}"})
    return pd.DataFrame(fifo), pd.DataFrame(negative), stock, expiry


def check_split_storage(inbound: pd.DataFrame) -> pd.DataFrame:
    g = (inbound.groupby(["sku", "expiry_date"])
         .agg(locations=("location_id", lambda s: sorted(set(s))), ts=("ts", "min"),
              batch_id=("batch_id", "first"))
         .reset_index())
    g = g[g["locations"].str.len() > 1].copy()
    g["detail"] = "same SKU and expiry in " + g["locations"].str.join(", ")
    return g


def check_expiry_risk(stock, expiry, picks: pd.DataFrame, snapshot: pd.Timestamp, horizon: int):
    recent = picks[(picks["status"] == "PICKED")
                   & (picks["ts"] >= snapshot - pd.Timedelta(days=DEMAND_WINDOW_DAYS))]
    rate = recent.groupby("sku")["qty"].sum() / DEMAND_WINDOW_DAYS
    rows = []
    for sku, entries in stock.items():
        per_batch, locs = defaultdict(int), defaultdict(list)
        for (loc, b), q in entries.items():
            if q > 0:
                per_batch[b] += q
                locs[b].append(loc)
        for b, q in per_batch.items():
            days_left = (expiry[b] - snapshot.date()).days
            sell = rate.get(sku, 0) * max(days_left, 0)
            if days_left <= horizon and q > sell:
                rows.append({"sku": sku, "batch_id": b, "expiry_date": expiry[b], "ts": snapshot,
                             "system_qty": q, "days_to_expiry": days_left, "location_id": ", ".join(locs[b]),
                             "detail": ("EXPIRED, still in stock" if days_left < 0 else
                                        f"{q} units, ~{sell:.1f} sellable in {days_left} days")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reliability score and pattern analysis
# --------------------------------------------------------------------------

def reliability(t, alerts: pd.DataFrame):
    """Daily share of active locations with no integrity alert that day."""
    picks = t["picks"][t["picks"]["status"] == "PICKED"]
    active = (pd.concat([picks[["ts", "location_deducted"]].rename(columns={"location_deducted": "location_id"}),
                         t["inbound"][["ts", "location_id"]]])
              .assign(date=lambda d: d["ts"].dt.date)[["date", "location_id"]].drop_duplicates())
    a = alerts.assign(location_id=alerts["location_id"].str.split(", ")).explode("location_id")
    flagged = a.assign(date=a["ts"].dt.date)[["date", "location_id"]].drop_duplicates()
    zone = t["locations"].set_index("location_id")["zone"]
    active["zone"] = active["location_id"].map(zone)
    active = active.merge(flagged.assign(flagged=1), on=["date", "location_id"], how="left").fillna({"flagged": 0})
    daily = (active.groupby(["date", "zone"])
             .agg(active_locations=("location_id", "count"), flagged_locations=("flagged", "sum"))
             .reset_index())
    daily["reliability_pct"] = 100 * (1 - daily["flagged_locations"] / daily["active_locations"])
    loc_scores = (a.groupby(["location_id", "check"]).size().unstack(fill_value=0)
                  .assign(total_alerts=lambda d: d.sum(axis=1)).sort_values("total_alerts", ascending=False))
    return daily, loc_scores


def mismatch_patterns(picks: pd.DataFrame, mismatches: pd.DataFrame, locations: pd.DataFrame):
    p = picks[picks["status"] == "PICKED"].copy()
    p["flag"] = p["pick_id"].isin(mismatches["pick_id"])
    by_hour = p.groupby(p["ts"].dt.hour)["flag"].mean().mul(100).rename("mismatch_rate_pct")
    loc = locations.set_index("location_id")
    p["bay"] = p["location_shown"].map(loc["aisle"]) + "-" + p["location_shown"].map(loc["shelf"]).map("{:02.0f}".format)
    by_bay = (p.groupby("bay")["flag"].agg(["mean", "size"])
              .rename(columns={"mean": "rate", "size": "picks"})
              .query("picks >= 100").sort_values("rate", ascending=False))
    by_bay["rate"] *= 100
    return by_hour, by_bay


# --------------------------------------------------------------------------
# Evaluation against ground truth
# --------------------------------------------------------------------------

def score(name, predicted: set, actual: set):
    tp, fp, fn = len(predicted & actual), len(predicted - actual), len(actual - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"check": name, "injected": len(actual), "flagged": len(predicted), "true_positives": tp,
            "false_positives": fp, "missed": fn, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3)}


def evaluate(data: Path, mismatches, fifo, splits, risk):
    gt = data / "ground_truth"
    labels = pd.read_csv(gt / "pick_labels.csv")
    split_truth = pd.read_csv(gt / "split_storage_truth.csv")
    risk_truth = pd.read_csv(gt / "expiry_risk_truth.csv")
    actual_mm = set(labels.loc[labels["gt_location_mismatch"] == 1, "pick_id"])
    rows = [
        score("LOCATION_MISMATCH", set(mismatches["pick_id"]), actual_mm),
        score("FIFO_VIOLATION", set(fifo["pick_id"]) if len(fifo) else set(),
              set(labels.loc[labels["gt_fifo_violation"] == 1, "pick_id"])),
        score("SPLIT_STORAGE", set(zip(splits["sku"], splits["batch_id"])),
              set(zip(split_truth["sku"], split_truth["batch_id"]))),
        score("EXPIRY_RISK", set(zip(risk["sku"], risk["batch_id"])) if len(risk) else set(),
              set(zip(risk_truth["sku"], risk_truth["batch_id"]))),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--no-eval", action="store_true", help="skip scoring against ground_truth/")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t = load(args.data)
    snapshot = pd.Timestamp(t["meta"]["snapshot_ts"])
    horizon = t["meta"]["config"]["expiry_risk_horizon_days"]

    mismatches = check_location_mismatch(t["picks"])
    fifo, negative, stock, expiry = replay_ledger(t, set(mismatches["pick_id"]))
    splits = check_split_storage(t["inbound"])
    risk = check_expiry_risk(stock, expiry, t["picks"], snapshot, horizon)

    alerts = pd.concat([
        mismatches.assign(check="LOCATION_MISMATCH", location_id=mismatches["location_shown"] + ", "
                          + mismatches["location_deducted"]),
        fifo.assign(check="FIFO_VIOLATION"),
        splits.assign(check="SPLIT_STORAGE", location_id=splits["locations"].str.join(", ")),
        risk.assign(check="EXPIRY_RISK"),
        negative.assign(check="NEGATIVE_STOCK"),
    ], ignore_index=True)
    alerts = alerts[["ts", "check", "sku", "location_id", "pick_id", "batch_id", "confidence", "detail"]]
    alerts = alerts.sort_values("ts").reset_index(drop=True)
    alerts.to_csv(args.out / "alerts.csv", index=False)

    daily, loc_scores = reliability(t, alerts)
    daily.to_csv(args.out / "daily_reliability.csv", index=False)
    loc_scores.to_csv(args.out / "location_scores.csv")
    by_hour, by_bay = mismatch_patterns(t["picks"], mismatches, t["locations"])
    by_hour.to_csv(args.out / "mismatch_rate_by_hour.csv")
    by_bay.to_csv(args.out / "mismatch_rate_by_shelf_bay.csv")

    print("Alerts raised")
    print(alerts["check"].value_counts().to_string(), "\n")
    last7 = daily[daily["date"] > (snapshot - pd.Timedelta(days=7)).date()]
    overall = 100 * (1 - last7["flagged_locations"].sum() / last7["active_locations"].sum())
    print(f"Inventory reliability, last 7 days: {overall:.1f}% of active locations alert-free")
    zone = last7.groupby("zone")[["active_locations", "flagged_locations"]].sum()
    print((100 * (1 - zone["flagged_locations"] / zone["active_locations"])).round(1).to_string(), "\n")
    print("Mismatch alert rate by hour (%)")
    print(by_hour.round(1).to_string(), "\n")
    print("Shelf bays with the highest mismatch alert rate (%)")
    print(by_bay.head(8).round(1).to_string(), "\n")

    if not args.no_eval and (args.data / "ground_truth").exists():
        ev = evaluate(args.data, mismatches, fifo, splits, risk)
        ev.to_csv(args.out / "evaluation.csv", index=False)
        print("Detection quality vs injected ground truth")
        print(ev.to_string(index=False))
        injected = json.loads((args.data / "ground_truth" / "injection_metadata.json").read_text())
        print(f"\nInjected problem shelves: {', '.join(injected['problem_shelves'])}")
    print(f"\nOutputs written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
