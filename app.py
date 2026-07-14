"""
LinChris Capital — Hotel Flag Feasibility Simulator

Compares staying independent vs. hard-flag (Marriott/Hilton/IHG-tier) vs.
soft-brand (Autograph/Tapestry/Vignette-tier) affiliation for a hotel
property, using STR performance data and adjustable franchise fee,
occupancy-lift, ADR-impact, and PIP/conversion-cost assumptions.

Each flag type carries three variants (pessimistic / base / optimistic)
so the output shows a range of outcomes rather than a single point estimate.
"""

import io
import json
import re

import numpy as np
import pandas as pd
import openpyxl
import altair as alt
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="Flag Feasibility Simulator", layout="wide")

# ===========================================================================
# Defaults — Hotel 1620 Plymouth Harbor, May 2026 STR report
# ===========================================================================
PROPERTY_DEFAULTS = {
    "hotel_name": "Hotel 1620",
    "location": "Plymouth, MA",
    "rooms": 177,
    "occ": 42.37,
    "adr": 181.37,
    "transient_occ": 15.58,
    "group_occ": 26.79,
    "comp_occ": 59.87,
    "comp_adr": 170.07,
}

FLAG_LABELS = {"hard": "Hard flag (Marriott / Hilton / IHG)", "soft": "Soft brand (Autograph / Tapestry / Vignette)"}
VARIANT_ORDER = ["pessimistic", "base", "optimistic"]
VARIANT_LABELS = {"pessimistic": "Underperforming", "base": "Base", "optimistic": "Optimistic"}

# Only the base case is a user-set slider. Optimistic/underperforming are
# derived automatically (see derive_variant_params) so you tune one set of
# assumptions per flag type instead of three.
FLAG_BASE_DEFAULTS = {
    "hard": {"fee": 11.0, "transient_lift": 12.0, "group_lift": 0.0, "adr_impact": -3.0, "pip_per_room": 18000.0},
    "soft": {"fee": 7.0, "transient_lift": 6.0, "group_lift": 0.0, "adr_impact": 1.0, "pip_per_room": 7500.0},
}
BASE_PARAM_ORDER = ["fee", "transient_lift", "group_lift", "adr_impact", "pip_per_room"]

# Fixed spread applied to the base case to derive optimistic/underperforming.
# Lift parameters (which can be positive or negative) swing by a percentage
# of their own magnitude in the favorable/unfavorable direction, so a zero
# base lift stays zero across all three cases. Fee and PIP cost scale
# multiplicatively (they're always positive). ADR impact swings by a fixed
# point spread since it's small and can sit near zero.
LIFT_SPREAD_PCT = 0.5
FEE_MULT = {"optimistic": 0.85, "pessimistic": 1.15}
PIP_MULT = {"optimistic": 0.70, "pessimistic": 1.30}
ADR_SWING_PP = 2.5

AMORT_YEARS_DEFAULT = 7

SCENARIO_COLORS = {
    "stay_independent": "#898781",
    "hard_pessimistic": "#F6C6C2", "hard_base": "#C0453A", "hard_optimistic": "#7A1F17",
    "soft_pessimistic": "#CFE6DE", "soft_base": "#3E8E75", "soft_optimistic": "#0F5C42",
}


def derive_variant_params(base_params: dict, variant: str) -> dict:
    """Base case is user-set; optimistic/underperforming are computed from it."""
    if variant == "base":
        return dict(base_params)
    sign = 1.0 if variant == "optimistic" else -1.0
    fee_mult = FEE_MULT[variant]
    pip_mult = PIP_MULT[variant]
    return {
        "transient_lift": base_params["transient_lift"] + sign * LIFT_SPREAD_PCT * abs(base_params["transient_lift"]),
        "group_lift": base_params["group_lift"] + sign * LIFT_SPREAD_PCT * abs(base_params["group_lift"]),
        "fee": base_params["fee"] * fee_mult,
        "adr_impact": base_params["adr_impact"] + sign * ADR_SWING_PP,
        "pip_per_room": base_params["pip_per_room"] * pip_mult,
    }


# ===========================================================================
# Session state — widget keys double as the persistence schema
# ===========================================================================
ROB_KEYS = ["rob_year", "rob_total_revenue", "rob_room_nights", "rob_group_revenue",
            "rob_transient_revenue", "rob_source_sheet"]

# Performance metrics that STR and BOB can both supply, and the comp-set
# metrics only STR supplies. Each uploader clears its own domain before
# applying a freshly parsed file so a new upload's numbers fully replace
# whatever the previous upload (possibly a different property) left behind,
# instead of blending old and new fields together.
PERFORMANCE_KEYS = ["occ", "adr", "transient_occ", "group_occ"]
COMP_KEYS = ["comp_occ", "comp_adr"]


def _reset_keys(keys, value=0.0):
    for k in keys:
        st.session_state[k] = value


def init_state():
    for k, v in PROPERTY_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("amort_years", AMORT_YEARS_DEFAULT)
    for flag, params in FLAG_BASE_DEFAULTS.items():
        for param, val in params.items():
            st.session_state.setdefault(f"{flag}_{param}", val)
    for k in ROB_KEYS:
        st.session_state.setdefault(k, None)


ALL_KEYS = (
    list(PROPERTY_DEFAULTS.keys())
    + ["amort_years"]
    + [f"{flag}_{param}" for flag, params in FLAG_BASE_DEFAULTS.items() for param in params]
    + ROB_KEYS
)

init_state()


# ===========================================================================
# Multi-hotel profiles
#
# All widget keys in ALL_KEYS live in flat session_state (single active
# hotel's data). Switching hotels snapshots the current widget values into
# hotel_profiles[old_hotel], then loads hotel_profiles[new_hotel] (creating a
# blank one on first use) back into those same widget keys. Streamlit Cloud
# session state doesn't survive across separate visits, so the sidebar
# save/load JSON exports the whole hotel_profiles dict rather than just the
# active hotel, giving a manual (but full-portfolio) persistence path.
# ===========================================================================
def _blank_hotel_profile(name: str) -> dict:
    profile = {k: 0.0 for k in PERFORMANCE_KEYS + COMP_KEYS}
    profile["hotel_name"] = name
    profile["location"] = ""
    profile["rooms"] = PROPERTY_DEFAULTS["rooms"]
    profile["amort_years"] = AMORT_YEARS_DEFAULT
    for flag, params in FLAG_BASE_DEFAULTS.items():
        for param, val in params.items():
            profile[f"{flag}_{param}"] = val
    for k in ROB_KEYS:
        profile[k] = None
    return profile


def _snapshot_current_profile() -> dict:
    return {k: st.session_state[k] for k in ALL_KEYS}


def _load_profile(profile: dict):
    for k in ALL_KEYS:
        st.session_state[k] = profile.get(k, PROPERTY_DEFAULTS.get(k))


def _switch_hotel(new_hotel: str):
    old_hotel = st.session_state["active_hotel"]
    if old_hotel in st.session_state["hotel_profiles"]:
        st.session_state["hotel_profiles"][old_hotel] = _snapshot_current_profile()
    if new_hotel not in st.session_state["hotel_profiles"]:
        st.session_state["hotel_profiles"][new_hotel] = _blank_hotel_profile(new_hotel)
    _load_profile(st.session_state["hotel_profiles"][new_hotel])
    st.session_state["active_hotel"] = new_hotel
    st.session_state["hotel_selector"] = new_hotel


def _on_hotel_selector_change():
    _switch_hotel(st.session_state["hotel_selector"])


def _add_hotel():
    name = st.session_state.get("new_hotel_input", "").strip()
    if not name:
        return
    if name not in st.session_state["hotel_list"]:
        st.session_state["hotel_list"].append(name)
    _switch_hotel(name)
    st.session_state["new_hotel_input"] = ""


st.session_state.setdefault("hotel_list", ["Hotel 1620"])
st.session_state.setdefault("active_hotel", "Hotel 1620")
st.session_state.setdefault("hotel_selector", st.session_state["active_hotel"])
if "hotel_profiles" not in st.session_state:
    st.session_state["hotel_profiles"] = {"Hotel 1620": _snapshot_current_profile()}


# ===========================================================================
# Core calculation engine
# ===========================================================================
def compute_scenario(base_adr, base_transient_occ, base_group_occ, rooms,
                      transient_lift_pp, group_lift_pp, adr_impact_pct,
                      fee_pct, pip_per_room, amort_years):
    """Returns per-scenario revenue, cost, and index metrics.

    Transient and group occupancy each get their own lift, since flag-driven
    demand concentrates in the loyalty/OTA/GDS-driven transient channel while
    group business is relationship-driven — group lift defaults to 0 (flat)
    but is independently adjustable. Total occupancy is the sum of the two
    lifted segments.
    """
    scenario_transient_occ = max(0.0, base_transient_occ + transient_lift_pp)
    scenario_group_occ = max(0.0, base_group_occ + group_lift_pp)
    scenario_occ = min(100.0, scenario_transient_occ + scenario_group_occ)
    scenario_adr = max(0.0, base_adr * (1.0 + adr_impact_pct / 100.0))

    room_nights = rooms * 365
    transient_revenue = room_nights * (scenario_transient_occ / 100.0) * scenario_adr
    group_revenue = room_nights * (scenario_group_occ / 100.0) * scenario_adr
    gross_revenue = transient_revenue + group_revenue

    fee_cost = gross_revenue * fee_pct / 100.0
    pip_total_cost = pip_per_room * rooms
    pip_annual_cost = pip_total_cost / amort_years if amort_years > 0 else 0.0
    net_revenue = gross_revenue - fee_cost - pip_annual_cost
    revpar = (scenario_occ / 100.0) * scenario_adr

    return {
        "occupancy_pct": scenario_occ,
        "transient_occ_pct": scenario_transient_occ,
        "adr": scenario_adr,
        "revpar": revpar,
        "gross_revenue": gross_revenue,
        "transient_revenue": transient_revenue,
        "group_revenue": group_revenue,
        "fee_cost": fee_cost,
        "pip_total_cost": pip_total_cost,
        "pip_annual_cost": pip_annual_cost,
        "net_revenue": net_revenue,
    }


def payback_years(pip_total_cost, annual_incremental_net, cap=40.0):
    if pip_total_cost <= 0:
        return 0.0
    if annual_incremental_net <= 0:
        return None  # never pays back
    return min(cap, pip_total_cost / annual_incremental_net)


def cumulative_benefit(year, annual_incremental_net, pip_total_cost):
    return -pip_total_cost + annual_incremental_net * year


def run_all_scenarios():
    rooms = st.session_state["rooms"]
    base_adr = st.session_state["adr"]
    base_transient_occ = st.session_state["transient_occ"]
    base_group_occ = st.session_state["group_occ"]
    amort_years = st.session_state["amort_years"]

    baseline = compute_scenario(base_adr, base_transient_occ, base_group_occ, rooms,
                                 transient_lift_pp=0.0, group_lift_pp=0.0, adr_impact_pct=0.0,
                                 fee_pct=0.0, pip_per_room=0.0, amort_years=1.0)

    results = {"stay_independent": {**baseline, "label": "Stay independent", "flag": "independent", "variant": "base"}}

    for flag in FLAG_BASE_DEFAULTS:
        base_params = {p: st.session_state[f"{flag}_{p}"] for p in BASE_PARAM_ORDER}
        for variant in VARIANT_ORDER:
            key = f"{flag}_{variant}"
            params = derive_variant_params(base_params, variant)

            r = compute_scenario(base_adr, base_transient_occ, base_group_occ, rooms,
                                  transient_lift_pp=params["transient_lift"], group_lift_pp=params["group_lift"],
                                  adr_impact_pct=params["adr_impact"], fee_pct=params["fee"],
                                  pip_per_room=params["pip_per_room"], amort_years=amort_years)

            annual_incremental = (r["gross_revenue"] - r["fee_cost"]) - baseline["net_revenue"]
            pb = payback_years(r["pip_total_cost"], annual_incremental)

            results[key] = {
                **r,
                "label": f"{FLAG_LABELS[flag].split(' (')[0]} — {VARIANT_LABELS[variant]}",
                "flag": flag,
                "variant": variant,
                "params": params,
                "net_vs_baseline": r["net_revenue"] - baseline["net_revenue"],
                "annual_incremental": annual_incremental,
                "payback_years": pb,
            }

    return results, baseline


# ===========================================================================
# STR report parser
#
# Standard STR "STAR Report" workbooks (as exported for LinChris properties)
# carry a fixed set of named tabs. The "Glance" tab ("Monthly Performance at
# a Glance") and "Segmentation Glance" tab ("Segmentation at a Glance") both
# use a two-level header: a metric-group row (Occupancy (%) / ADR / RevPAR,
# or Transient / Group / Contract / Total) followed by a sub-header row
# (My Prop / Comp Set / Index, or My Property), with data rows labeled
# Current Month / Year To Date / etc. We locate values by walking that
# header structure rather than assuming fixed cell coordinates, since column
# widths shift slightly between properties/exports. A generic label-scan
# heuristic (with plausibility-range guards) is kept as a fallback for CSVs
# and any workbook that doesn't match this tab layout.
# ===========================================================================
LABEL_RULES = [
    (lambda l: "occupancy" in l and "comp" in l, "comp_occ"),
    (lambda l: "adr" in l and "comp" in l, "comp_adr"),
    (lambda l: "transient" in l and "occ" in l, "transient_occ"),
    (lambda l: "group" in l and "occ" in l, "group_occ"),
    (lambda l: l.strip() == "occupancy" or ("occupancy" in l and "trans" not in l and "group" not in l and "comp" not in l), "occ"),
    (lambda l: l.strip() == "adr" or "average daily rate" in l, "adr"),
    (lambda l: "room" in l and ("count" in l or "total" in l), "rooms"),
]

# Plausibility bounds per field — STR reports pack multiple numeric columns
# (current/prior period, % change, rank, comp-set value) after a label, so
# the first number to the right of a match isn't reliably the right one.
# Skipping implausible values (e.g. a bare year like 2024 mistaken for an
# ADR) reduces — but doesn't eliminate — false matches on unfamiliar layouts.
FIELD_RANGES = {
    "comp_occ": (0.0, 100.0),
    "comp_adr": (20.0, 900.0),
    "transient_occ": (0.0, 100.0),
    "group_occ": (0.0, 100.0),
    "occ": (0.0, 100.0),
    "adr": (20.0, 900.0),
    "rooms": (1.0, 3000.0),
}


def _generic_label_scan(arr) -> dict:
    found = {}
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            cell = arr[r, c]
            if not isinstance(cell, str):
                continue
            label = cell.lower().strip()
            for check, field in LABEL_RULES:
                if field in found or not check(label):
                    continue
                lo, hi = FIELD_RANGES[field]
                for cc in range(c + 1, arr.shape[1]):
                    val = arr[r, cc]
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    if lo <= fval <= hi:
                        found[field] = fval
                        break
    return found


def _parse_glance_sheet(rows) -> dict:
    """'Monthly Performance at a Glance' tab: Occupancy (%) / ADR / RevPAR
    group headers, My Prop / Comp Set / Index sub-headers, Current Month row."""
    group_row_idx = next(
        (i for i, row in enumerate(rows)
         if any(isinstance(v, str) and v.strip().lower() == "occupancy (%)" for v in row)),
        None,
    )
    if group_row_idx is None:
        return {}
    group_row = rows[group_row_idx]
    groups = {v.strip().lower(): i for i, v in enumerate(group_row) if isinstance(v, str) and v.strip()}

    sub_row = next(
        (rows[j] for j in range(group_row_idx + 1, min(group_row_idx + 4, len(rows)))
         if any(isinstance(v, str) and v.strip().lower() in ("my prop", "my property") for v in rows[j])),
        None,
    )
    if sub_row is None:
        return {}

    data_row = next(
        (rows[j] for j in range(group_row_idx, min(group_row_idx + 15, len(rows)))
         if any(isinstance(v, str) and v.strip().lower() == "current month" for v in rows[j])),
        None,
    )
    if data_row is None:
        return {}

    def find_val(group_key, subcol_substrs):
        gcol = groups.get(group_key)
        if gcol is None:
            return None
        for k in range(gcol, min(gcol + 6, len(sub_row))):
            sv = sub_row[k]
            if isinstance(sv, str) and any(s in sv.lower() for s in subcol_substrs):
                if k < len(data_row) and isinstance(data_row[k], (int, float)):
                    return float(data_row[k])
        return None

    out = {}
    for field, group_key, subcols in [
        ("occ", "occupancy (%)", ["my prop"]),
        ("comp_occ", "occupancy (%)", ["comp set"]),
        ("adr", "adr", ["my prop"]),
        ("comp_adr", "adr", ["comp set"]),
    ]:
        val = find_val(group_key, subcols)
        if val is not None:
            out[field] = val
    return out


def _parse_segmentation_glance_sheet(rows) -> dict:
    """'Segmentation at a Glance' tab: Transient / Group / Contract / Total
    column groups, with an Occupancy (%) row block carrying My Property
    values inline (segment breakdown has no separate comp-set column)."""
    group_row_idx = next(
        (i for i, row in enumerate(rows)
         if any(isinstance(v, str) and v.strip().lower() == "transient" for v in row)
         and any(isinstance(v, str) and v.strip().lower() == "group" for v in row)),
        None,
    )
    if group_row_idx is None:
        return {}
    group_row = rows[group_row_idx]
    groups = {v.strip().lower(): i for i, v in enumerate(group_row) if isinstance(v, str) and v.strip()}

    occ_row = next(
        (rows[j] for j in range(group_row_idx + 1, min(group_row_idx + 4, len(rows)))
         if any(isinstance(v, str) and v.strip().lower() == "occupancy (%)" for v in rows[j])),
        None,
    )
    if occ_row is None:
        return {}

    def find_val_in_row(row, gcol):
        for k in range(gcol, min(gcol + 4, len(row))):
            if isinstance(row[k], (int, float)):
                return float(row[k])
        return None

    out = {}
    if "transient" in groups:
        val = find_val_in_row(occ_row, groups["transient"])
        if val is not None:
            out["transient_occ"] = val
    if "group" in groups:
        val = find_val_in_row(occ_row, groups["group"])
        if val is not None:
            out["group_occ"] = val
    return out


def _comp_set_metric_block(rows, header_row_idx):
    """Within one metric block (Occupancy (%) / ADR / RevPAR) of a STR Tab 4
    Competitive Set Report, finds the 'Running 12 Month' column group, picks
    its most recent year sub-column, and reads 'My Property' /
    'Competitive Set' values from that column."""
    header_row = rows[header_row_idx]
    r12_col = next(
        (i for i, v in enumerate(header_row)
         if isinstance(v, str) and "running 12 month" in v.strip().lower()),
        None,
    )
    if r12_col is None:
        return None, None

    year_row = rows[header_row_idx + 1] if header_row_idx + 1 < len(rows) else None
    if year_row is None:
        return None, None
    year_cols = [
        (i, int(v)) for i, v in enumerate(year_row)
        if i >= r12_col and isinstance(v, (int, float)) and 2000 <= v <= 2100
    ]
    if not year_cols:
        return None, None
    latest_col = max(year_cols, key=lambda pair: pair[1])[0]

    my_prop_row = next(
        (rows[j] for j in range(header_row_idx + 1, min(header_row_idx + 6, len(rows)))
         if any(isinstance(v, str) and v.strip().lower() == "my property" for v in rows[j])),
        None,
    )
    comp_set_row = next(
        (rows[j] for j in range(header_row_idx + 1, min(header_row_idx + 6, len(rows)))
         if any(isinstance(v, str) and v.strip().lower() == "competitive set" for v in rows[j])),
        None,
    )

    def val(row):
        if row is None or latest_col >= len(row) or not isinstance(row[latest_col], (int, float)):
            return None
        return float(row[latest_col])

    return val(my_prop_row), val(comp_set_row)


def _parse_comp_set_report_sheet(rows) -> dict:
    """STR 'Tab 4 - Competitive Set Report' tab: separate Occupancy (%) / ADR
    / RevPAR blocks, each with My Property + Competitive Set rows and a
    Running 12 Month column group (one sub-column per year — we take the
    most recent). We only need Occupancy and ADR; RevPAR is derivable."""
    out = {}
    for label, my_field, comp_field in [
        ("occupancy (%)", "occ", "comp_occ"),
        ("adr", "adr", "comp_adr"),
    ]:
        header_idx = next(
            (i for i, row in enumerate(rows)
             if any(isinstance(v, str) and v.strip().lower() == label for v in row)),
            None,
        )
        if header_idx is None:
            continue
        my_val, comp_val = _comp_set_metric_block(rows, header_idx)
        if my_val is not None:
            out[my_field] = my_val
        if comp_val is not None:
            out[comp_field] = comp_val
    return out


def parse_str_report(file_bytes: bytes, filename: str) -> dict:
    if filename.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=object)
        except Exception as e:
            return {"_error": str(e)}
        return _generic_label_scan(df.to_numpy())

    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception as e:
        return {"_error": str(e)}

    found = {}
    matched_known_tab = False
    for name in wb.sheetnames:
        lname = name.lower()
        rows = None
        if "segmentation" in lname and "glance" in lname:
            rows = list(wb[name].iter_rows(values_only=True))
            partial = _parse_segmentation_glance_sheet(rows)
        elif "glance" in lname:
            rows = list(wb[name].iter_rows(values_only=True))
            partial = _parse_glance_sheet(rows)
        elif "comp" in lname:
            rows = list(wb[name].iter_rows(values_only=True))
            partial = _parse_comp_set_report_sheet(rows)
        else:
            continue
        if partial:
            matched_known_tab = True
        for k, v in partial.items():
            found.setdefault(k, v)

    if not matched_known_tab:
        # Not a recognized STR "STAR Report" export — fall back to a
        # generic scan of every sheet using the label-proximity heuristic.
        for name in wb.sheetnames:
            arr = np.array(list(wb[name].iter_rows(values_only=True)), dtype=object)
            if arr.size == 0:
                continue
            for k, v in _generic_label_scan(arr).items():
                found.setdefault(k, v)

    return found


# ===========================================================================
# ROB (Revenue on the Books) workbook parser
#
# ROB Master Workbooks carry six week-snapshot tabs ("wk one" .. "wk six")
# for a given reporting month; each is a point-in-time pull, so later tabs
# should show more picked-up revenue than earlier ones. In practice not
# every tab is populated for a given month (e.g. "wk six" may be a stale
# leftover with a different, unrelated layout) — rather than trust tab
# order, we take the tab with the highest trailing-year TOTAL revenue,
# since pickup only accumulates over time.
# ===========================================================================
def parse_rob_workbook(file_bytes: bytes) -> dict:
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    except Exception as e:
        return {"_error": str(e)}

    candidates = []
    for name in wb.sheetnames:
        rows = list(wb[name].iter_rows(values_only=True))
        total_row_idx, year_col_map = None, None
        for i, row in enumerate(rows):
            if row and isinstance(row[0], str) and row[0].strip().upper() == "TOTAL":
                year_cols = {int(v): j for j, v in enumerate(row) if isinstance(v, (int, float)) and 2000 <= v <= 2100}
                if year_cols:
                    total_row_idx, year_col_map = i, year_cols
                    break
        if total_row_idx is None:
            continue

        revenue_row = rows[total_row_idx + 1] if total_row_idx + 1 < len(rows) else None
        roomnights_row = rows[total_row_idx + 2] if total_row_idx + 2 < len(rows) else None
        if not (revenue_row and isinstance(revenue_row[0], str) and revenue_row[0].strip().upper() == "REVENUE"):
            continue

        latest_year = max(year_col_map)
        col = year_col_map[latest_year]
        total_revenue = revenue_row[col] if col < len(revenue_row) else None
        if not isinstance(total_revenue, (int, float)):
            continue
        room_nights = roomnights_row[col] if roomnights_row and col < len(roomnights_row) else None

        group_revenue = None
        for j in range(total_row_idx, min(total_row_idx + 12, len(rows))):
            r = rows[j]
            if r and isinstance(r[0], str) and r[0].strip().upper() == "TOTAL GROUP":
                grp_rev_row = rows[j + 1] if j + 1 < len(rows) else None
                if (grp_rev_row and isinstance(grp_rev_row[0], str) and grp_rev_row[0].strip().upper() == "REVENUE"
                        and col < len(grp_rev_row) and isinstance(grp_rev_row[col], (int, float))):
                    group_revenue = grp_rev_row[col]
                break

        candidates.append({
            "sheet": name, "year": latest_year, "total_revenue": float(total_revenue),
            "room_nights": float(room_nights) if isinstance(room_nights, (int, float)) else None,
            "group_revenue": float(group_revenue) if group_revenue is not None else None,
        })

    if not candidates:
        return {}
    best = max(candidates, key=lambda c: c["total_revenue"])
    out = {
        "rob_year": best["year"],
        "rob_total_revenue": best["total_revenue"],
        "rob_source_sheet": best["sheet"],
    }
    if best["room_nights"] is not None:
        out["rob_room_nights"] = best["room_nights"]
    if best["group_revenue"] is not None:
        out["rob_group_revenue"] = best["group_revenue"]
        out["rob_transient_revenue"] = best["total_revenue"] - best["group_revenue"]
    return out


# ===========================================================================
# BOB (Business on the Books By Date Range) report parser
#
# BOB exports are a single flat CSV: a two-row header (a group-label row
# followed by a sub-column-label row) over daily rows, interspersed with
# per-month subtotal rows and a final grand-total ("TOTALS") row. Column
# order/labels are re-detected from the two header rows on every parse
# rather than assumed by position, since export layouts can add, drop, or
# reorder columns. Segment occupancy splits are derived from the room-count
# mix (group / individual room-nights as a share of total room-nights sold)
# rather than a separately-tracked room inventory, since those room-night
# counts sum exactly to total rooms sold in these exports.
# ===========================================================================
def _bob_header_columns(row0, row1) -> dict:
    """Maps (group_label, sub_label) -> column index. group_label is None
    for single (non-grouped) columns like Date/RMS/OCC%."""
    n = len(row0)
    group_starts = [i for i, v in enumerate(row0) if isinstance(v, str) and v.strip()]
    cols = {}
    for gi, start in enumerate(group_starts):
        end = group_starts[gi + 1] if gi + 1 < len(group_starts) else n
        glabel = row0[start].strip().lower()
        if end - start == 1:
            cols[(None, glabel)] = start
        else:
            for c in range(start, end):
                sub = row1[c] if c < len(row1) else None
                if isinstance(sub, str) and sub.strip():
                    cols[(glabel, sub.strip().lower())] = c
    return cols


def _bob_find(cols, group_substr, sub_substr=None):
    for (glabel, sublabel), idx in cols.items():
        if group_substr is None:
            if glabel is None and sub_substr in sublabel:
                return idx
        elif glabel is not None and group_substr in glabel:
            if sub_substr is None or sub_substr in sublabel:
                return idx
    return None


def parse_bob_report(file_bytes: bytes) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=object)
    except Exception as e:
        return {"_error": str(e)}
    arr = df.to_numpy()
    if arr.shape[0] < 3:
        return {}

    cols = _bob_header_columns(arr[0], arr[1])
    total_col = _bob_find(cols, "total", "rvn")
    occ_col = _bob_find(cols, None, "occ")
    rms_col = _bob_find(cols, None, "rms")
    group_rvn_col = _bob_find(cols, "group", "rvn")
    group_picked_col = _bob_find(cols, "group", "picked up")
    indiv_rvn_col = _bob_find(cols, "individual", "rvn")
    indiv_count_col = _bob_find(cols, "individual", "count")
    if None in (total_col, occ_col, rms_col, group_rvn_col, group_picked_col, indiv_rvn_col, indiv_count_col):
        return {}

    totals_row = next(
        (arr[r] for r in range(arr.shape[0])
         if isinstance(arr[r][0], str) and arr[r][0].strip().upper() == "TOTALS"),
        None,
    )
    if totals_row is None:
        return {}

    def num(row, idx):
        try:
            return float(row[idx]) if idx < len(row) else 0.0
        except (TypeError, ValueError):
            return 0.0

    total_revenue = num(totals_row, total_col)
    occ_pct = num(totals_row, occ_col)
    total_rooms_sold = num(totals_row, rms_col)
    group_revenue = num(totals_row, group_rvn_col)
    group_rooms = num(totals_row, group_picked_col)
    transient_revenue = num(totals_row, indiv_rvn_col)
    transient_rooms = num(totals_row, indiv_count_col)

    out = {
        "occ": occ_pct,
        "rob_total_revenue": total_revenue,
        "rob_group_revenue": group_revenue,
        "rob_transient_revenue": transient_revenue,
        "rob_room_nights": total_rooms_sold,
        "rob_source_sheet": "BOB report",
    }
    if total_rooms_sold:
        out["adr"] = total_revenue / total_rooms_sold
        out["group_occ"] = occ_pct * (group_rooms / total_rooms_sold)
        out["transient_occ"] = occ_pct * (transient_rooms / total_rooms_sold)

    years = sorted({
        int(m.group(1))
        for r in range(arr.shape[0])
        if isinstance(arr[r][0], str)
        for m in [re.match(r"^[A-Za-z]{3}\s+(\d{4})$", arr[r][0].strip())]
        if m
    })
    if years:
        out["rob_year"] = years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}"

    return out


# ===========================================================================
# Soft-flag breakeven calculation
#
# Holding group revenue and ADR flat, how much would transient revenue need
# to grow for a flag conversion to break even at a given fee rate? The fee
# applies to *total* revenue (including the group revenue you already have),
# so growing transient revenue also raises the fee bill on that group piece:
#   breakeven ΔT = (fee × current_gross_revenue + annualized_PIP) / (1 - fee)
# ===========================================================================
def compute_breakeven(current_gross_revenue, current_transient_revenue, fee_pct, pip_per_room, rooms, amort_years):
    fee = fee_pct / 100.0
    pip_annual = (pip_per_room * rooms) / amort_years if amort_years > 0 else 0.0
    delta_t = (fee * current_gross_revenue + pip_annual) / (1 - fee) if fee < 1 else float("inf")
    pct_increase = (delta_t / current_transient_revenue * 100.0) if current_transient_revenue > 0 else None
    return {
        "pip_annual": pip_annual,
        "delta_t": delta_t,
        "pct_increase": pct_increase,
        "breakeven_transient_revenue": current_transient_revenue + delta_t,
    }


def build_summary_pdf(hotel_name, location, rooms, source_label, sections, verdict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Flag Feasibility Summary", styles["Title"]),
        Paragraph(f"{hotel_name} — {location} ({rooms} rooms)", styles["Normal"]),
        Paragraph(f"Revenue source: {source_label}", styles["Normal"]),
        Spacer(1, 16),
    ]
    for title, rows in sections:
        elements.append(Paragraph(title, styles["Heading2"]))
        table = Table(rows, colWidths=[3.2 * inch, 2.6 * inch])
        table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 14))
    elements.append(Paragraph(verdict, styles["Normal"]))
    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# UI — header
# ===========================================================================
st.markdown(
    """
    <div style="padding:0.5rem 0 1.5rem;border-bottom:1px solid #2A3D63;margin-bottom:1.5rem;">
      <div style="color:#C9A84C;font-size:12px;letter-spacing:0.08em;text-transform:uppercase;font-weight:600;">
        Revenue strategy tool
      </div>
      <div style="font-size:30px;font-weight:600;margin-top:4px;">Flag feasibility simulator</div>
      <div style="color:#B9C2D4;font-size:14px;margin-top:4px;max-width:720px;">
        Compare staying independent against a hard flag or soft brand affiliation, across
        pessimistic / base / optimistic cases, using your property's actual STR performance data.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Hotel")
    st.selectbox("Active hotel", options=st.session_state["hotel_list"], key="hotel_selector",
                 on_change=_on_hotel_selector_change)
    c1, c2 = st.columns([3, 1])
    c1.text_input("Add new hotel", key="new_hotel_input", placeholder="e.g. Inn at Middletown",
                  label_visibility="collapsed")
    c2.button("Add", on_click=_add_hotel, width="stretch")

    st.divider()
    st.markdown("### Save / load scenario")
    # Snapshot the live widget values into the active hotel's profile before
    # exporting, so the download reflects any unsaved edits made this session.
    st.session_state["hotel_profiles"][st.session_state["active_hotel"]] = _snapshot_current_profile()
    export_data = {
        "hotel_list": st.session_state["hotel_list"],
        "active_hotel": st.session_state["active_hotel"],
        "hotel_profiles": st.session_state["hotel_profiles"],
    }
    st.download_button(
        "Download all hotels (.json)",
        data=json.dumps(export_data, indent=2),
        file_name="linchris_flag_scenarios.json",
        mime="application/json",
        width="stretch",
    )
    uploaded_scenario = st.file_uploader("Load scenario (.json)", type=["json"], key="scenario_uploader")
    if uploaded_scenario is not None:
        try:
            loaded = json.loads(uploaded_scenario.read())
            if "hotel_profiles" in loaded:
                st.session_state["hotel_list"] = loaded.get("hotel_list", list(loaded["hotel_profiles"].keys()))
                st.session_state["hotel_profiles"] = loaded["hotel_profiles"]
                _switch_hotel(loaded.get("active_hotel", st.session_state["hotel_list"][0]))
                st.success("Scenario loaded.")
                st.rerun()
            else:
                # Legacy single-hotel export — apply to the current active hotel.
                changed = False
                for k, v in loaded.items():
                    if k in ALL_KEYS and st.session_state.get(k) != v:
                        st.session_state[k] = v
                        changed = True
                if changed:
                    st.success("Scenario loaded.")
                    st.rerun()
        except Exception as e:
            st.error(f"Couldn't read that file: {e}")

tab_property, tab_scenarios, tab_results, tab_sensitivity, tab_summary = st.tabs(
    ["1 · Property & comp set", "2 · Flag scenarios", "3 · Results", "4 · Sensitivity", "5 · Summary"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Property & comp set
# ---------------------------------------------------------------------------
with tab_property:
    st.markdown("Upload an STR report to auto-fill what we can find, or edit the fields directly below. Every field stays editable — STR layouts vary property to property.")

    uploaded_str = st.file_uploader("Upload STR report (.xlsx / .csv)", type=["xlsx", "xls", "csv"], key="str_uploader")
    if uploaded_str is not None:
        found = parse_str_report(uploaded_str.read(), uploaded_str.name)
        if "_error" in found:
            st.error(f"Couldn't read that file: {found['_error']}")
        else:
            _reset_keys(PERFORMANCE_KEYS + COMP_KEYS)
            field_map = {"occ": "occ", "adr": "adr", "transient_occ": "transient_occ",
                         "group_occ": "group_occ", "comp_occ": "comp_occ", "comp_adr": "comp_adr",
                         "rooms": "rooms"}
            matched = 0
            for f, key in field_map.items():
                if f in found:
                    st.session_state[key] = int(found[f]) if key == "rooms" else found[f]
                    matched += 1
            if matched:
                st.success(f"Cleared previous property metrics and auto-filled {matched} field(s) from "
                           f"{uploaded_str.name} — fields not found in this file were reset to 0; please "
                           f"verify against the source.")
            else:
                st.warning(f"Couldn't auto-detect fields in {uploaded_str.name}. Previous property metrics "
                           f"were cleared — enter values manually below.")

    st.markdown("**Actuals from ROB (optional — powers the Summary tab)**")
    uploaded_rob = st.file_uploader("Upload ROB Master Workbook (.xlsx)", type=["xlsx", "xls"], key="rob_uploader")
    if uploaded_rob is not None:
        rob_found = parse_rob_workbook(uploaded_rob.read())
        if "_error" in rob_found:
            st.error(f"Couldn't read that file: {rob_found['_error']}")
        else:
            _reset_keys(ROB_KEYS, value=None)
            if rob_found:
                for k, v in rob_found.items():
                    st.session_state[k] = v
                st.success(f"Cleared previous ROB actuals and pulled trailing {rob_found.get('rob_year', '')} "
                           f"totals from {uploaded_rob.name} (sheet '{rob_found.get('rob_source_sheet', '')}') — "
                           f"${rob_found.get('rob_total_revenue', 0):,.0f} total revenue.")
            else:
                st.warning("Couldn't find a recognizable TOTAL revenue section in that file. Previous ROB "
                           "actuals were cleared.")

    st.markdown("**Actuals from BOB (Business on the Books) — optional, full-year totals**")
    uploaded_bob = st.file_uploader("Upload BOB report (.csv)", type=["csv"], key="bob_uploader")
    if uploaded_bob is not None:
        bob_found = parse_bob_report(uploaded_bob.read())
        if "_error" in bob_found:
            st.error(f"Couldn't read that file: {bob_found['_error']}")
        else:
            _reset_keys(PERFORMANCE_KEYS)
            _reset_keys(ROB_KEYS, value=None)
            if bob_found:
                for f in PERFORMANCE_KEYS:
                    if f in bob_found:
                        st.session_state[f] = bob_found[f]
                for k in ROB_KEYS:
                    if k in bob_found:
                        st.session_state[k] = bob_found[k]
                st.success(f"Cleared previous performance/revenue metrics and pulled "
                           f"{bob_found.get('rob_year', '')} totals from {uploaded_bob.name} — "
                           f"${bob_found.get('rob_total_revenue', 0):,.0f} total revenue, "
                           f"{bob_found.get('occ', 0):.1f}% occupancy.")
            else:
                st.warning("Couldn't find a recognizable TOTALS row in that BOB file. Previous performance/"
                           "revenue metrics were cleared.")

    c1, c2 = st.columns(2)
    c1.text_input("Hotel name", key="hotel_name")
    c2.text_input("Location", key="location")
    st.slider("Room count", min_value=1, max_value=3000, step=1, key="rooms")

    st.markdown("**Current performance**")
    c1, c2 = st.columns(2)
    c1.slider("Occupancy (%)", min_value=0.0, max_value=100.0, step=0.1, key="occ")
    c2.slider("ADR ($)", min_value=0.0, max_value=900.0, step=1.0, key="adr")
    c1.slider("Transient occ. (%)", min_value=0.0, max_value=100.0, step=0.1, key="transient_occ")
    c2.slider("Group occ. (%)", min_value=0.0, max_value=100.0, step=0.1, key="group_occ")

    st.markdown("**Comp set**")
    c1, c2 = st.columns(2)
    c1.slider("Comp set occupancy (%)", min_value=0.0, max_value=100.0, step=0.1, key="comp_occ")
    c2.slider("Comp set ADR ($)", min_value=0.0, max_value=900.0, step=1.0, key="comp_adr")

    st.slider("Amortize PIP over (years)", min_value=1, max_value=20, step=1, key="amort_years")

# ---------------------------------------------------------------------------
# Tab 2 — Flag scenarios
# ---------------------------------------------------------------------------
with tab_scenarios:
    st.markdown("Set the **base case** assumptions for each flag type — optimistic and underperforming cases are derived automatically from these (wider lift / lower fee & PIP for optimistic, narrower lift / higher fee & PIP for underperforming), so you only tune one set of numbers per flag.")

    for flag in ["hard", "soft"]:
        st.markdown(f"#### {FLAG_LABELS[flag]}")
        c1, c2 = st.columns(2)
        c1.slider("Franchise + royalty fee (%)", min_value=0.0, max_value=25.0, step=0.5, key=f"{flag}_fee")
        c2.slider("PIP cost ($/room)", min_value=0.0, max_value=100000.0, step=500.0, key=f"{flag}_pip_per_room")
        c1.slider("Expected transient occupancy lift (pp)", min_value=-10.0, max_value=30.0, step=0.5,
                  key=f"{flag}_transient_lift")
        c2.slider("Expected group occupancy lift (pp)", min_value=-10.0, max_value=30.0, step=0.5,
                  key=f"{flag}_group_lift")
        st.slider("ADR impact (%)", min_value=-20.0, max_value=20.0, step=0.5, key=f"{flag}_adr_impact")

        base_params = {p: st.session_state[f"{flag}_{p}"] for p in BASE_PARAM_ORDER}
        opt = derive_variant_params(base_params, "optimistic")
        pess = derive_variant_params(base_params, "pessimistic")
        with st.expander(f"Derived optimistic / underperforming assumptions"):
            rows = ["Franchise fee", "Transient lift", "Group lift", "ADR impact", "PIP cost/room"]

            def _fmt(p):
                return [f"{p['fee']:.1f}%", f"{p['transient_lift']:.1f}pp", f"{p['group_lift']:.1f}pp",
                        f"{p['adr_impact']:.1f}%", f"${p['pip_per_room']:,.0f}"]

            derived_df = pd.DataFrame({"Underperforming": _fmt(pess), "Base": _fmt(base_params), "Optimistic": _fmt(opt)},
                                       index=rows)
            st.dataframe(derived_df, use_container_width=True)
        st.divider()

# ---------------------------------------------------------------------------
# Run model
# ---------------------------------------------------------------------------
results, baseline = run_all_scenarios()
scenario_keys = ["stay_independent"] + [f"{flag}_{variant}" for flag in ["hard", "soft"] for variant in VARIANT_ORDER]

# ---------------------------------------------------------------------------
# Tab 3 — Results
# ---------------------------------------------------------------------------
with tab_results:
    best_key = max(scenario_keys, key=lambda k: results[k]["net_revenue"])
    best = results[best_key]

    if best_key == "stay_independent":
        verdict = (f"At these inputs, staying independent nets ${best['net_revenue']:,.0f}/yr — more than any flag "
                   f"option after fees and PIP costs. No scenario pays for itself unless the lift or fee assumptions change.")
    else:
        verdict = (f"**{best['label']}** nets ${best['net_revenue']:,.0f}/yr, ${best['net_vs_baseline']:,.0f} above "
                   f"staying independent — the strongest option at these inputs.")
        if best["payback_years"] is not None:
            verdict += f" PIP cost pays back in {best['payback_years']:.1f} years."
        else:
            verdict += " PIP cost never pays back under these assumptions."

    st.markdown(
        f"""<div style="background:#1C2D4E;border:1px solid #2A3D63;border-radius:10px;padding:1rem 1.25rem;margin-bottom:1.25rem;">
        {verdict}</div>""",
        unsafe_allow_html=True,
    )

    # Scenario comparison table
    rows = []
    for k in scenario_keys:
        r = results[k]
        rows.append({
            "Scenario": r["label"],
            "Occupancy": f"{r['occupancy_pct']:.1f}%",
            "ADR": f"${r['adr']:,.0f}",
            "Gross revenue": f"${r['gross_revenue']:,.0f}",
            "Fee cost": f"${r['fee_cost']:,.0f}",
            "PIP (annualized)": f"${r['pip_annual_cost']:,.0f}",
            "Net revenue": f"${r['net_revenue']:,.0f}",
            "Net vs. independent": "—" if k == "stay_independent" else f"${r['net_vs_baseline']:,.0f}",
            "Payback (yrs)": "—" if k == "stay_independent" else (
                f"{r['payback_years']:.1f}" if r["payback_years"] is not None else "never"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    labels_order = [results[k]["label"] for k in scenario_keys]
    color_domain = [results[k]["label"] for k in scenario_keys]
    color_range = [SCENARIO_COLORS[k] for k in scenario_keys]

    col1, col2 = st.columns(2)

    with col1:
        df_net = pd.DataFrame({
            "Scenario": labels_order,
            "Net revenue": [results[k]["net_revenue"] for k in scenario_keys],
        })
        bars = alt.Chart(df_net).mark_bar().encode(
            x=alt.X("Scenario:N", sort=labels_order, title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Net revenue:Q", title="Net revenue ($)"),
            color=alt.Color("Scenario:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
            tooltip=["Scenario", alt.Tooltip("Net revenue:Q", format="$,.0f")],
        )
        rule = alt.Chart(pd.DataFrame({"y": [baseline["net_revenue"]]})).mark_rule(
            color="#C9A84C", strokeDash=[5, 4], size=2
        ).encode(y="y:Q")
        st.altair_chart((bars + rule).properties(height=420, title="Annual net revenue by scenario"),
                         width="stretch")

    with col2:
        df_stack = pd.DataFrame({
            "Scenario": labels_order * 2,
            "Segment": ["Transient"] * len(scenario_keys) + ["Group"] * len(scenario_keys),
            "Revenue": [results[k]["transient_revenue"] for k in scenario_keys]
                       + [results[k]["group_revenue"] for k in scenario_keys],
        })
        stacked = alt.Chart(df_stack).mark_bar().encode(
            x=alt.X("Scenario:N", sort=labels_order, title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("Revenue:Q", title="Gross revenue ($)", stack="zero"),
            color=alt.Color("Segment:N", scale=alt.Scale(domain=["Transient", "Group"],
                                                           range=["#4C8DC9", "#C9A84C"])),
            order=alt.Order("Segment:N"),
            tooltip=["Scenario", "Segment", alt.Tooltip("Revenue:Q", format="$,.0f")],
        )
        st.altair_chart(stacked.properties(height=420, title="Transient vs. group revenue by scenario"),
                         width="stretch")

    # Payback curve
    years = np.arange(0, 16)
    records = []
    for k in scenario_keys:
        if k == "stay_independent":
            continue
        r = results[k]
        for y in years:
            records.append({
                "Year": y,
                "Cumulative net benefit": cumulative_benefit(y, r["annual_incremental"], r["pip_total_cost"]),
                "Scenario": r["label"],
            })
    df_cum = pd.DataFrame(records)
    non_indep_domain = [results[k]["label"] for k in scenario_keys if k != "stay_independent"]
    non_indep_range = [SCENARIO_COLORS[k] for k in scenario_keys if k != "stay_independent"]
    line = alt.Chart(df_cum).mark_line(point=True).encode(
        x=alt.X("Year:Q", title="Years since conversion"),
        y=alt.Y("Cumulative net benefit:Q", title="Cumulative net benefit ($)"),
        color=alt.Color("Scenario:N", scale=alt.Scale(domain=non_indep_domain, range=non_indep_range),
                         title="Scenario"),
        tooltip=["Scenario", "Year", alt.Tooltip("Cumulative net benefit:Q", format="$,.0f")],
    )
    zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#B9C2D4", size=1.5).encode(y="y:Q")
    st.altair_chart((line + zero_rule).properties(
        height=460, title="Cumulative net benefit over 15 years vs. staying independent"),
        width="stretch")

    st.caption("Flag-driven occupancy lift is applied to total occupancy for revenue, and separately to the transient segment for the transient/group split — group business is relationship-driven, not brand-driven, so it's held flat. PIP cost is amortized straight-line over the years set in tab 1.")

# ---------------------------------------------------------------------------
# Tab 4 — Sensitivity
# ---------------------------------------------------------------------------
with tab_sensitivity:
    st.markdown("How sensitive is the net-vs-baseline outcome to the franchise fee and occupancy lift assumptions? Grid holds ADR impact and PIP cost at the selected variant's values.")

    c1, c2 = st.columns(2)
    sens_flag = c1.selectbox("Flag type", ["hard", "soft"], format_func=lambda f: FLAG_LABELS[f], key="sens_flag")
    sens_variant = c2.selectbox("Hold ADR/PIP at", VARIANT_ORDER, index=1,
                                 format_func=lambda v: VARIANT_LABELS[v], key="sens_variant")

    base_params = {p: st.session_state[f"{sens_flag}_{p}"] for p in BASE_PARAM_ORDER}
    held_params = derive_variant_params(base_params, sens_variant)
    adr_impact = held_params["adr_impact"]
    pip_per_room = held_params["pip_per_room"]
    total_lift = held_params["transient_lift"] + held_params["group_lift"]
    transient_ratio = (held_params["transient_lift"] / total_lift) if total_lift else 0.5

    fee_range = np.arange(4.0, 16.5, 0.5)
    lift_range = np.arange(0.0, 22.0, 1.0)

    rooms = st.session_state["rooms"]
    base_adr = st.session_state["adr"]
    base_transient_occ = st.session_state["transient_occ"]
    base_group_occ = st.session_state["group_occ"]
    amort_years = st.session_state["amort_years"]

    grid = np.zeros((len(lift_range), len(fee_range)))
    for i, lift in enumerate(lift_range):
        for j, fee in enumerate(fee_range):
            r = compute_scenario(base_adr, base_transient_occ, base_group_occ, rooms,
                                  transient_lift_pp=lift * transient_ratio, group_lift_pp=lift * (1 - transient_ratio),
                                  adr_impact_pct=adr_impact, fee_pct=fee,
                                  pip_per_room=pip_per_room, amort_years=amort_years)
            grid[i, j] = r["net_revenue"] - baseline["net_revenue"]

    vmax = max(abs(grid.min()), abs(grid.max()), 1.0)
    grid_records = [
        {"Fee (%)": fee, "Occupancy lift (pp)": lift, "Net vs baseline": grid[i, j]}
        for i, lift in enumerate(lift_range)
        for j, fee in enumerate(fee_range)
    ]
    df_grid = pd.DataFrame(grid_records)
    heatmap = alt.Chart(df_grid).mark_rect().encode(
        x=alt.X("Fee (%):O", title="Franchise fee rate (%)"),
        y=alt.Y("Occupancy lift (pp):O", title="Total occupancy lift (pp)", sort=list(lift_range)),
        color=alt.Color("Net vs baseline:Q",
                         scale=alt.Scale(scheme="redyellowgreen", domain=[-vmax, 0, vmax]),
                         title="Net vs baseline ($)"),
        tooltip=["Fee (%)", "Occupancy lift (pp)", alt.Tooltip("Net vs baseline:Q", format="$,.0f")],
    ).properties(height=520, title=f"Net revenue vs. baseline — fee × occupancy lift ({FLAG_LABELS[sens_flag]})")
    st.altair_chart(heatmap, width="stretch")
    st.caption("Green = flag beats staying independent at that fee/lift combination; red = it doesn't. The break-even line runs roughly diagonally — higher fees require proportionally more occupancy lift to pay for themselves.")

# ---------------------------------------------------------------------------
# Tab 5 — Summary
# ---------------------------------------------------------------------------
with tab_summary:
    st.markdown("A simple one-page snapshot: hard actuals, the soft-flag breakeven threshold, and how the current soft-flag base-case assumptions stack up against it.")

    rooms = st.session_state["rooms"]
    room_nights = rooms * 365
    amort_years = st.session_state["amort_years"]

    if st.session_state.get("rob_total_revenue"):
        source_label = f"ROB actuals (trailing {st.session_state.get('rob_year', '')}, sheet '{st.session_state.get('rob_source_sheet', '')}')"
        current_gross = st.session_state["rob_total_revenue"]
        current_transient = st.session_state.get("rob_transient_revenue")
        current_group = st.session_state.get("rob_group_revenue")
        if current_transient is None:
            current_transient = room_nights * (st.session_state["transient_occ"] / 100.0) * st.session_state["adr"]
            current_group = current_gross - current_transient
    else:
        source_label = "STR-derived (current month occupancy × ADR, annualized — upload a ROB workbook in tab 1 for full-year actuals)"
        current_gross = room_nights * (st.session_state["occ"] / 100.0) * st.session_state["adr"]
        current_transient = room_nights * (st.session_state["transient_occ"] / 100.0) * st.session_state["adr"]
        current_group = current_gross - current_transient

    st.caption(f"Revenue source: {source_label}")

    soft_fee = st.session_state["soft_fee"]
    soft_pip = st.session_state["soft_pip_per_room"]
    be = compute_breakeven(current_gross, current_transient, soft_fee, soft_pip, rooms, amort_years)
    occ_pp_equiv = be["delta_t"] / (room_nights * st.session_state["adr"]) * 100.0 if st.session_state["adr"] > 0 else None
    current_transient_occ = st.session_state["transient_occ"]
    target_transient_occ = (current_transient_occ + occ_pp_equiv) if occ_pp_equiv is not None else None
    occ_pct_increase = (occ_pp_equiv / current_transient_occ * 100.0) if (
        occ_pp_equiv is not None and current_transient_occ > 0) else None

    soft_base = results["soft_base"]
    baseline_r = results["stay_independent"]
    transient_ratio = (soft_base["transient_revenue"] / baseline_r["transient_revenue"]
                        if baseline_r["transient_revenue"] else 1.0)
    group_ratio = (soft_base["group_revenue"] / baseline_r["group_revenue"]
                   if baseline_r["group_revenue"] else 1.0)
    projected_transient = current_transient * transient_ratio
    projected_group = current_group * group_ratio
    projected_gross = projected_transient + projected_group

    soft_params = soft_base["params"]
    projected_fee_cost = projected_gross * soft_params["fee"] / 100.0
    projected_pip_annual = (soft_params["pip_per_room"] * rooms) / amort_years if amort_years > 0 else 0.0
    projected_net = projected_gross - projected_fee_cost - projected_pip_annual
    projected_net_vs_current = projected_net - current_gross
    projected_delta_t = projected_transient - current_transient
    gap = projected_delta_t - be["delta_t"]

    if gap >= 0:
        verdict = (f"Soft flag base-case assumptions project a transient revenue increase of ${projected_delta_t:,.0f}, "
                   f"which clears the ${be['delta_t']:,.0f} breakeven bar by ${gap:,.0f}.")
    else:
        verdict = (f"Soft flag base-case assumptions project a transient revenue increase of ${projected_delta_t:,.0f}, "
                   f"which falls short of the ${be['delta_t']:,.0f} breakeven bar by ${abs(gap):,.0f}.")

    st.markdown(
        f"""<div style="background:#1C2D4E;border:1px solid #2A3D63;border-radius:10px;padding:1rem 1.25rem;margin:0.5rem 0 1.25rem;">
        {verdict}</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Current performance (actuals)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Annual gross revenue", f"${current_gross:,.0f}")
    c2.metric("Annual transient revenue", f"${current_transient:,.0f}")
    c3.metric("Annual group revenue", f"${current_group:,.0f}")

    st.markdown(f"#### Soft flag breakeven (at {soft_fee:.1f}% fee, ${soft_pip:,.0f}/room PIP)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Annualized PIP cost", f"${be['pip_annual']:,.0f}")
    c2.metric("Transient revenue increase needed", f"${be['delta_t']:,.0f}",
              f"+{be['pct_increase']:.1f}%" if be["pct_increase"] is not None else None)
    if target_transient_occ is not None:
        occ_delta = f"+{occ_pp_equiv:.1f}pp" + (f" ({occ_pct_increase:.1f}%)" if occ_pct_increase is not None else "")
        c3.metric("Transient occupancy needed", f"{target_transient_occ:.1f}%", occ_delta)
        relative_clause = f", a {occ_pct_increase:.1f}% relative increase" if occ_pct_increase is not None else ""
        st.caption(f"Current transient occupancy is {current_transient_occ:.1f}% — breakeven needs it to reach "
                   f"{target_transient_occ:.1f}% (+{occ_pp_equiv:.1f} percentage points{relative_clause}).")
    else:
        c3.metric("Transient occupancy needed", "—")

    st.markdown("#### Projected outcome (soft flag base case)")
    c1, c2, c3 = st.columns(3)
    projected_transient_pct = (projected_delta_t / current_transient * 100.0) if current_transient > 0 else None
    projected_delta_label = f"+${projected_delta_t:,.0f}" + (
        f" ({projected_transient_pct:.1f}%)" if projected_transient_pct is not None else "")
    c1.metric("Projected transient revenue", f"${projected_transient:,.0f}", projected_delta_label)
    c2.metric("Projected gross revenue", f"${projected_gross:,.0f}")
    c3.metric("Projected net vs. current", f"${projected_net_vs_current:,.0f}")

    pdf_sections = [
        ("Current performance (actuals)", [
            ["Annual gross revenue", f"${current_gross:,.0f}"],
            ["Annual transient revenue", f"${current_transient:,.0f}"],
            ["Annual group revenue", f"${current_group:,.0f}"],
        ]),
        (f"Soft flag breakeven ({soft_fee:.1f}% fee, ${soft_pip:,.0f}/room PIP)", [
            ["Annualized PIP cost", f"${be['pip_annual']:,.0f}"],
            ["Transient revenue increase needed", f"${be['delta_t']:,.0f}"],
            ["% increase over current transient revenue",
             f"{be['pct_increase']:.1f}%" if be["pct_increase"] is not None else "—"],
            ["Current transient occupancy", f"{current_transient_occ:.1f}%"],
            ["Transient occupancy needed", f"{target_transient_occ:.1f}%" if target_transient_occ is not None else "—"],
            ["Occupancy increase needed", (
                f"+{occ_pp_equiv:.1f}pp" + (f" ({occ_pct_increase:.1f}%)" if occ_pct_increase is not None else "")
            ) if occ_pp_equiv is not None else "—"],
        ]),
        ("Projected outcome (soft flag base case)", [
            ["Projected transient revenue", f"${projected_transient:,.0f}"],
            ["Projected transient revenue increase", f"+${projected_delta_t:,.0f}" + (
                f" ({projected_transient_pct:.1f}%)" if projected_transient_pct is not None else "")],
            ["Projected gross revenue", f"${projected_gross:,.0f}"],
            ["Projected net vs. current", f"${projected_net_vs_current:,.0f}"],
        ]),
    ]
    pdf_bytes = build_summary_pdf(st.session_state["hotel_name"], st.session_state["location"], rooms,
                                   source_label, pdf_sections, verdict)
    st.download_button(
        "Download summary as PDF",
        data=pdf_bytes,
        file_name=f"{st.session_state['hotel_name'].replace(' ', '_')}_flag_summary.pdf",
        mime="application/pdf",
    )
