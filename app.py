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

import numpy as np
import pandas as pd
import openpyxl
import altair as alt
import streamlit as st

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
VARIANT_LABELS = {"pessimistic": "Pessimistic", "base": "Base", "optimistic": "Optimistic"}

FLAG_DEFAULTS = {
    "hard": {
        "pessimistic": {"fee": 13.0, "occ_lift": 4.0, "trans_lift": 6.0, "adr_impact": -6.0, "pip_per_room": 24000.0},
        "base":        {"fee": 11.0, "occ_lift": 8.0, "trans_lift": 12.0, "adr_impact": -3.0, "pip_per_room": 18000.0},
        "optimistic":  {"fee": 9.0, "occ_lift": 14.0, "trans_lift": 18.0, "adr_impact": -1.0, "pip_per_room": 12000.0},
    },
    "soft": {
        "pessimistic": {"fee": 9.0, "occ_lift": 2.0, "trans_lift": 3.0, "adr_impact": 0.0, "pip_per_room": 9500.0},
        "base":        {"fee": 7.0, "occ_lift": 4.0, "trans_lift": 6.0, "adr_impact": 1.0, "pip_per_room": 7500.0},
        "optimistic":  {"fee": 5.0, "occ_lift": 6.0, "trans_lift": 9.0, "adr_impact": 3.0, "pip_per_room": 5500.0},
    },
}

AMORT_YEARS_DEFAULT = 7

SCENARIO_COLORS = {
    "stay_independent": "#898781",
    "hard_pessimistic": "#F6C6C2", "hard_base": "#C0453A", "hard_optimistic": "#7A1F17",
    "soft_pessimistic": "#CFE6DE", "soft_base": "#3E8E75", "soft_optimistic": "#0F5C42",
}


# ===========================================================================
# Session state — widget keys double as the persistence schema
# ===========================================================================
def init_state():
    for k, v in PROPERTY_DEFAULTS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("amort_years", AMORT_YEARS_DEFAULT)
    for flag, variants in FLAG_DEFAULTS.items():
        for variant, params in variants.items():
            for param, val in params.items():
                st.session_state.setdefault(f"{flag}_{variant}_{param}", val)


ALL_KEYS = (
    list(PROPERTY_DEFAULTS.keys())
    + ["amort_years"]
    + [f"{flag}_{variant}_{param}"
       for flag, variants in FLAG_DEFAULTS.items()
       for variant, params in variants.items()
       for param in params]
)

init_state()


# ===========================================================================
# Core calculation engine
# ===========================================================================
def compute_scenario(base_occ, base_adr, base_transient_occ, rooms,
                      occ_lift_pp, transient_lift_pp, adr_impact_pct,
                      fee_pct, pip_per_room, amort_years):
    """Returns per-scenario revenue, cost, and index metrics.

    Total occupancy lift drives gross revenue directly. Transient lift is
    modeled separately and used to split gross revenue into transient vs.
    group (group = residual), since flag-driven demand concentrates in the
    loyalty/OTA/GDS-driven transient channel — group business is
    relationship-driven, not brand-driven, so it's held flat.
    """
    scenario_occ = max(0.0, min(100.0, base_occ + occ_lift_pp))
    scenario_transient_occ = max(0.0, base_transient_occ + transient_lift_pp)
    scenario_adr = max(0.0, base_adr * (1.0 + adr_impact_pct / 100.0))

    room_nights = rooms * 365
    gross_revenue = room_nights * (scenario_occ / 100.0) * scenario_adr
    transient_revenue = room_nights * (scenario_transient_occ / 100.0) * scenario_adr
    transient_revenue = min(transient_revenue, gross_revenue)
    group_revenue = gross_revenue - transient_revenue

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
    base_occ = st.session_state["occ"]
    base_adr = st.session_state["adr"]
    base_transient_occ = st.session_state["transient_occ"]
    amort_years = st.session_state["amort_years"]

    baseline = compute_scenario(base_occ, base_adr, base_transient_occ, rooms,
                                 occ_lift_pp=0.0, transient_lift_pp=0.0, adr_impact_pct=0.0,
                                 fee_pct=0.0, pip_per_room=0.0, amort_years=1.0)

    results = {"stay_independent": {**baseline, "label": "Stay independent", "flag": "independent", "variant": "base"}}

    for flag, variants in FLAG_DEFAULTS.items():
        for variant in VARIANT_ORDER:
            key = f"{flag}_{variant}"
            fee = st.session_state[f"{key}_fee"]
            occ_lift = st.session_state[f"{key}_occ_lift"]
            trans_lift = st.session_state[f"{key}_trans_lift"]
            adr_impact = st.session_state[f"{key}_adr_impact"]
            pip_per_room = st.session_state[f"{key}_pip_per_room"]

            r = compute_scenario(base_occ, base_adr, base_transient_occ, rooms,
                                  occ_lift_pp=occ_lift, transient_lift_pp=trans_lift,
                                  adr_impact_pct=adr_impact, fee_pct=fee,
                                  pip_per_room=pip_per_room, amort_years=amort_years)

            annual_incremental = (r["gross_revenue"] - r["fee_cost"]) - baseline["net_revenue"]
            pb = payback_years(r["pip_total_cost"], annual_incremental)

            results[key] = {
                **r,
                "label": f"{FLAG_LABELS[flag].split(' (')[0]} — {VARIANT_LABELS[variant]}",
                "flag": flag,
                "variant": variant,
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
    st.markdown("### Save / load scenario")
    export_data = {k: st.session_state[k] for k in ALL_KEYS}
    st.download_button(
        "Download scenario (.json)",
        data=json.dumps(export_data, indent=2),
        file_name=f"{st.session_state['hotel_name'].replace(' ', '_')}_scenario.json",
        mime="application/json",
        width="stretch",
    )
    uploaded_scenario = st.file_uploader("Load scenario (.json)", type=["json"], key="scenario_uploader")
    if uploaded_scenario is not None:
        try:
            loaded = json.loads(uploaded_scenario.read())
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

tab_property, tab_scenarios, tab_results, tab_sensitivity = st.tabs(
    ["1 · Property & comp set", "2 · Flag scenarios", "3 · Results", "4 · Sensitivity"]
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
        elif found:
            field_map = {"occ": "occ", "adr": "adr", "transient_occ": "transient_occ",
                         "group_occ": "group_occ", "comp_occ": "comp_occ", "comp_adr": "comp_adr",
                         "rooms": "rooms"}
            matched = 0
            for f, key in field_map.items():
                if f in found:
                    st.session_state[key] = int(found[f]) if key == "rooms" else found[f]
                    matched += 1
            if matched:
                st.success(f"Auto-filled {matched} field(s) from {uploaded_str.name} — please verify against the source.")
            else:
                st.warning(f"Couldn't auto-detect fields in {uploaded_str.name}. Enter values manually below.")
        else:
            st.warning("No recognizable STR fields found. Enter values manually below.")

    c1, c2, c3 = st.columns(3)
    c1.text_input("Hotel name", key="hotel_name")
    c2.text_input("Location", key="location")
    c3.number_input("Room count", min_value=1, step=1, key="rooms")

    st.markdown("**Current performance**")
    c1, c2, c3, c4 = st.columns(4)
    c1.number_input("Occupancy (%)", min_value=0.0, max_value=100.0, step=0.1, key="occ")
    c2.number_input("ADR ($)", min_value=0.0, step=1.0, key="adr")
    c3.number_input("Transient occ. (%)", min_value=0.0, max_value=100.0, step=0.1, key="transient_occ")
    c4.number_input("Group occ. (%)", min_value=0.0, max_value=100.0, step=0.1, key="group_occ")

    st.markdown("**Comp set**")
    c1, c2 = st.columns(2)
    c1.number_input("Comp set occupancy (%)", min_value=0.0, max_value=100.0, step=0.1, key="comp_occ")
    c2.number_input("Comp set ADR ($)", min_value=0.0, step=1.0, key="comp_adr")

    st.number_input("Amortize PIP over (years)", min_value=1, max_value=20, step=1, key="amort_years")

# ---------------------------------------------------------------------------
# Tab 2 — Flag scenarios
# ---------------------------------------------------------------------------
with tab_scenarios:
    st.markdown("Assumptions for each flag type, split into pessimistic / base / optimistic cases. Adjust to match what you're actually being quoted — PIP cost is the line item that moves this model most.")

    for flag in ["hard", "soft"]:
        st.markdown(f"#### {FLAG_LABELS[flag]}")
        cols = st.columns(3)
        for col, variant in zip(cols, VARIANT_ORDER):
            with col:
                st.markdown(f"**{VARIANT_LABELS[variant]}**")
                key = f"{flag}_{variant}"
                st.number_input("Franchise + royalty fee (%)", min_value=0.0, max_value=25.0, step=0.5,
                                 key=f"{key}_fee")
                st.number_input("Total occupancy lift (pp)", min_value=-10.0, max_value=30.0, step=0.5,
                                 key=f"{key}_occ_lift")
                st.number_input("Transient occupancy lift (pp)", min_value=-10.0, max_value=30.0, step=0.5,
                                 key=f"{key}_trans_lift")
                st.number_input("ADR impact (%)", min_value=-20.0, max_value=20.0, step=0.5,
                                 key=f"{key}_adr_impact")
                st.number_input("PIP cost ($/room)", min_value=0.0, step=500.0,
                                 key=f"{key}_pip_per_room")
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

    base_params = FLAG_DEFAULTS[sens_flag][sens_variant]
    hold_key = f"{sens_flag}_{sens_variant}"
    adr_impact = st.session_state[f"{hold_key}_adr_impact"]
    pip_per_room = st.session_state[f"{hold_key}_pip_per_room"]
    trans_lift_ratio = (st.session_state[f"{hold_key}_trans_lift"] /
                         st.session_state[f"{hold_key}_occ_lift"]) if st.session_state[f"{hold_key}_occ_lift"] else 1.0

    fee_range = np.arange(4.0, 16.5, 0.5)
    lift_range = np.arange(0.0, 22.0, 1.0)

    rooms = st.session_state["rooms"]
    base_occ = st.session_state["occ"]
    base_adr = st.session_state["adr"]
    base_transient_occ = st.session_state["transient_occ"]
    amort_years = st.session_state["amort_years"]

    grid = np.zeros((len(lift_range), len(fee_range)))
    for i, lift in enumerate(lift_range):
        for j, fee in enumerate(fee_range):
            r = compute_scenario(base_occ, base_adr, base_transient_occ, rooms,
                                  occ_lift_pp=lift, transient_lift_pp=lift * trans_lift_ratio,
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
