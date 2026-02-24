#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import yaml
import os

trim_data = True
highlight_errors = False
network_index = None  #incompatible with trim_data

if trim_data == True:
    # Leave FILTER_CORE as None to show the full network (default behavior).
    FILTER_CORE: str | None = 'TALDICE'
    # Depth range on FILTER_CORE. Values are in raw depth units from tiepoint files.
    # Either order is accepted; the script will sort to [min, max].
    FILTER_TOP_DEPTH: float | None = 1000.0
    FILTER_BOTTOM_DEPTH: float | None = 1020.0
    # If True, hide depth ranges that have no visible points after filtering.
    CROP_UNUSED_DEPTH_AFTER_FILTER: bool = True
    # If True, each core is independently normalized to the same vertical display range.
    # Hover values remain in raw depth units.
    PER_CORE_DEPTH_SCALE: bool = True
    # If True, highlight all tie segments belonging to networks with within-row errors.
    HIGHLIGHT_ERROR_NETWORKS: bool = highlight_errors
    # Optional: set to an Excel "Index" value to show only ties in that one network.
    NETWORK_INDEX_FILTER: str | None = None
else:
    FILTER_CORE: str | None = None
    FILTER_TOP_DEPTH: float | None = None
    FILTER_BOTTOM_DEPTH: float | None = None
    CROP_UNUSED_DEPTH_AFTER_FILTER: bool = False
    PER_CORE_DEPTH_SCALE: bool = False
    HIGHLIGHT_ERROR_NETWORKS: bool = highlight_errors
    NETWORK_INDEX_FILTER: str | None = network_index
    
# Error-network highlighting settings
ERROR_NETWORK_EXCEL_PATH: str = "table_out/Antarctic_full.xlsx"
ERROR_COLUMN_NAME: str = "Within Row Errors"
NETWORK_MATCH_TOLERANCE_M: float = 0.15
NETWORK_INDEX_DEPTH_PAD_M: float = 1.0

#set working directory
os.chdir(Path(__file__).resolve().parent)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive 3D tiepoint network for all ice-core pairs."
    )
    parser.add_argument(
        "--project",
        default="Antarctic",
        help="Project folder name (default: Antarctic).",
    )
    parser.add_argument(
        "--root",
        default="/Users/quinnmackay/Documents/GitHub/BICC/Antarctic Chronology Accuracy Project",
        help="Repository root path that contains the project folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="table_out",
        help="Directory where the HTML visualization will be written.",
    )
    parser.add_argument(
        "--radius-multiplier",
        type=float,
        default=4.0,
        help="Core separation multiplier. Higher values spread columns farther apart.",
    )
    parser.add_argument(
        "--min-radius",
        type=float,
        default=20.0,
        help="Minimum radius for core layout circle.",
    )
    parser.add_argument(
        "--z-stretch",
        type=float,
        default=3.0,
        help="Depth stretch factor for visualization (raw depth is preserved in hover text).",
    )
    parser.add_argument(
        "--aspect-z",
        type=float,
        default=2.5,
        help="Z axis aspect ratio factor for 3D camera.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Try to open an interactive renderer in addition to writing HTML.",
    )
    parser.add_argument(
        "--filter-core",
        default=FILTER_CORE,
        help="Core to filter by depth range (e.g., EDC). Leave unset to show all tiepoints.",
    )
    parser.add_argument(
        "--filter-top-depth",
        type=float,
        default=FILTER_TOP_DEPTH,
        help="Top depth bound for filter core (raw depth units).",
    )
    parser.add_argument(
        "--filter-bottom-depth",
        type=float,
        default=FILTER_BOTTOM_DEPTH,
        help="Bottom depth bound for filter core (raw depth units).",
    )
    parser.add_argument(
        "--crop-unused-depth",
        action=argparse.BooleanOptionalAction,
        default=CROP_UNUSED_DEPTH_AFTER_FILTER,
        help=(
            "When filtered, crop z-axis to only the visible depth span. "
            "Use --no-crop-unused-depth to keep full-depth range."
        ),
    )
    parser.add_argument(
        "--per-core-depth-scale",
        action=argparse.BooleanOptionalAction,
        default=PER_CORE_DEPTH_SCALE,
        help=(
            "Use separate normalized depth scaling per core so columns are parallel. "
            "Use --no-per-core-depth-scale for global raw-depth scaling."
        ),
    )
    parser.add_argument(
        "--highlight-error-networks",
        action=argparse.BooleanOptionalAction,
        default=HIGHLIGHT_ERROR_NETWORKS,
        help="Highlight tie segments belonging to any network row with Within Row Errors.",
    )
    parser.add_argument(
        "--error-network-excel",
        default=ERROR_NETWORK_EXCEL_PATH,
        help="Excel file with network rows (e.g., table_out/Antarctic_full.xlsx).",
    )
    parser.add_argument(
        "--error-column-name",
        default=ERROR_COLUMN_NAME,
        help="Column name used to flag network errors in the Excel file.",
    )
    parser.add_argument(
        "--network-match-tolerance",
        type=float,
        default=NETWORK_MATCH_TOLERANCE_M,
        help="Depth tolerance (m) used to map network rows back to raw tie segments.",
    )
    parser.add_argument(
        "--network-index",
        default=NETWORK_INDEX_FILTER,
        help='Optional Excel "Index" value. If set, only ties in that network are shown.',
    )
    return parser.parse_args()


def load_tiepoints(root: Path, project: str) -> tuple[list[str], pd.DataFrame]:
    with open(root / project / "parameters.yml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cores = data["list_sites"]
    pairs = [f"{a}-{b}" for a, b in itertools.combinations(cores, 2)]

    records: list[dict[str, object]] = []
    for pair in pairs:
        core_a, core_b = pair.split("-")
        pair_dir = root / project / pair
        if not pair_dir.is_dir():
            continue

        for txt in sorted(pair_dir.glob("*.txt")):
            df = pd.read_csv(txt, sep="\t", comment="#")
            if not {"depth1", "depth2"}.issubset(df.columns):
                continue

            if "reference" in df.columns:
                refs = df["reference"]
            else:
                refs = pd.Series(["unknown"] * len(df))

            for depth1, depth2, ref in zip(df["depth1"], df["depth2"], refs):
                if pd.isna(depth1) or pd.isna(depth2):
                    continue
                records.append(
                    {
                        "pair": pair,
                        "core_a": core_a,
                        "core_b": core_b,
                        "depth_a_raw": float(depth1),
                        "depth_b_raw": float(depth2),
                        "reference": str(ref),
                        "source_file": txt.name,
                    }
                )

    tie_df = pd.DataFrame(records)
    if tie_df.empty:
        raise ValueError("No tiepoints found. Check project path and tiepoint files.")

    return cores, tie_df


def build_figure(
    cores: list[str],
    tie_df: pd.DataFrame,
    project: str,
    radius_multiplier: float,
    min_radius: float,
    z_stretch: float,
    aspect_z: float,
    z_bounds_raw: tuple[float, float],
    per_core_depth_scale: bool,
    parameter_summary_text: str,
) -> go.Figure:
    n_cores = len(cores)
    theta = np.linspace(0.0, 2.0 * np.pi, n_cores, endpoint=False)
    radius = max(min_radius, n_cores * radius_multiplier)
    core_xy = {
        core: (radius * np.cos(angle), radius * np.sin(angle))
        for core, angle in zip(cores, theta)
    }

    tie_df = tie_df.copy()
    z_raw_min, z_raw_max = z_bounds_raw
    z_min = z_raw_min * z_stretch
    z_max = z_raw_max * z_stretch
    z_pad = max(1.0, 0.10 * (z_max - z_min))

    core_depth_ranges: dict[str, tuple[float, float]] = {}
    if per_core_depth_scale:
        for core in cores:
            depth_vals = np.concatenate(
                [
                    tie_df.loc[tie_df["core_a"] == core, "depth_a_raw"].to_numpy(),
                    tie_df.loc[tie_df["core_b"] == core, "depth_b_raw"].to_numpy(),
                ]
            )
            if len(depth_vals) == 0:
                core_depth_ranges[core] = (z_raw_min, z_raw_max)
            else:
                cmin = float(np.nanmin(depth_vals))
                cmax = float(np.nanmax(depth_vals))
                if cmax <= cmin:
                    cmax = cmin + 1e-9
                core_depth_ranges[core] = (cmin, cmax)

    def map_depth_to_plot(core: str, depth_raw: float) -> float:
        if not per_core_depth_scale:
            return depth_raw * z_stretch
        cmin, cmax = core_depth_ranges[core]
        frac = (depth_raw - cmin) / (cmax - cmin)
        frac = min(1.0, max(0.0, frac))
        mapped_raw = z_raw_min + frac * (z_raw_max - z_raw_min)
        return mapped_raw * z_stretch

    fig = go.Figure()

    for core in cores:
        x, y = core_xy[core]
        fig.add_trace(
            go.Scatter3d(
                x=[x, x],
                y=[y, y],
                z=[z_min - z_pad, z_max + z_pad],
                mode="lines+text",
                line=dict(width=12),
                text=["", core],
                textposition="top center",
                name=f"Core {core}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for pair, grp in tie_df.groupby("pair", sort=False):
        xs: list[float | None] = []
        ys: list[float | None] = []
        zs: list[float | None] = []
        hover_text: list[str | None] = []
        err_xs: list[float | None] = []
        err_ys: list[float | None] = []
        err_zs: list[float | None] = []
        err_hover: list[str | None] = []

        for row in grp.itertuples(index=False):
            xa, ya = core_xy[row.core_a]
            xb, yb = core_xy[row.core_b]
            xs.extend([xa, xb, None])
            ys.extend([ya, yb, None])
            depth_a_plot = map_depth_to_plot(row.core_a, row.depth_a_raw)
            depth_b_plot = map_depth_to_plot(row.core_b, row.depth_b_raw)
            zs.extend([depth_a_plot, depth_b_plot, None])

            tip = (
                f"Pair: {row.pair}<br>"
                f"{row.core_a} depth: {row.depth_a_raw:.3f}<br>"
                f"{row.core_b} depth: {row.depth_b_raw:.3f}<br>"
                f"Reference: {row.reference}<br>"
                f"Source: {row.source_file}"
            )
            hover_text.extend([tip, tip, None])
            if bool(getattr(row, "error_network", False)):
                err_xs.extend([xa, xb, None])
                err_ys.extend([ya, yb, None])
                err_zs.extend([depth_a_plot, depth_b_plot, None])
                err_tip = tip + "<br><b>Error network: Yes</b>"
                err_hover.extend([err_tip, err_tip, None])

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=3),
                name=f"{pair} ({len(grp)})",
                hovertext=hover_text,
                hoverinfo="text",
            )
        )
        if err_xs:
            fig.add_trace(
                go.Scatter3d(
                    x=err_xs,
                    y=err_ys,
                    z=err_zs,
                    mode="lines",
                    line=dict(width=7, color="red"),
                    name=f"{pair} (error networks)",
                    hovertext=err_hover,
                    hoverinfo="text",
                )
            )

    z_label_divisor = z_stretch if abs(z_stretch) > 1e-12 else 1.0
    if trim_data is False:
        z_axis_top = 0.0
        z_axis_bottom = max(z_max + z_pad, 0.0)
    else:
        z_axis_top = z_min - z_pad
        z_axis_bottom = z_max + z_pad
    z_tick_vals = np.linspace(z_axis_top, z_axis_bottom, 8)
    z_tick_text = [f"{(tick / z_label_divisor):.0f}" for tick in z_tick_vals]
    zaxis_title_text = (
        "Depth (normalized and relative, no units)"
        if per_core_depth_scale
        else "Depth (raw units, global)"
    )
    zaxis_config: dict[str, object] = {
        "range": [z_axis_bottom, z_axis_top],
        "tickmode": "array",
        "tickvals": z_tick_vals.tolist(),
        "ticktext": z_tick_text,
    }
    if per_core_depth_scale:
        zaxis_config["showticklabels"] = False
        zaxis_config["ticks"] = ""

    fig.update_layout(
        title=(
            f"3D Tiepoint Network: {project} (n={len(tie_df)} tiepoints)"
            f"<br><sup>{parameter_summary_text}</sup>"
        ),
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(showticklabels=False, ticks=""),
            yaxis=dict(showticklabels=False, ticks=""),
            zaxis_title=zaxis_title_text,
            zaxis=zaxis_config,
            aspectratio=dict(x=1.0, y=1.0, z=aspect_z),
            camera=dict(eye=dict(x=2.0, y=2.0, z=1.4)),
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


def filter_tiepoints_for_core_depth(
    tie_df: pd.DataFrame,
    filter_core: str | None,
    top_depth: float | None,
    bottom_depth: float | None,
) -> tuple[pd.DataFrame, str, bool]:
    if not filter_core:
        return tie_df, "No depth filter (full network).", False

    if top_depth is None or bottom_depth is None:
        return tie_df, (
            f"Filter core={filter_core} ignored because one/both bounds are missing; "
            "showing full network."
        ), False

    depth_min = min(top_depth, bottom_depth)
    depth_max = max(top_depth, bottom_depth)

    mask_core_a = (
        (tie_df["core_a"] == filter_core)
        & (tie_df["depth_a_raw"] >= depth_min)
        & (tie_df["depth_a_raw"] <= depth_max)
    )
    mask_core_b = (
        (tie_df["core_b"] == filter_core)
        & (tie_df["depth_b_raw"] >= depth_min)
        & (tie_df["depth_b_raw"] <= depth_max)
    )
    filtered = tie_df[mask_core_a | mask_core_b].copy()

    note = (
        f"Filtered to core={filter_core}, depth range=[{depth_min}, {depth_max}] "
        f"with {len(filtered)} matching tiepoints."
    )
    return filtered, note, True


def _collect_error_networks_from_excel(
    excel_path: Path,
    cores: list[str],
    error_column_name: str,
) -> list[dict[str, list[float]]]:
    if not excel_path.is_file():
        return []

    df = pd.read_excel(excel_path, sheet_name=0)
    if error_column_name not in df.columns:
        return []

    core_columns = [col for col in df.columns if str(col).split(".")[0] in cores]
    if not core_columns:
        return []

    error_rows = df[
        df[error_column_name].notna()
        & (df[error_column_name].astype(str).str.strip() != "")
    ]
    networks: list[dict[str, list[float]]] = []
    for _, row in error_rows.iterrows():
        core_depths: dict[str, list[float]] = {}
        for col in core_columns:
            base_core = str(col).split(".")[0]
            value = row[col]
            if pd.isna(value):
                continue
            core_depths.setdefault(base_core, []).append(float(value))
        if len(core_depths) >= 2:
            networks.append(core_depths)
    return networks


def _match_ties_to_error_networks(
    tie_df: pd.DataFrame,
    error_networks: list[dict[str, list[float]]],
    tolerance_m: float,
) -> pd.Series:
    if not error_networks:
        return pd.Series(False, index=tie_df.index)

    flags = np.zeros(len(tie_df), dtype=bool)
    for i, row in enumerate(tie_df.itertuples(index=False)):
        for network in error_networks:
            a_depths = network.get(row.core_a)
            if not a_depths:
                continue
            b_depths = network.get(row.core_b)
            if not b_depths:
                continue
            a_hit = np.any(np.abs(np.array(a_depths) - row.depth_a_raw) <= tolerance_m)
            if not a_hit:
                continue
            b_hit = np.any(np.abs(np.array(b_depths) - row.depth_b_raw) <= tolerance_m)
            if b_hit:
                flags[i] = True
                break

    return pd.Series(flags, index=tie_df.index)


def _collect_network_by_index_from_excel(
    excel_path: Path,
    cores: list[str],
    network_index: str | None,
) -> dict[str, list[float]] | None:
    if network_index is None:
        return None
    if not excel_path.is_file():
        return None

    df = pd.read_excel(excel_path, sheet_name=0)
    if "Index" not in df.columns:
        return None

    core_columns = [col for col in df.columns if str(col).split(".")[0] in cores]
    if not core_columns:
        return None

    target = str(network_index).strip()
    index_as_str = df["Index"].astype(str).str.strip()
    selected = df[index_as_str == target]
    if selected.empty:
        try:
            target_num = float(target)
            index_num = pd.to_numeric(df["Index"], errors="coerce")
            selected = df[(index_num - target_num).abs() <= 1e-9]
        except Exception:  # noqa: BLE001
            selected = df.iloc[0:0]

    if selected.empty:
        return None

    row = selected.iloc[0]
    core_depths: dict[str, list[float]] = {}
    for col in core_columns:
        base_core = str(col).split(".")[0]
        value = row[col]
        if pd.isna(value):
            continue
        core_depths.setdefault(base_core, []).append(float(value))

    if len(core_depths) < 2:
        return None
    return core_depths


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cores, tie_df = load_tiepoints(root=root, project=args.project)
    full_depth_min = float(
        np.nanmin(np.concatenate([tie_df["depth_a_raw"].to_numpy(), tie_df["depth_b_raw"].to_numpy()]))
    )
    full_depth_max = float(
        np.nanmax(np.concatenate([tie_df["depth_a_raw"].to_numpy(), tie_df["depth_b_raw"].to_numpy()]))
    )

    tie_df_view, filter_note, filter_active = filter_tiepoints_for_core_depth(
        tie_df=tie_df,
        filter_core=args.filter_core,
        top_depth=args.filter_top_depth,
        bottom_depth=args.filter_bottom_depth,
    )
    if tie_df_view.empty:
        raise ValueError(
            "Filter produced zero tiepoints. Adjust filter core/depth bounds or disable filtering."
        )

    excel_path = Path(args.error_network_excel)
    if not excel_path.is_absolute():
        excel_path = root / excel_path

    network_index_note = "Network index: all"
    network_index_depth_window: tuple[float, float] | None = None
    if args.network_index is not None and str(args.network_index).strip() != "":
        selected_network = _collect_network_by_index_from_excel(
            excel_path=excel_path,
            cores=cores,
            network_index=str(args.network_index),
        )
        if selected_network is None:
            raise ValueError(
                f'No valid network found for Index="{args.network_index}" in {excel_path}.'
            )
        selected_mask = _match_ties_to_error_networks(
            tie_df=tie_df_view,
            error_networks=[selected_network],
            tolerance_m=args.network_match_tolerance,
        )
        tie_df_view = tie_df_view[selected_mask].copy()
        if tie_df_view.empty:
            raise ValueError(
                f'Network Index="{args.network_index}" matched no ties in the current filtered view.'
            )
        network_depths = [d for depths in selected_network.values() for d in depths]
        if network_depths:
            depth_low = min(network_depths) - NETWORK_INDEX_DEPTH_PAD_M
            depth_high = max(network_depths) + NETWORK_INDEX_DEPTH_PAD_M
            network_index_depth_window = (depth_low, depth_high)
            network_range_mask = (
                tie_df_view["depth_a_raw"].between(depth_low, depth_high)
                & tie_df_view["depth_b_raw"].between(depth_low, depth_high)
            )
            tie_df_view = tie_df_view[network_range_mask].copy()
            if tie_df_view.empty:
                raise ValueError(
                    f'Network Index="{args.network_index}" has no ties inside padded depth window '
                    f'[{depth_low:.3f}, {depth_high:.3f}].'
                )
        network_index_note = f'Network index: {args.network_index}'

    view_depth_min = float(
        np.nanmin(np.concatenate([tie_df_view["depth_a_raw"].to_numpy(), tie_df_view["depth_b_raw"].to_numpy()]))
    )
    view_depth_max = float(
        np.nanmax(np.concatenate([tie_df_view["depth_a_raw"].to_numpy(), tie_df_view["depth_b_raw"].to_numpy()]))
    )
    if network_index_depth_window is not None:
        z_bounds_raw = network_index_depth_window
    elif filter_active and args.crop_unused_depth:
        z_bounds_raw = (view_depth_min, view_depth_max)
    else:
        z_bounds_raw = (full_depth_min, full_depth_max)

    tie_df_view = tie_df_view.copy()
    tie_df_view["error_network"] = False
    highlighted_count = 0
    error_network_count = 0
    if args.highlight_error_networks:
        error_networks = _collect_error_networks_from_excel(
            excel_path=excel_path,
            cores=cores,
            error_column_name=args.error_column_name,
        )
        error_network_count = len(error_networks)
        if error_networks:
            tie_df_view["error_network"] = _match_ties_to_error_networks(
                tie_df=tie_df_view,
                error_networks=error_networks,
                tolerance_m=args.network_match_tolerance,
            )
            highlighted_count = int(tie_df_view["error_network"].sum())

    filter_core_display = args.filter_core if filter_active else "None (full network)"
    top_display = (
        f"{min(args.filter_top_depth, args.filter_bottom_depth):.3f}"
        if filter_active and args.filter_top_depth is not None and args.filter_bottom_depth is not None
        else "N/A"
    )
    bottom_display = (
        f"{max(args.filter_top_depth, args.filter_bottom_depth):.3f}"
        if filter_active and args.filter_top_depth is not None and args.filter_bottom_depth is not None
        else "N/A"
    )
    parameter_summary_text = (
        f"Filter core: {filter_core_display} | "
        f"Depth top: {top_display} | "
        f"Depth bottom: {bottom_display} | "
        f"Cropped: {args.crop_unused_depth} | "
        f"Per-core depth scales: {args.per_core_depth_scale} | "
        f"Error highlight: {args.highlight_error_networks} | "
        f"{network_index_note}"
    )

    fig = build_figure(
        cores=cores,
        tie_df=tie_df_view,
        project=args.project,
        radius_multiplier=args.radius_multiplier,
        min_radius=args.min_radius,
        z_stretch=args.z_stretch,
        aspect_z=args.aspect_z,
        z_bounds_raw=z_bounds_raw,
        per_core_depth_scale=args.per_core_depth_scale,
        parameter_summary_text=parameter_summary_text,
    )

    out_html = output_dir / f"{args.project}_3d_tiepoint_network.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")

    print(
        "Loaded "
        f"{len(tie_df_view)} tiepoints from {tie_df_view['source_file'].nunique()} files "
        f"across {tie_df_view['pair'].nunique()} core pairs."
    )
    print(f"Wrote interactive 3D plot to: {out_html}")
    print(filter_note)
    print(
        "Display settings: "
        f"radius_multiplier={args.radius_multiplier}, "
        f"min_radius={args.min_radius}, "
        f"z_stretch={args.z_stretch}, "
        f"aspect_z={args.aspect_z}, "
        f"crop_unused_depth={args.crop_unused_depth}, "
        f"per_core_depth_scale={args.per_core_depth_scale}, "
        f"highlight_error_networks={args.highlight_error_networks}, "
        f"network_index={args.network_index}, "
        f"network_index_depth_window={network_index_depth_window}, "
        f"error_networks={error_network_count}, "
        f"highlighted_ties={highlighted_count}"
    )

    if args.show:
        try:
            pio.show(fig)
        except Exception as exc:  # noqa: BLE001
            print(f"Inline renderer unavailable: {exc}")


if __name__ == "__main__":
    main()
