from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.widgets import Button, TextBox


CORE_PAIR: tuple[str, str] = ("wdc", "edml")

SHOW_VERTICAL_TIE_LINES = True
MAX_DRAWN_TIEPOINTS = 20
MAX_VISIBLE_LAYERS = 50
VERTICAL_OFFSET_STEP = 0.14
MIN_VERTICAL_PAD_FRACTION = 0.04
AUTO_PAD_FRACTION_STEP = 0.12
DEFAULT_Y_RANGE_SCALE = 1.0
MIN_Y_RANGE_SCALE = 0.2
LAYER_SPAN_HALF_WIDTH = 0.005

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_SULFATE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PLOT_SULFATE_DIR.parent
BIG_TABLE_PATH = PROJECT_DIR / "all_tiepoints" / "big_table.csv"

# Source files requested by the user from the legacy layer-count scripts.
HOLCENE_REVISION_WORK_DIR = Path("/Users/quinnmackay/Documents/GitHub/BICC/Holcene Revision Work")

EDML_LC_DIR = HOLCENE_REVISION_WORK_DIR / "EDML LC"
EDML_DATA_DIR = EDML_LC_DIR / "Data Files"
EDML_LAYER_COUNT_FILE = EDML_DATA_DIR / "EDML Layer Count.xlsx"
EDML_CFA_FILES = [
	EDML_DATA_DIR / "EDML_CFA_113-1443.499m_1mm_resolution.txt",
	EDML_DATA_DIR / "EDML_CFA_1443.5-2774m_1mm_resolution.txt",
]
EDML_ECM_FILE = EDML_DATA_DIR / "EDML ECM.txt"

WDC_LC_DIR = HOLCENE_REVISION_WORK_DIR / "WD LC"
WDC_DATA_DIR = WDC_LC_DIR / "Data Files"
WDC_RAW_LAYER_COUNT_FILE = WDC_DATA_DIR / "Original WD2014 Layer Count.tab"
WDC_SHALLOW_ION_FILE = WDC_DATA_DIR / "Sigl2015_SOM4_Antarctica.xlsx"
WDC_SHALLOW_DEP_FILE = WDC_DATA_DIR / "DRI Ions.txt"
WDC_BRITTLE_CHEM_FILE = WDC_DATA_DIR / "WDC06A 577-1300 m Chemistry.xlsx"
WDC_DEP_BRITTLE_DIR = WDC_LC_DIR / "DEP files brittle ice"

CORE_ALIASES = {
	"edml": "EDML",
	"wdc": "WDC",
}


@dataclass
class SignalSeries:
	core_key: str
	canonical_name: str
	source_label: str
	data: pd.DataFrame
	value_label: str


@dataclass
class LayerCountState:
	edml: pd.DataFrame
	wdc_raw: pd.DataFrame


Tiepoint = tuple[float, float, str, str]


def canonical_core_name(core: str) -> str:
	key = core.strip().lower()
	if key not in CORE_ALIASES:
		supported = ", ".join(sorted(CORE_ALIASES))
		raise ValueError(f"Unsupported core '{core}'. Supported keys: {supported}")
	return CORE_ALIASES[key]


def normalize_depth_value(
	depth_values: pd.Series,
	data_values: pd.Series,
) -> pd.DataFrame:
	frame = pd.DataFrame(
		{
			"depth": pd.to_numeric(depth_values, errors="coerce"),
			"value": pd.to_numeric(data_values, errors="coerce"),
		}
	).dropna()
	return frame.sort_values("depth").reset_index(drop=True)


def build_signal(
	core_key: str,
	source_label: str,
	depth_values: pd.Series,
	data_values: pd.Series,
	value_label: str,
) -> SignalSeries:
	return SignalSeries(
		core_key=core_key,
		canonical_name=canonical_core_name(core_key),
		source_label=source_label,
		data=normalize_depth_value(depth_values, data_values),
		value_label=value_label,
	)


def load_edml_signals() -> list[SignalSeries]:
	frames: list[pd.DataFrame] = []
	for path in EDML_CFA_FILES:
		if not path.exists():
			raise FileNotFoundError(f"Missing EDML CFA source: {path}")
		frame = pd.read_csv(
			path,
			header=None,
			skiprows=1,
			sep="\t",
			names=[
				"Depth(m)",
				"Na(ng/g)",
				"NH4(ng/g)",
				"Ca(ng/g)",
				"Dust(particles/ml)",
				"Cond(mikroS/cm)",
			],
		)
		frames.append(frame)

	edml_cfa = pd.concat(frames, ignore_index=True)
	if not EDML_ECM_FILE.exists():
		raise FileNotFoundError(f"Missing EDML ECM source: {EDML_ECM_FILE}")
	edml_ecm = pd.read_csv(EDML_ECM_FILE, skiprows=102, sep="\t")

	cfa_source = "+".join(path.name for path in EDML_CFA_FILES)

	signals = [
		build_signal(
			"edml",
			f"{cfa_source} [NH4(ng/g)]",
			edml_cfa["Depth(m)"],
			edml_cfa["NH4(ng/g)"],
			"NH4(ng/g)",
		),
		build_signal(
			"edml",
			f"{cfa_source} [Ca(ng/g)]",
			edml_cfa["Depth(m)"],
			edml_cfa["Ca(ng/g)"],
			"Ca(ng/g)",
		),
		build_signal(
			"edml",
			f"{cfa_source} [Cond(mikroS/cm)]",
			edml_cfa["Depth(m)"],
			edml_cfa["Cond(mikroS/cm)"],
			"Cond(mikroS/cm)",
		),
		build_signal(
			"edml",
			f"{EDML_ECM_FILE.name} [ec]",
			edml_ecm["depth_m"],
			edml_ecm["ec"],
			"ECM",
		),
	]
	return signals


def parse_dep_file(path: Path, tabular_case: bool) -> pd.DataFrame:
	start_line: int | None = None
	with path.open("r", errors="ignore") as handle:
		for idx, line in enumerate(handle):
			if "END HEADER" in line:
				start_line = idx + (1 if tabular_case else 2)
				break

	if start_line is None:
		raise ValueError(f"Could not find END HEADER marker in DEP file: {path}")

	separator = r"\s+" if tabular_case else ","
	return pd.read_csv(
		path,
		skiprows=start_line,
		header=None,
		sep=separator,
		names=["Depth(m)", "Conductance(uS)"],
	)


def load_wdc_dep_brittle() -> pd.DataFrame:
	tab_files = ["0550RA.D50", "0550RB.D50"]
	comma_files = [f"{depth:04d}S.d50" for depth in range(600, 1301, 50)]

	frames: list[pd.DataFrame] = []
	for name in tab_files:
		path = WDC_DEP_BRITTLE_DIR / name
		if path.exists():
			frames.append(parse_dep_file(path, tabular_case=True))

	for name in comma_files:
		path = WDC_DEP_BRITTLE_DIR / name
		if path.exists():
			frames.append(parse_dep_file(path, tabular_case=False))

	if not frames:
		raise FileNotFoundError(
			f"No expected brittle DEP files found in {WDC_DEP_BRITTLE_DIR}. "
			"Expected files include 0550RA.D50, 0550RB.D50 and 0600S.d50..1300S.d50."
		)

	return pd.concat(frames, ignore_index=True)


def load_wdc_signals() -> list[SignalSeries]:
	if not WDC_SHALLOW_ION_FILE.exists():
		raise FileNotFoundError(f"Missing WDC shallow chemistry source: {WDC_SHALLOW_ION_FILE}")
	shallow_ion = pd.read_excel(
		WDC_SHALLOW_ION_FILE,
		header=None,
		sheet_name="1 - WDC06A_layer_count",
		skiprows=1,
		names=[
			"Depth_m",
			"Depth_mweq",
			"Decimal_Year_CE",
			"BC",
			"Na",
			"Sr",
			"nssS",
			"nssS_Na_ratio",
			"Br",
			"nh4",
			"nssCa",
		],
	)
	for col in ["BC", "Na", "Sr", "nssS", "nssS_Na_ratio", "Br", "nh4", "nssCa"]:
		shallow_ion[col] = shallow_ion[col].mask(shallow_ion[col] < -3)

	if not WDC_SHALLOW_DEP_FILE.exists():
		raise FileNotFoundError(f"Missing WDC shallow DEP source: {WDC_SHALLOW_DEP_FILE}")
	shallow_dep = pd.read_csv(WDC_SHALLOW_DEP_FILE, sep="\t")
	shallow_dep["Cond(uS)"] = pd.to_numeric(shallow_dep["Cond(uS)"], errors="coerce")
	shallow_dep["Cond(uS)"] = shallow_dep["Cond(uS)"].mask(shallow_dep["Cond(uS)"] < 0)

	if not WDC_BRITTLE_CHEM_FILE.exists():
		raise FileNotFoundError(f"Missing WDC brittle chemistry source: {WDC_BRITTLE_CHEM_FILE}")
	brittle_chem = pd.read_excel(
		WDC_BRITTLE_CHEM_FILE,
		header=None,
		skiprows=5,
		names=[
			"Depth(m)",
			"Cl(ng/g)",
			"NO3(ng/g)",
			"SO4(ng/g)",
			"Na(ng/g)",
			"K(ng/g)",
			"Mg(ng/g)",
			"Ca(ng/g)",
		],
	)
	brittle_chem["SO4(ng/g)"] = brittle_chem["SO4(ng/g)"].clip(upper=70)

	brittle_dep = load_wdc_dep_brittle()

	signals = [
		build_signal(
			"wdc",
			f"{WDC_SHALLOW_ION_FILE.name} [nssS]",
			shallow_ion["Depth_m"],
			shallow_ion["nssS"],
			"nssS",
		),
		build_signal(
			"wdc",
			f"{WDC_SHALLOW_ION_FILE.name} [Na]",
			shallow_ion["Depth_m"],
			shallow_ion["Na"],
			"Na",
		),
		build_signal(
			"wdc",
			f"{WDC_SHALLOW_DEP_FILE.name} [Cond(uS)]",
			shallow_dep["Depth(m)"],
			shallow_dep["Cond(uS)"],
			"Cond(uS)",
		),
		build_signal(
			"wdc",
			"DEP files brittle ice [Conductance(uS)]",
			brittle_dep["Depth(m)"],
			brittle_dep["Conductance(uS)"],
			"DEP Conductance(uS)",
		),
		build_signal(
			"wdc",
			f"{WDC_BRITTLE_CHEM_FILE.name} [SO4(ng/g)]",
			brittle_chem["Depth(m)"],
			brittle_chem["SO4(ng/g)"],
			"SO4(ng/g)",
		),
		build_signal(
			"wdc",
			f"{WDC_BRITTLE_CHEM_FILE.name} [Na(ng/g)]",
			brittle_chem["Depth(m)"],
			brittle_chem["Na(ng/g)"],
			"Na(ng/g)",
		),
		build_signal(
			"wdc",
			f"{WDC_BRITTLE_CHEM_FILE.name} [NO3(ng/g)]",
			brittle_chem["Depth(m)"],
			brittle_chem["NO3(ng/g)"],
			"NO3(ng/g)",
		),
	]
	return signals


def load_layer_count_state() -> LayerCountState:
	if not EDML_LAYER_COUNT_FILE.exists():
		raise FileNotFoundError(f"Missing EDML layer-count source: {EDML_LAYER_COUNT_FILE}")
	edml_layer_count = pd.read_excel(
		EDML_LAYER_COUNT_FILE,
		sheet_name=1,
		skiprows=2,
		header=None,
		usecols=[0, 1, 2, 3],
		names=["depth", "count", "year_b2k", "mce"],
	)
	edml_layer_count["depth"] = pd.to_numeric(edml_layer_count["depth"], errors="coerce")
	edml_layer_count["count"] = pd.to_numeric(edml_layer_count["count"], errors="coerce")
	edml_layer_count["year_b2k"] = pd.to_numeric(edml_layer_count["year_b2k"], errors="coerce")
	edml_layer_count = edml_layer_count.dropna(subset=["depth", "count"])

	if not WDC_RAW_LAYER_COUNT_FILE.exists():
		raise FileNotFoundError(f"Missing WDC raw layer-count source: {WDC_RAW_LAYER_COUNT_FILE}")
	wdc_raw = pd.read_csv(
		WDC_RAW_LAYER_COUNT_FILE,
		comment="#",
		delimiter="\t",
		names=[
			"depth",
			"age_ka",
			"age_err_layer",
			"age_err_ch4",
			"gas_age",
			"gas_age_err",
			"delta_age",
			"delta_age_err",
		],
	)
	wdc_raw["depth"] = pd.to_numeric(wdc_raw["depth"], errors="coerce")
	wdc_raw["age_yr_bp1950"] = pd.to_numeric(wdc_raw["age_ka"], errors="coerce") * 1000
	wdc_raw = wdc_raw.dropna(subset=["depth"])

	return LayerCountState(edml=edml_layer_count, wdc_raw=wdc_raw)


def find_big_table_pair_columns(core_a: str, core_b: str) -> tuple[str, str, str, str, str]:
	canonical_a = canonical_core_name(core_a)
	canonical_b = canonical_core_name(core_b)

	header = pd.read_csv(BIG_TABLE_PATH, nrows=0)
	columns = [str(col) for col in header.columns]

	for reference_col in columns:
		if not reference_col.endswith("_reference"):
			continue

		prefix = reference_col[: -len("_reference")]
		code_col = f"{prefix}_code"
		if code_col not in columns:
			continue

		core_columns = [
			col
			for col in columns
			if col.startswith(f"{prefix}_") and col not in {reference_col, code_col}
		]
		if len(core_columns) != 2:
			continue

		core_to_column: dict[str, str] = {}
		for col in core_columns:
			core_name = col[len(prefix) + 1 :]
			core_to_column[core_name] = col

		if canonical_a in core_to_column and canonical_b in core_to_column:
			return prefix, core_to_column[canonical_a], core_to_column[canonical_b], reference_col, code_col

	raise ValueError(
		f"Could not find columns in {BIG_TABLE_PATH} for pair {canonical_a}-{canonical_b}."
	)


def label_from_code(code: str, fallback_index: int) -> str:
	text = code.strip()
	if not text:
		return str(fallback_index)
	label = text.rsplit("_", 1)[-1].strip()
	if not label:
		return str(fallback_index)
	return label


def load_tiepoint_pairs_from_big_table(core_a: str, core_b: str) -> tuple[str, list[Tiepoint]]:
	if not BIG_TABLE_PATH.exists():
		raise FileNotFoundError(f"Missing tiepoint table: {BIG_TABLE_PATH}")

	pair_prefix, col_a, col_b, reference_col, code_col = find_big_table_pair_columns(core_a, core_b)
	df = pd.read_csv(
		BIG_TABLE_PATH,
		usecols=[col_a, col_b, reference_col, code_col],
		dtype={reference_col: "string", code_col: "string"},
		low_memory=False,
	)

	depth_a = pd.to_numeric(df[col_a], errors="coerce")
	depth_b = pd.to_numeric(df[col_b], errors="coerce")
	reference_values = df[reference_col].fillna("unknown").astype(str).str.strip()
	code_values = df[code_col].fillna("").astype(str).str.strip()

	valid_rows = depth_a.notna() & depth_b.notna()
	pairs: list[Tiepoint] = []
	for row_idx, (value_a, value_b, code, reference_raw) in enumerate(
		zip(depth_a[valid_rows], depth_b[valid_rows], code_values[valid_rows], reference_values[valid_rows])
	):
		value_a_num = float(value_a)
		value_b_num = float(value_b)
		label = label_from_code(code, row_idx)

		reference = str(reference_raw).strip() if isinstance(reference_raw, str) else ""
		if not reference:
			reference = "unknown"

		pairs.append((value_a_num, value_b_num, label, reference))

	return pair_prefix, pairs


def auto_window_from_values(values: Iterable[float], pad_fraction: float = 0.05, min_pad: float = 1.0) -> tuple[float, float] | None:
	values_list = [float(value) for value in values]
	if not values_list:
		return None
	lo = min(values_list)
	hi = max(values_list)
	span = hi - lo
	pad = max(span * pad_fraction, min_pad)
	return lo - pad, hi + pad


def clamp_window(window: tuple[float, float], bounds: tuple[float, float]) -> tuple[float, float]:
	min_value = max(bounds[0], min(window[0], window[1]))
	max_value = min(bounds[1], max(window[0], window[1]))
	if min_value >= max_value:
		return bounds
	return min_value, max_value


def clip_depth_window(df: pd.DataFrame, min_depth: float, max_depth: float) -> pd.DataFrame:
	return df[(df["depth"] >= min_depth) & (df["depth"] <= max_depth)].copy()


def pairs_in_bottom_window(
	all_pairs: list[Tiepoint],
	window_b: tuple[float, float],
) -> list[Tiepoint]:
	visible: list[Tiepoint] = []
	for depth_a, depth_b, label, reference in all_pairs:
		if window_b[0] <= depth_b <= window_b[1]:
			visible.append((depth_a, depth_b, label, reference))
	return visible


def depth_bounds_from_signals(signals: Sequence[SignalSeries]) -> tuple[float, float]:
	if not signals:
		raise ValueError("At least one signal must be loaded.")
	min_depth = min(float(signal.data["depth"].min()) for signal in signals)
	max_depth = max(float(signal.data["depth"].max()) for signal in signals)
	if min_depth >= max_depth:
		max_depth = min_depth + 1.0
	return min_depth, max_depth


def summarize_selected_source_labels(labels: Sequence[str], max_labels: int = 2) -> str:
	if not labels:
		return "No sources"
	if len(labels) <= max_labels:
		return ", ".join(labels)
	remaining = len(labels) - max_labels
	leading = ", ".join(labels[:max_labels])
	return f"{leading} (+{remaining} more)"


def set_offset_axis_limits(
	axis: plt.Axes,
	values: pd.Series,
	rank: int,
	count: int,
	y_range_scale: float,
) -> None:
	y_min = float(values.min())
	y_max = float(values.max())
	y_span = y_max - y_min
	if y_span == 0:
		y_span = 1.0

	safe_scale = max(float(y_range_scale), MIN_Y_RANGE_SCALE)
	base_pad = (0.08 + AUTO_PAD_FRACTION_STEP * max(count - 1, 0)) * safe_scale
	if count <= 1:
		shift = 0.0
	else:
		centered_rank = rank - (count - 1) / 2.0
		shift = centered_rank * VERTICAL_OFFSET_STEP * safe_scale

	lower_pad_frac = max(base_pad + shift, MIN_VERTICAL_PAD_FRACTION)
	upper_pad_frac = max(base_pad - shift, MIN_VERTICAL_PAD_FRACTION)

	lower = y_min - y_span * lower_pad_frac
	upper = y_max + y_span * upper_pad_frac
	if lower >= upper:
		lower = y_min - y_span * base_pad
		upper = y_max + y_span * base_pad
	axis.set_ylim(lower, upper)


def add_edml_layer_spans(axis: plt.Axes, layer_state: LayerCountState, window: tuple[float, float]) -> None:
	edml = layer_state.edml
	in_window = edml[(edml["depth"] >= window[0]) & (edml["depth"] <= window[1])]

	for row in in_window.itertuples(index=False):
		depth = float(row.depth)
		count = float(row.count)
		if np.isclose(count, 0.5):
			axis.axvspan(
				depth - LAYER_SPAN_HALF_WIDTH,
				depth + LAYER_SPAN_HALF_WIDTH,
				alpha=0.28,
				hatch="///",
				edgecolor="black",
				facecolor="none",
				zorder=0,
			)
		elif np.isclose(count, 1.0):
			axis.axvspan(
				depth - LAYER_SPAN_HALF_WIDTH,
				depth + LAYER_SPAN_HALF_WIDTH,
				color="0.45",
				alpha=0.24,
				zorder=0,
			)


def count_edml_layers(layer_state: LayerCountState, window: tuple[float, float]) -> int:
	edml = layer_state.edml
	in_window = edml[(edml["depth"] >= window[0]) & (edml["depth"] <= window[1])]
	known = in_window[in_window["count"].isin([0.5, 1.0])]
	return int(len(known))


def add_wdc_layer_spans(axis: plt.Axes, layer_state: LayerCountState, window: tuple[float, float]) -> None:
	wdc_raw = layer_state.wdc_raw
	in_window = wdc_raw[(wdc_raw["depth"] >= window[0]) & (wdc_raw["depth"] <= window[1])]

	for row in in_window.itertuples(index=False):
		depth = float(row.depth)
		axis.axvspan(
			depth - LAYER_SPAN_HALF_WIDTH,
			depth + LAYER_SPAN_HALF_WIDTH,
			color="0.55",
			alpha=0.24,
			zorder=0,
		)


def count_wdc_layers(layer_state: LayerCountState, window: tuple[float, float]) -> int:
	wdc_raw = layer_state.wdc_raw
	in_window = wdc_raw[(wdc_raw["depth"] >= window[0]) & (wdc_raw["depth"] <= window[1])]
	return int(len(in_window))


def build_view(
	core_a: str,
	core_b: str,
) -> tuple[list[SignalSeries], list[SignalSeries], list[Tiepoint], tuple[float, float], tuple[float, float], str]:
	pair_prefix, paired = load_tiepoint_pairs_from_big_table(core_a, core_b)
	tie_a = [depth_a for depth_a, _, _, _ in paired]
	tie_b = [depth_b for _, depth_b, _, _ in paired]

	if not paired:
		raise ValueError(f"No tiepoints found in {BIG_TABLE_PATH} for pair prefix {pair_prefix}.")

	if core_a.strip().lower() == "wdc":
		signals_a = load_wdc_signals()
	else:
		signals_a = load_edml_signals()

	if core_b.strip().lower() == "wdc":
		signals_b = load_wdc_signals()
	else:
		signals_b = load_edml_signals()

	bounds_a = depth_bounds_from_signals(signals_a)
	bounds_b = depth_bounds_from_signals(signals_b)

	window_a = auto_window_from_values(tie_a)
	window_b = auto_window_from_values(tie_b)
	if window_a is None:
		window_a = bounds_a
	if window_b is None:
		window_b = bounds_b

	window_a = clamp_window(window_a, bounds_a)
	window_b = clamp_window(window_b, bounds_b)

	return signals_a, signals_b, paired, window_a, window_b, pair_prefix


def plot_pair(core_a: str, core_b: str) -> None:
	canonical_a = canonical_core_name(core_a)
	canonical_b = canonical_core_name(core_b)
	layer_state = load_layer_count_state()
	signals_a, signals_b, all_pairs, window_a, window_b, pair_prefix = build_view(core_a, core_b)

	bounds_a = depth_bounds_from_signals(signals_a)
	bounds_b = depth_bounds_from_signals(signals_b)

	fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
	plt.subplots_adjust(left=0.08, right=0.80, top=0.93, bottom=0.22, hspace=0.16)

	state: dict[str, object] = {
		"window_a": window_a,
		"window_b": window_b,
		"base_window_a": window_a,
		"base_window_b": window_b,
		"data_bounds_a": bounds_a,
		"data_bounds_b": bounds_b,
		"connectors": [],
		"extra_axes_a": [],
		"extra_axes_b": [],
		"y_scale_a": DEFAULT_Y_RANGE_SCALE,
		"y_scale_b": DEFAULT_Y_RANGE_SCALE,
	}

	box_a_min_ax = fig.add_axes([0.18, 0.14, 0.14, 0.035])
	box_a_max_ax = fig.add_axes([0.35, 0.14, 0.14, 0.035])
	box_b_min_ax = fig.add_axes([0.60, 0.14, 0.14, 0.035])
	box_b_max_ax = fig.add_axes([0.77, 0.14, 0.14, 0.035])
	y_scale_a_ax = fig.add_axes([0.18, 0.02, 0.14, 0.035])
	y_scale_b_ax = fig.add_axes([0.60, 0.02, 0.14, 0.035])
	apply_ax = fig.add_axes([0.28, 0.07, 0.16, 0.045])
	sync_ax = fig.add_axes([0.46, 0.07, 0.16, 0.045])
	reset_ax = fig.add_axes([0.64, 0.07, 0.16, 0.045])

	box_a_min = TextBox(box_a_min_ax, f"{canonical_a} min", initial=f"{window_a[0]:g}")
	box_a_max = TextBox(box_a_max_ax, f"{canonical_a} max", initial=f"{window_a[1]:g}")
	box_b_min = TextBox(box_b_min_ax, f"{canonical_b} min", initial=f"{window_b[0]:g}")
	box_b_max = TextBox(box_b_max_ax, f"{canonical_b} max", initial=f"{window_b[1]:g}")
	y_scale_a_box = TextBox(y_scale_a_ax, f"{canonical_a} Y-scale", initial=f"{DEFAULT_Y_RANGE_SCALE:g}")
	y_scale_b_box = TextBox(y_scale_b_ax, f"{canonical_b} Y-scale", initial=f"{DEFAULT_Y_RANGE_SCALE:g}")
	apply_button = Button(apply_ax, "Apply")
	sync_button = Button(sync_ax, "Sync")
	reset_button = Button(reset_ax, "Reset")

	def parse_window(box_min: TextBox, box_max: TextBox, bounds: tuple[float, float], label: str) -> tuple[float, float]:
		try:
			parsed_min = float(box_min.text.strip())
			parsed_max = float(box_max.text.strip())
		except ValueError as exc:
			raise ValueError(f"{label} window values must be numeric.") from exc

		if parsed_min >= parsed_max:
			raise ValueError(f"{label} min must be less than max.")

		return clamp_window((parsed_min, parsed_max), bounds)

	def parse_scale(box: TextBox, label: str) -> float:
		try:
			parsed = float(box.text.strip())
		except ValueError as exc:
			raise ValueError(f"{label} Y-scale must be numeric.") from exc
		if parsed <= 0:
			raise ValueError(f"{label} Y-scale must be > 0.")
		return max(parsed, MIN_Y_RANGE_SCALE)

	def sync_boxes() -> None:
		current_window_a = state["window_a"]
		current_window_b = state["window_b"]
		current_y_scale_a = state["y_scale_a"]
		current_y_scale_b = state["y_scale_b"]
		assert isinstance(current_window_a, tuple)
		assert isinstance(current_window_b, tuple)
		assert isinstance(current_y_scale_a, float)
		assert isinstance(current_y_scale_b, float)
		box_a_min.set_val(f"{current_window_a[0]:g}")
		box_a_max.set_val(f"{current_window_a[1]:g}")
		box_b_min.set_val(f"{current_window_b[0]:g}")
		box_b_max.set_val(f"{current_window_b[1]:g}")
		y_scale_a_box.set_val(f"{current_y_scale_a:g}")
		y_scale_b_box.set_val(f"{current_y_scale_b:g}")

	def synced_window_a_from_window_b(current_window_b: tuple[float, float]) -> tuple[float, float]:
		pairs_in_b = [
			(depth_a, depth_b)
			for depth_a, depth_b, _, _ in all_pairs
			if current_window_b[0] <= depth_b <= current_window_b[1]
		]

		if len(pairs_in_b) < 2:
			raise ValueError("Sync requires at least 2 tiepoints in Core B depth range.")

		pairs_in_b.sort(key=lambda item: item[1])
		a_first, b_first = pairs_in_b[0]
		a_last, b_last = pairs_in_b[-1]

		if b_last == b_first:
			raise ValueError("Cannot sync: first and last Core B tiepoints are identical.")

		scale = (a_last - a_first) / (b_last - b_first)
		a_min = a_first + (current_window_b[0] - b_first) * scale
		a_max = a_first + (current_window_b[1] - b_first) * scale

		if a_min == a_max:
			raise ValueError("Cannot sync: computed Core A range is zero-width.")

		bounds = state["data_bounds_a"]
		assert isinstance(bounds, tuple)
		return clamp_window((a_min, a_max), bounds)

	def redraw() -> None:
		for connector in state["connectors"]:
			try:
				connector.remove()
			except ValueError:
				pass
		state["connectors"] = []

		for extra_key in ("extra_axes_a", "extra_axes_b"):
			extra_axes = state[extra_key]
			assert isinstance(extra_axes, list)
			for extra_ax in extra_axes:
				try:
					extra_ax.remove()
				except ValueError:
					pass
			state[extra_key] = []

		current_window_a = state["window_a"]
		current_window_b = state["window_b"]
		current_y_scale_a = state["y_scale_a"]
		current_y_scale_b = state["y_scale_b"]
		assert isinstance(current_window_a, tuple)
		assert isinstance(current_window_b, tuple)
		assert isinstance(current_y_scale_a, float)
		assert isinstance(current_y_scale_b, float)

		visible = pairs_in_bottom_window(all_pairs, current_window_b)
		draw_tiepoints = len(visible) < MAX_DRAWN_TIEPOINTS
		wdc_layer_count = count_wdc_layers(layer_state, current_window_a)
		edml_layer_count = count_edml_layers(layer_state, current_window_b)
		draw_wdc_layers = wdc_layer_count < MAX_VISIBLE_LAYERS
		draw_edml_layers = edml_layer_count < MAX_VISIBLE_LAYERS

		ax_a.clear()
		ax_b.clear()

		cmap_a = plt.get_cmap("tab10")
		cmap_b = plt.get_cmap("Dark2")

		visible_a: list[tuple[int, SignalSeries, pd.DataFrame]] = []
		visible_b: list[tuple[int, SignalSeries, pd.DataFrame]] = []

		for idx, signal in enumerate(signals_a):
			view = clip_depth_window(signal.data, current_window_a[0], current_window_a[1])
			if not view.empty:
				visible_a.append((idx, signal, view))

		for idx, signal in enumerate(signals_b):
			view = clip_depth_window(signal.data, current_window_b[0], current_window_b[1])
			if not view.empty:
				visible_b.append((idx, signal, view))

		for rank, (idx, signal, view) in enumerate(visible_a):
			color = cmap_a(idx % cmap_a.N)
			if idx == 0:
				target_ax = ax_a
				y_label = f"{signal.canonical_name} {signal.value_label}"
			else:
				target_ax = ax_a.twinx()
				target_ax.spines["right"].set_position(("axes", 1.0 + 0.12 * (idx - 1)))
				target_ax.patch.set_visible(False)
				extra_axes = state["extra_axes_a"]
				assert isinstance(extra_axes, list)
				extra_axes.append(target_ax)
				y_label = signal.value_label

			target_ax.plot(view["depth"], view["value"], color=color, lw=1.2)
			set_offset_axis_limits(target_ax, view["value"], rank, len(visible_a), current_y_scale_a)
			target_ax.set_ylabel(y_label, color=color, fontsize=8)
			target_ax.tick_params(axis="y", colors=color, labelsize=8)
			target_ax.grid(False)

		for rank, (idx, signal, view) in enumerate(visible_b):
			color = cmap_b(idx % cmap_b.N)
			if idx == 0:
				target_ax = ax_b
				y_label = f"{signal.canonical_name} {signal.value_label}"
			else:
				target_ax = ax_b.twinx()
				target_ax.spines["right"].set_position(("axes", 1.0 + 0.12 * (idx - 1)))
				target_ax.patch.set_visible(False)
				extra_axes = state["extra_axes_b"]
				assert isinstance(extra_axes, list)
				extra_axes.append(target_ax)
				y_label = signal.value_label

			target_ax.plot(view["depth"], view["value"], color=color, lw=1.2)
			set_offset_axis_limits(target_ax, view["value"], rank, len(visible_b), current_y_scale_b)
			target_ax.set_ylabel(y_label, color=color, fontsize=8)
			target_ax.tick_params(axis="y", colors=color, labelsize=8)
			target_ax.grid(False)

		if not visible_a:
			ax_a.text(0.5, 0.5, "No data in window", transform=ax_a.transAxes, ha="center", va="center")
		if not visible_b:
			ax_b.text(0.5, 0.5, "No data in window", transform=ax_b.transAxes, ha="center", va="center")

		ax_a.set_xlim(*current_window_a)
		ax_b.set_xlim(*current_window_b)
		ax_b.set_xlabel("Depth (m)")
		ax_a.grid(alpha=0.25)
		ax_b.grid(alpha=0.25)

		sources_a = [signal.source_label for signal in signals_a]
		sources_b = [signal.source_label for signal in signals_b]
		ax_a.set_title(summarize_selected_source_labels(sources_a))
		ax_b.set_title(summarize_selected_source_labels(sources_b))

		if draw_wdc_layers:
			add_wdc_layer_spans(ax_a, layer_state, current_window_a)
		else:
			ax_a.text(
				0.5,
				0.02,
				f"WDC layers hidden: {wdc_layer_count} in range (needs < {MAX_VISIBLE_LAYERS})",
				transform=ax_a.transAxes,
				ha="center",
				va="bottom",
				color="dimgray",
				fontsize=9,
			)

		if draw_edml_layers:
			add_edml_layer_spans(ax_b, layer_state, current_window_b)
		else:
			ax_b.text(
				0.5,
				0.02,
				f"EDML layers hidden: {edml_layer_count} in range (needs < {MAX_VISIBLE_LAYERS})",
				transform=ax_b.transAxes,
				ha="center",
				va="bottom",
				color="dimgray",
				fontsize=9,
			)

		file_handles_a = [
			Line2D([0], [0], color=cmap_a(idx % cmap_a.N), lw=1.5, label=signal.source_label)
			for idx, signal in enumerate(signals_a)
		]
		if file_handles_a:
			data_legend_a = ax_a.legend(
				file_handles_a,
				[handle.get_label() for handle in file_handles_a],
				loc="upper right",
				title="Data sources",
				fontsize=7,
				title_fontsize=8,
			)
			ax_a.add_artist(data_legend_a)

		file_handles_b = [
			Line2D([0], [0], color=cmap_b(idx % cmap_b.N), lw=1.5, label=signal.source_label)
			for idx, signal in enumerate(signals_b)
		]
		if file_handles_b:
			data_legend_b = ax_b.legend(
				file_handles_b,
				[handle.get_label() for handle in file_handles_b],
				loc="upper right",
				title="Data sources",
				fontsize=7,
				title_fontsize=8,
			)
			ax_b.add_artist(data_legend_b)

		layer_handles_wdc = [Patch(facecolor="0.55", alpha=0.24, label="WDC raw layers")]
		layer_handles_edml = [
			Patch(facecolor="0.45", alpha=0.24, label="EDML count=1"),
			Patch(facecolor="none", edgecolor="black", hatch="///", alpha=0.28, label="EDML count=0.5"),
		]
		if draw_wdc_layers:
			layer_legend_wdc = ax_a.legend(handles=layer_handles_wdc, loc="upper left", fontsize=8, title="Layer count")
			ax_a.add_artist(layer_legend_wdc)
		if draw_edml_layers:
			layer_legend_edml = ax_b.legend(handles=layer_handles_edml, loc="upper left", fontsize=8, title="Layer count")
			ax_b.add_artist(layer_legend_edml)

		if draw_tiepoints:
			y_min_a, y_max_a = ax_a.get_ylim()
			y_min_b, y_max_b = ax_b.get_ylim()
			y_span_a = max(y_max_a - y_min_a, 1.0)
			y_span_b = max(y_max_b - y_min_b, 1.0)
			y_line_a = y_min_a + 0.92 * y_span_a
			y_line_b = y_min_b + 0.92 * y_span_b
			y_text_a = y_min_a + 0.965 * y_span_a
			y_text_b = y_min_b + 0.965 * y_span_b

			for depth_a, depth_b, label, _reference in visible:
				if SHOW_VERTICAL_TIE_LINES:
					ax_a.axvline(depth_a, color="black", linestyle="--", lw=1.1, alpha=0.45, zorder=2)
					ax_b.axvline(depth_b, color="black", linestyle="--", lw=1.1, alpha=0.45, zorder=2)

				ax_a.text(
					depth_a,
					y_text_a,
					str(label),
					color="black",
					fontsize=8,
					ha="center",
					va="bottom",
					zorder=6,
				)
				ax_b.text(
					depth_b,
					y_text_b,
					str(label),
					color="black",
					fontsize=8,
					ha="center",
					va="bottom",
					zorder=6,
				)

				connector = ConnectionPatch(
					xyA=(depth_a, y_line_a),
					coordsA=ax_a.transData,
					xyB=(depth_b, y_line_b),
					coordsB=ax_b.transData,
					color="black",
					linestyle="--",
					lw=0.9,
					alpha=0.55,
					zorder=4,
				)
				fig.add_artist(connector)
				state["connectors"].append(connector)

			title_suffix = f" ({len(visible)} volcanic links in bottom range)"
		else:
			ax_a.text(
				0.5,
				0.03,
				f"Volcanic links hidden: {len(visible)} in bottom range (needs < {MAX_DRAWN_TIEPOINTS})",
				transform=ax_a.transAxes,
				ha="center",
				va="bottom",
				color="crimson",
				fontsize=10,
			)
			title_suffix = (
				f" ({len(visible)} volcanic links in bottom range; hidden because >= {MAX_DRAWN_TIEPOINTS})"
			)

		fig.suptitle(
			f"{canonical_a} vs {canonical_b} volcanic links (dashed){title_suffix}",
			fontsize=13,
		)

		fig.canvas.draw_idle()

	def apply_windows(_event=None) -> None:
		try:
			bounds_a = state["data_bounds_a"]
			bounds_b = state["data_bounds_b"]
			assert isinstance(bounds_a, tuple)
			assert isinstance(bounds_b, tuple)
			new_window_a = parse_window(box_a_min, box_a_max, bounds_a, canonical_a)
			new_window_b = parse_window(box_b_min, box_b_max, bounds_b, canonical_b)
			new_y_scale_a = parse_scale(y_scale_a_box, canonical_a)
			new_y_scale_b = parse_scale(y_scale_b_box, canonical_b)
		except ValueError as exc:
			fig.suptitle(str(exc), fontsize=12, color="crimson")
			fig.canvas.draw_idle()
			return

		state["window_a"] = new_window_a
		state["window_b"] = new_window_b
		state["y_scale_a"] = new_y_scale_a
		state["y_scale_b"] = new_y_scale_b
		sync_boxes()
		redraw()

	def on_sync(_event) -> None:
		current_window_b = state["window_b"]
		assert isinstance(current_window_b, tuple)
		try:
			new_window_a = synced_window_a_from_window_b(current_window_b)
		except ValueError as exc:
			fig.suptitle(str(exc), fontsize=12, color="crimson")
			fig.canvas.draw_idle()
			return

		state["window_a"] = new_window_a
		sync_boxes()
		redraw()

	def on_reset(_event) -> None:
		bounds_a = state["data_bounds_a"]
		bounds_b = state["data_bounds_b"]
		base_window_a = state["base_window_a"]
		base_window_b = state["base_window_b"]
		assert isinstance(bounds_a, tuple)
		assert isinstance(bounds_b, tuple)
		assert isinstance(base_window_a, tuple)
		assert isinstance(base_window_b, tuple)
		state["window_a"] = clamp_window(base_window_a, bounds_a)
		state["window_b"] = clamp_window(base_window_b, bounds_b)
		state["y_scale_a"] = DEFAULT_Y_RANGE_SCALE
		state["y_scale_b"] = DEFAULT_Y_RANGE_SCALE
		sync_boxes()
		redraw()

	box_a_min.on_submit(apply_windows)
	box_a_max.on_submit(apply_windows)
	box_b_min.on_submit(apply_windows)
	box_b_max.on_submit(apply_windows)
	y_scale_a_box.on_submit(apply_windows)
	y_scale_b_box.on_submit(apply_windows)
	apply_button.on_clicked(apply_windows)
	sync_button.on_clicked(on_sync)
	reset_button.on_clicked(on_reset)

	sync_boxes()
	redraw()

	print("WDC signal sources:")
	for signal in signals_a:
		print(f"  - {signal.source_label}")
	print("EDML signal sources:")
	for signal in signals_b:
		print(f"  - {signal.source_label}")
	print(f"Tiepoint table: {BIG_TABLE_PATH}")
	print(f"Tiepoint pair prefix: {pair_prefix}")
	print(f"Layer-count sources: {WDC_RAW_LAYER_COUNT_FILE}, {EDML_LAYER_COUNT_FILE}")
	print(f"{canonical_a} window: [{window_a[0]:g}, {window_a[1]:g}]")
	print(f"{canonical_b} window: [{window_b[0]:g}, {window_b[1]:g}]")
	print(f"{canonical_a} Y-scale: {state['y_scale_a']}")
	print(f"{canonical_b} Y-scale: {state['y_scale_b']}")
	print(f"Volcanic links loaded: {len(all_pairs)}")

	plt.show()


def main() -> None:
	core_a, core_b = CORE_PAIR
	plot_pair(core_a, core_b)


if __name__ == "__main__":
	main()