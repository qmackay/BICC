from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch


# -----------------------------
# User-adjustable configuration
# -----------------------------
CORE_PAIR: Tuple[str, str] = ("td", "wdc")

# Reference depth window applied to CORE_PAIR[0] (the first core in CORE_PAIR).
REFERENCE_MIN_DEPTH: float = 0.0
REFERENCE_MAX_DEPTH: float = 500.0

# Optional manual nudges for the derived CORE_PAIR[1] depth window.
# Final second-core window is:
#   auto_min + SECOND_CORE_MIN_ADJUST, auto_max + SECOND_CORE_MAX_ADJUST
SECOND_CORE_MIN_ADJUST: float = 0.0
SECOND_CORE_MAX_ADJUST: float = 0.0

# Optional explicit override for second-core depth window.
# Set to None to use auto-derived window from tiepoints.
# Example: (120.0, 185.0)
SECOND_CORE_MANUAL_WINDOW: Tuple[float, float] | None = None

# Draw faint vertical lines at each tiepoint depth in each panel.
SHOW_VERTICAL_TIE_LINES: bool = True

# Set a file name for a core to force that dataset (for example, "df_ecm.txt").
# Use None to auto-pick a file from out/<core>/.
CORE_DATA_FILES: Dict[str, str | None] = {
	"df": None,
	"edc": None,
	"edml": None,
	"td": None,
	"wdc": None,
}

# Auto-pick priority when CORE_DATA_FILES[core] is None.
PREFERRED_DATA_KEYWORDS: Tuple[str, ...] = ("sulfate", "ecm", "dep")


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_SULFATE_DIR = SCRIPT_DIR.parent
OUT_DIR = PLOT_SULFATE_DIR / "out"
TIEPOINT_DIR = PLOT_SULFATE_DIR / "tiepoints"

CORE_ALIASES = {
	"df": "DF",
	"edc": "EDC",
	"edml": "EDML",
	"td": "TALDICE",
	"taldice": "TALDICE",
	"wdc": "WDC",
}


def canonical_core_name(core: str) -> str:
	key = core.strip().lower()
	if key not in CORE_ALIASES:
		supported = ", ".join(sorted(CORE_ALIASES.keys()))
		raise ValueError(f"Unsupported core '{core}'. Supported keys: {supported}")
	return CORE_ALIASES[key]


def core_out_folder(core: str) -> Path:
	key = core.strip().lower()
	if key == "taldice":
		key = "td"
	folder = OUT_DIR / key
	if not folder.exists():
		raise FileNotFoundError(f"Missing output folder for core '{core}': {folder}")
	return folder


def rank_data_file(file_path: Path) -> Tuple[int, str]:
	name = file_path.name.lower()
	for idx, keyword in enumerate(PREFERRED_DATA_KEYWORDS):
		if keyword in name:
			return idx, name
	return len(PREFERRED_DATA_KEYWORDS), name


def choose_data_file(
	core: str,
	prefer_window: Tuple[float, float] | None = None,
) -> Path:
	core_key = core.strip().lower()
	requested = CORE_DATA_FILES.get(core_key)
	folder = core_out_folder(core_key)

	if requested:
		candidate = folder / requested
		if not candidate.exists():
			raise FileNotFoundError(
				f"Requested data file not found for core '{core_key}': {candidate}"
			)
		return candidate

	txt_files = sorted(p for p in folder.glob("*.txt") if p.is_file())
	if not txt_files:
		raise FileNotFoundError(f"No .txt data files found in {folder}")

	if prefer_window is not None:
		w_min, w_max = prefer_window
		window_scores: List[Tuple[int, int, str, Path]] = []
		for p in txt_files:
			try:
				df_tmp = load_core_data(p)
				overlap = int(((df_tmp["depth"] >= w_min) & (df_tmp["depth"] <= w_max)).sum())
			except Exception:
				overlap = -1
			pref_rank, name_rank = rank_data_file(p)
			window_scores.append((overlap, -pref_rank, name_rank, p))

		# Prefer highest overlap in the requested window; break ties by keyword preference.
		window_scores.sort(reverse=True)
		best_overlap, _, _, best_file = window_scores[0]
		if best_overlap > 0:
			return best_file

	txt_files.sort(key=rank_data_file)
	return txt_files[0]


def load_core_data(file_path: Path) -> pd.DataFrame:
	# Data files are primarily CSV, but this fallback handles whitespace-delimited data.
	df = pd.read_csv(file_path)
	if df.shape[1] < 2:
		df = pd.read_csv(file_path, sep=r"\s+", engine="python")

	depth_col = None
	for col in df.columns:
		if "depth" in str(col).strip().lower():
			depth_col = col
			break
	if depth_col is None:
		depth_col = df.columns[0]

	value_col = next((c for c in df.columns if c != depth_col), None)
	if value_col is None:
		raise ValueError(f"Could not identify a data column in {file_path}")

	out_df = pd.DataFrame(
		{
			"depth": pd.to_numeric(df[depth_col], errors="coerce"),
			"value": pd.to_numeric(df[value_col], errors="coerce"),
		}
	).dropna()

	return out_df


def find_tiepoint_pair_dir(core_a: str, core_b: str) -> Path:
	a = canonical_core_name(core_a)
	b = canonical_core_name(core_b)
	direct = TIEPOINT_DIR / f"{a}-{b}"
	reverse = TIEPOINT_DIR / f"{b}-{a}"

	if direct.exists():
		return direct
	if reverse.exists():
		return reverse
	raise FileNotFoundError(f"No tiepoint folder found for pair {a}-{b}")


def load_tiepoint_depths(pair_dir: Path, core: str) -> List[float]:
	canonical = canonical_core_name(core)
	tie_file = pair_dir / f"{canonical}.txt"
	if not tie_file.exists():
		raise FileNotFoundError(f"Missing tiepoint file for {canonical}: {tie_file}")

	# Files contain one optional header line, then one depth value per line.
	ser = pd.read_csv(tie_file, header=None, comment="#").iloc[:, 0]
	depths = pd.to_numeric(ser, errors="coerce").dropna().tolist()
	return [float(d) for d in depths]


def clip_depth_window(df: pd.DataFrame, min_depth: float, max_depth: float) -> pd.DataFrame:
	return df[(df["depth"] >= min_depth) & (df["depth"] <= max_depth)].copy()


def in_window_pairs(
	depths_a: Iterable[float],
	depths_b: Iterable[float],
	min_depth: float,
	max_depth: float,
) -> List[Tuple[float, float, int]]:
	pairs = list(zip(depths_a, depths_b))
	visible: List[Tuple[float, float, int]] = []
	for idx, (a, b) in enumerate(pairs):
		if min_depth <= a <= max_depth and min_depth <= b <= max_depth:
			visible.append((a, b, idx))
	return visible


def derive_other_core_window(
	paired_depths_other: Iterable[float],
	min_adjust: float,
	max_adjust: float,
	manual_window: Tuple[float, float] | None,
) -> Tuple[float, float]:
	if manual_window is not None:
		manual_min, manual_max = manual_window
		if manual_min >= manual_max:
			raise ValueError("SECOND_CORE_MANUAL_WINDOW must have min < max.")
		return float(manual_min), float(manual_max)

	other_depths = [float(x) for x in paired_depths_other]
	if not other_depths:
		raise ValueError(
			"No tiepoints from the second core match the reference-core depth range."
		)

	auto_min = min(other_depths)
	auto_max = max(other_depths)
	final_min = auto_min + float(min_adjust)
	final_max = auto_max + float(max_adjust)

	if final_min >= final_max:
		raise ValueError("Second-core window is invalid after manual adjustments.")

	return final_min, final_max


def marker_y_positions(values: pd.Series) -> Tuple[float, float]:
	y_min = values.min()
	y_max = values.max()
	y_span = y_max - y_min
	if y_span == 0:
		y_span = 1.0
	y_line = y_min + 0.92 * y_span
	y_text = y_min + 0.965 * y_span
	return y_line, y_text


def plot_pair(
	core_a: str,
	core_b: str,
	ref_min_depth: float,
	ref_max_depth: float,
	other_min_adjust: float,
	other_max_adjust: float,
	other_manual_window: Tuple[float, float] | None,
) -> None:
	if ref_min_depth >= ref_max_depth:
		raise ValueError("REFERENCE_MIN_DEPTH must be < REFERENCE_MAX_DEPTH.")

	file_a = choose_data_file(core_a, prefer_window=(ref_min_depth, ref_max_depth))
	canonical_a = canonical_core_name(core_a)
	canonical_b = canonical_core_name(core_b)

	full_a = load_core_data(file_a)

	pair_dir = find_tiepoint_pair_dir(core_a, core_b)
	ties_a = load_tiepoint_depths(pair_dir, core_a)
	ties_b = load_tiepoint_depths(pair_dir, core_b)

	if len(ties_a) != len(ties_b):
		n = min(len(ties_a), len(ties_b))
		ties_a = ties_a[:n]
		ties_b = ties_b[:n]

	# Step 1: select tiepoints using the reference window on the first core.
	reference_pairs = in_window_pairs(ties_a, ties_b, ref_min_depth, ref_max_depth)
	if not reference_pairs:
		raise ValueError(
			"No tiepoint pairs found in the reference-core depth range."
		)

	# Step 2: derive second-core window from corresponding paired tiepoints.
	other_depths_from_pairs = [b for _, b, _ in reference_pairs]
	other_min_depth, other_max_depth = derive_other_core_window(
		other_depths_from_pairs,
		other_min_adjust,
		other_max_adjust,
		other_manual_window,
	)

	file_b = choose_data_file(core_b, prefer_window=(other_min_depth, other_max_depth))
	full_b = load_core_data(file_b)

	# Step 3: clip data separately for each core's final window.
	data_a = clip_depth_window(full_a, ref_min_depth, ref_max_depth)
	data_b = clip_depth_window(full_b, other_min_depth, other_max_depth)
	if data_a.empty:
		raise ValueError("No reference-core data in the requested depth range.")
	if data_b.empty:
		raise ValueError("No second-core data in the derived depth range.")

	# Draw only tiepoints visible in both final panel windows.
	visible_pairs = [
		(a, b, idx)
		for a, b, idx in reference_pairs
		if other_min_depth <= b <= other_max_depth
	]

	fig, (ax_top, ax_bottom) = plt.subplots(
		2,
		1,
		figsize=(14, 8),
		sharex=False,
		constrained_layout=True,
		gridspec_kw={"hspace": 0.08},
	)

	ax_top.plot(data_a["depth"], data_a["value"], color="tab:blue", lw=1.15)
	ax_bottom.plot(data_b["depth"], data_b["value"], color="tab:orange", lw=1.15)

	ax_top.set_ylabel(f"{canonical_a} data")
	ax_bottom.set_ylabel(f"{canonical_b} data")
	ax_bottom.set_xlabel("Depth")
	ax_top.set_xlim(ref_min_depth, ref_max_depth)
	ax_bottom.set_xlim(other_min_depth, other_max_depth)

	y_line_top, y_text_top = marker_y_positions(data_a["value"])
	y_line_bottom, y_text_bottom = marker_y_positions(data_b["value"])
	seg_half_width_top = (ref_max_depth - ref_min_depth) * 0.008
	seg_half_width_bottom = (other_max_depth - other_min_depth) * 0.008

	for depth_a, depth_b, idx in visible_pairs:
		if SHOW_VERTICAL_TIE_LINES:
			ax_top.axvline(depth_a, color="gray", lw=3, alpha=0.2, zorder=1)
			ax_bottom.axvline(depth_b, color="gray", lw=3, alpha=0.2, zorder=1)

		ax_top.hlines(
			y=y_line_top,
			xmin=depth_a - seg_half_width_top,
			xmax=depth_a + seg_half_width_top,
			color="crimson",
			lw=1.0,
			zorder=5,
		)
		ax_bottom.hlines(
			y=y_line_bottom,
			xmin=depth_b - seg_half_width_bottom,
			xmax=depth_b + seg_half_width_bottom,
			color="crimson",
			lw=1.0,
			zorder=5,
		)

		ax_top.text(
			depth_a,
			y_text_top,
			str(idx),
			color="crimson",
			fontsize=8,
			ha="center",
			va="bottom",
			zorder=6,
		)
		ax_bottom.text(
			depth_b,
			y_text_bottom,
			str(idx),
			color="crimson",
			fontsize=8,
			ha="center",
			va="bottom",
			zorder=6,
		)

		connector = ConnectionPatch(
			xyA=(depth_a, y_line_top),
			coordsA=ax_top.transData,
			xyB=(depth_b, y_line_bottom),
			coordsB=ax_bottom.transData,
			color="crimson",
			lw=0.8,
			alpha=0.55,
			zorder=4,
		)
		fig.add_artist(connector)

	ax_top.grid(alpha=0.25)
	ax_bottom.grid(alpha=0.25)

	fig.suptitle(
		f"{canonical_a} [{ref_min_depth:g}-{ref_max_depth:g}] vs "
		f"{canonical_b} [{other_min_depth:g}-{other_max_depth:g}]",
		fontsize=12,
	)

	if not visible_pairs:
		print("No tiepoints from the reference window fall inside the second-core panel window.")

	print(f"Core A file: {file_a}")
	print(f"Core B file: {file_b}")
	print(f"Tiepoint folder: {pair_dir}")
	print(f"Reference core ({canonical_a}) window: [{ref_min_depth:g}, {ref_max_depth:g}]")
	print(f"Second core ({canonical_b}) window: [{other_min_depth:g}, {other_max_depth:g}]")
	print(f"Tiepoint pairs in reference window: {len(reference_pairs)}")
	print(f"Tiepoint pairs drawn: {len(visible_pairs)}")
	if other_manual_window is None:
		print(
			f"Second-core auto window from tiepoints plus adjustments: "
			f"dmin += {other_min_adjust:g}, dmax += {other_max_adjust:g}"
		)
	else:
		print("Second-core window source: SECOND_CORE_MANUAL_WINDOW override")

	plt.show()


if __name__ == "__main__":
	core_a, core_b = CORE_PAIR
	plot_pair(
		core_a=core_a,
		core_b=core_b,
		ref_min_depth=REFERENCE_MIN_DEPTH,
		ref_max_depth=REFERENCE_MAX_DEPTH,
		other_min_adjust=SECOND_CORE_MIN_ADJUST,
		other_max_adjust=SECOND_CORE_MAX_ADJUST,
		other_manual_window=SECOND_CORE_MANUAL_WINDOW,
	)
