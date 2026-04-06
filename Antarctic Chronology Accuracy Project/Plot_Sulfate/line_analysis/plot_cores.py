from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Button, TextBox


CORE_PAIR: tuple[str, str] = ("wdc", "td")

CORE_DATA_FILES: dict[str, str | None] = {
	"df": None,
	"edc": None,
	"edml": None,
	"td": None,
	"wdc": 'wdc_sulfate.txt',
}

PREFERRED_DATA_KEYWORDS: tuple[str, ...] = ("sulfate", "ecm", "dep")
SHOW_VERTICAL_TIE_LINES = True
LABEL_START_INDEX = 0
MAX_DRAWN_TIEPOINTS = 25

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


@dataclass
class CoreSeries:
	core_key: str
	canonical_name: str
	file_path: Path
	data: pd.DataFrame
	value_label: str


def canonical_core_name(core: str) -> str:
	key = core.strip().lower()
	if key not in CORE_ALIASES:
		supported = ", ".join(sorted(CORE_ALIASES))
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


def rank_data_file(file_path: Path) -> tuple[int, str]:
	name = file_path.name.lower()
	for idx, keyword in enumerate(PREFERRED_DATA_KEYWORDS):
		if keyword in name:
			return idx, name
	return len(PREFERRED_DATA_KEYWORDS), name


def load_core_data(file_path: Path) -> pd.DataFrame:
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

	value_col = next((col for col in df.columns if col != depth_col), None)
	if value_col is None:
		raise ValueError(f"Could not identify a data column in {file_path}")

	out_df = pd.DataFrame(
		{
			"depth": pd.to_numeric(df[depth_col], errors="coerce"),
			"value": pd.to_numeric(df[value_col], errors="coerce"),
		}
	).dropna()
	out_df.attrs["value_label"] = str(value_col)
	return out_df


def choose_data_file(core: str, prefer_window: tuple[float, float] | None = None) -> Path:
	core_key = core.strip().lower()
	requested = CORE_DATA_FILES.get(core_key)
	folder = core_out_folder(core_key)

	if requested:
		candidate = folder / requested
		if not candidate.exists():
			raise FileNotFoundError(f"Requested data file not found for core '{core_key}': {candidate}")
		return candidate

	txt_files = sorted(p for p in folder.glob("*.txt") if p.is_file())
	if not txt_files:
		raise FileNotFoundError(f"No .txt data files found in {folder}")

	if prefer_window is not None:
		w_min, w_max = prefer_window
		scored_files: list[tuple[int, int, str, Path]] = []
		for candidate in txt_files:
			try:
				preview = load_core_data(candidate)
				overlap = int(((preview["depth"] >= w_min) & (preview["depth"] <= w_max)).sum())
			except Exception:
				overlap = -1
			pref_rank, name_rank = rank_data_file(candidate)
			scored_files.append((overlap, -pref_rank, name_rank, candidate))
		scored_files.sort(reverse=True)
		best_overlap, _, _, best_file = scored_files[0]
		if best_overlap > 0:
			return best_file

	txt_files.sort(key=rank_data_file)
	return txt_files[0]


def load_core_series(core: str, prefer_window: tuple[float, float] | None = None) -> CoreSeries:
	canonical_name = canonical_core_name(core)
	file_path = choose_data_file(core, prefer_window=prefer_window)
	data = load_core_data(file_path)
	value_label = str(data.attrs.get("value_label", "value"))
	return CoreSeries(
		core_key=core.strip().lower(),
		canonical_name=canonical_name,
		file_path=file_path,
		data=data,
		value_label=value_label,
	)


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


def load_tiepoint_depths(pair_dir: Path, core: str) -> list[float]:
	canonical = canonical_core_name(core)
	tie_file = pair_dir / f"{canonical}.txt"
	if not tie_file.exists():
		raise FileNotFoundError(f"Missing tiepoint file for {canonical}: {tie_file}")

	series = pd.read_csv(tie_file, header=None, comment="#").iloc[:, 0]
	depths = pd.to_numeric(series, errors="coerce").dropna().tolist()
	return [float(depth) for depth in depths]


def paired_depths(depths_a: Iterable[float], depths_b: Iterable[float]) -> list[tuple[float, float, int]]:
	pairs = list(zip(depths_a, depths_b))
	return [(float(a), float(b), idx + LABEL_START_INDEX) for idx, (a, b) in enumerate(pairs)]


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


def visible_pairs(
	all_pairs: list[tuple[float, float, int]],
	window_a: tuple[float, float],
	window_b: tuple[float, float],
) -> list[tuple[float, float, int]]:
	visible: list[tuple[float, float, int]] = []
	for depth_a, depth_b, idx in all_pairs:
		if window_a[0] <= depth_a <= window_a[1] and window_b[0] <= depth_b <= window_b[1]:
			visible.append((depth_a, depth_b, idx))
	return visible


def marker_y_positions(values: pd.Series) -> tuple[float, float]:
	y_min = float(values.min())
	y_max = float(values.max())
	y_span = y_max - y_min
	if y_span == 0:
		y_span = 1.0
	y_line = y_min + 0.92 * y_span
	y_text = y_min + 0.965 * y_span
	return y_line, y_text


def build_view(
	core_a: str,
	core_b: str,
) -> tuple[CoreSeries, CoreSeries, list[tuple[float, float, int]], tuple[float, float], tuple[float, float]]:
	pair_dir = find_tiepoint_pair_dir(core_a, core_b)
	tie_a = load_tiepoint_depths(pair_dir, core_a)
	tie_b = load_tiepoint_depths(pair_dir, core_b)
	paired = paired_depths(tie_a, tie_b)

	window_a = auto_window_from_values(tie_a)
	window_b = auto_window_from_values(tie_b)

	series_a = load_core_series(core_a, prefer_window=window_a)
	series_b = load_core_series(core_b, prefer_window=window_b)

	data_bounds_a = (float(series_a.data["depth"].min()), float(series_a.data["depth"].max()))
	data_bounds_b = (float(series_b.data["depth"].min()), float(series_b.data["depth"].max()))

	if window_a is None:
		window_a = data_bounds_a
	else:
		window_a = clamp_window(window_a, data_bounds_a)

	if window_b is None:
		window_b = data_bounds_b
	else:
		window_b = clamp_window(window_b, data_bounds_b)

	return series_a, series_b, paired, window_a, window_b


def plot_pair(core_a: str, core_b: str) -> None:
	series_a, series_b, all_pairs, window_a, window_b = build_view(core_a, core_b)
	data_bounds_a = (float(series_a.data["depth"].min()), float(series_a.data["depth"].max()))
	data_bounds_b = (float(series_b.data["depth"].min()), float(series_b.data["depth"].max()))

	fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
	plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.22, hspace=0.16)

	state: dict[str, object] = {
		"window_a": window_a,
		"window_b": window_b,
		"connectors": [],
	}

	box_a_min_ax = fig.add_axes([0.18, 0.14, 0.14, 0.035])
	box_a_max_ax = fig.add_axes([0.35, 0.14, 0.14, 0.035])
	box_b_min_ax = fig.add_axes([0.60, 0.14, 0.14, 0.035])
	box_b_max_ax = fig.add_axes([0.77, 0.14, 0.14, 0.035])
	apply_ax = fig.add_axes([0.28, 0.07, 0.16, 0.045])
	sync_ax = fig.add_axes([0.46, 0.07, 0.16, 0.045])
	reset_ax = fig.add_axes([0.64, 0.07, 0.16, 0.045])

	box_a_min = TextBox(box_a_min_ax, f"{series_a.canonical_name} min", initial=f"{window_a[0]:g}")
	box_a_max = TextBox(box_a_max_ax, f"{series_a.canonical_name} max", initial=f"{window_a[1]:g}")
	box_b_min = TextBox(box_b_min_ax, f"{series_b.canonical_name} min", initial=f"{window_b[0]:g}")
	box_b_max = TextBox(box_b_max_ax, f"{series_b.canonical_name} max", initial=f"{window_b[1]:g}")
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

	def sync_boxes() -> None:
		current_window_a = state["window_a"]
		current_window_b = state["window_b"]
		assert isinstance(current_window_a, tuple)
		assert isinstance(current_window_b, tuple)
		box_a_min.set_val(f"{current_window_a[0]:g}")
		box_a_max.set_val(f"{current_window_a[1]:g}")
		box_b_min.set_val(f"{current_window_b[0]:g}")
		box_b_max.set_val(f"{current_window_b[1]:g}")

	def synced_window_b_from_window_a(current_window_a: tuple[float, float]) -> tuple[float, float]:
		pairs_in_a = [
			(depth_a, depth_b, idx)
			for depth_a, depth_b, idx in all_pairs
			if current_window_a[0] <= depth_a <= current_window_a[1]
		]

		if len(pairs_in_a) < 2:
			raise ValueError(
				"Sync requires at least 2 tiepoints in Core A depth range."
			)

		pairs_in_a.sort(key=lambda item: item[0])
		a_first, b_first, _ = pairs_in_a[0]
		a_last, b_last, _ = pairs_in_a[-1]

		if a_last == a_first:
			raise ValueError("Cannot sync: first and last Core A tiepoints are identical.")

		scale = (b_last - b_first) / (a_last - a_first)
		b_min = b_first + (current_window_a[0] - a_first) * scale
		b_max = b_first + (current_window_a[1] - a_first) * scale

		if b_min == b_max:
			raise ValueError("Cannot sync: computed Core B range is zero-width.")

		return clamp_window((b_min, b_max), data_bounds_b)

	def redraw() -> None:
		for connector in state["connectors"]:
			try:
				connector.remove()
			except ValueError:
				pass
		state["connectors"] = []

		current_window_a = state["window_a"]
		current_window_b = state["window_b"]
		assert isinstance(current_window_a, tuple)
		assert isinstance(current_window_b, tuple)

		view_a = clip_depth_window(series_a.data, current_window_a[0], current_window_a[1])
		view_b = clip_depth_window(series_b.data, current_window_b[0], current_window_b[1])
		visible = visible_pairs(all_pairs, current_window_a, current_window_b)
		draw_tiepoints = len(visible) <= MAX_DRAWN_TIEPOINTS

		ax_a.clear()
		ax_b.clear()

		if not view_a.empty:
			ax_a.plot(view_a["depth"], view_a["value"], color="tab:blue", lw=1.2)
			y_line_a, y_text_a = marker_y_positions(view_a["value"])
		else:
			y_line_a, y_text_a = 0.0, 0.0
			ax_a.text(0.5, 0.5, "No data in window", transform=ax_a.transAxes, ha="center", va="center")

		if not view_b.empty:
			ax_b.plot(view_b["depth"], view_b["value"], color="tab:orange", lw=1.2)
			y_line_b, y_text_b = marker_y_positions(view_b["value"])
		else:
			y_line_b, y_text_b = 0.0, 0.0
			ax_b.text(0.5, 0.5, "No data in window", transform=ax_b.transAxes, ha="center", va="center")

		ax_a.set_xlim(*current_window_a)
		ax_b.set_xlim(*current_window_b)
		ax_a.set_ylabel(f"{series_a.canonical_name} {series_a.value_label}")
		ax_b.set_ylabel(f"{series_b.canonical_name} {series_b.value_label}")
		ax_b.set_xlabel("Depth")
		ax_a.set_title(series_a.file_path.name)
		ax_b.set_title(series_b.file_path.name)
		ax_a.grid(alpha=0.25)
		ax_b.grid(alpha=0.25)

		seg_half_width_a = max((current_window_a[1] - current_window_a[0]) * 0.008, 0.01)
		seg_half_width_b = max((current_window_b[1] - current_window_b[0]) * 0.008, 0.01)

		if draw_tiepoints:
			for depth_a, depth_b, idx in visible:
				if SHOW_VERTICAL_TIE_LINES:
					ax_a.axvline(depth_a, color="gray", lw=2.5, alpha=0.15, zorder=1)
					ax_b.axvline(depth_b, color="gray", lw=2.5, alpha=0.15, zorder=1)

				ax_a.hlines(
					y=y_line_a,
					xmin=depth_a - seg_half_width_a,
					xmax=depth_a + seg_half_width_a,
					color="crimson",
					lw=1.2,
					zorder=5,
				)
				ax_b.hlines(
					y=y_line_b,
					xmin=depth_b - seg_half_width_b,
					xmax=depth_b + seg_half_width_b,
					color="crimson",
					lw=1.2,
					zorder=5,
				)

				ax_a.text(
					depth_a,
					y_text_a,
					str(idx),
					color="crimson",
					fontsize=8,
					ha="center",
					va="bottom",
					zorder=6,
				)
				ax_b.text(
					depth_b,
					y_text_b,
					str(idx),
					color="crimson",
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
					color="crimson",
					lw=0.8,
					alpha=0.55,
					zorder=4,
				)
				fig.add_artist(connector)
				state["connectors"].append(connector)

			title_suffix = f" ({len(visible)} visible tiepoints)"
		else:
			ax_a.text(
				0.5,
				0.03,
				f"Tiepoints hidden: {len(visible)} in range (> {MAX_DRAWN_TIEPOINTS})",
				transform=ax_a.transAxes,
				ha="center",
				va="bottom",
				color="crimson",
				fontsize=10,
			)
			title_suffix = (
				f" ({len(visible)} visible tiepoints; hidden because > {MAX_DRAWN_TIEPOINTS})"
			)

		fig.suptitle(
			f"{series_a.canonical_name} vs {series_b.canonical_name} tiepoints{title_suffix}",
			fontsize=13,
		)

		fig.canvas.draw_idle()

	def apply_windows(_event=None) -> None:
		try:
			new_window_a = parse_window(box_a_min, box_a_max, data_bounds_a, series_a.canonical_name)
			new_window_b = parse_window(box_b_min, box_b_max, data_bounds_b, series_b.canonical_name)
		except ValueError as exc:
			fig.suptitle(str(exc), fontsize=12, color="crimson")
			fig.canvas.draw_idle()
			return

		state["window_a"] = new_window_a
		state["window_b"] = new_window_b
		sync_boxes()
		redraw()

	def on_sync(_event) -> None:
		current_window_a = state["window_a"]
		assert isinstance(current_window_a, tuple)
		try:
			new_window_b = synced_window_b_from_window_a(current_window_a)
		except ValueError as exc:
			fig.suptitle(str(exc), fontsize=12, color="crimson")
			fig.canvas.draw_idle()
			return

		state["window_b"] = new_window_b
		sync_boxes()
		redraw()

	def on_reset(_event) -> None:
		state["window_a"] = window_a
		state["window_b"] = window_b
		sync_boxes()
		redraw()

	box_a_min.on_submit(apply_windows)
	box_a_max.on_submit(apply_windows)
	box_b_min.on_submit(apply_windows)
	box_b_max.on_submit(apply_windows)
	apply_button.on_clicked(apply_windows)
	sync_button.on_clicked(on_sync)
	reset_button.on_clicked(on_reset)

	sync_boxes()
	redraw()

	print(f"Core A file: {series_a.file_path}")
	print(f"Core B file: {series_b.file_path}")
	print(f"Tiepoint folder: {find_tiepoint_pair_dir(core_a, core_b)}")
	print(f"Core A window: [{window_a[0]:g}, {window_a[1]:g}]")
	print(f"Core B window: [{window_b[0]:g}, {window_b[1]:g}]")
	print(f"Tiepoints loaded: {len(all_pairs)}")

	plt.show()


def main() -> None:
	core_a, core_b = CORE_PAIR
	plot_pair(core_a, core_b)


if __name__ == "__main__":
	main()
