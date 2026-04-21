from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.widgets import Button, TextBox


CORE_PAIR: tuple[str, str] = ("wdc", "edml")

CORE_DATA_FILES: dict[str, str | None] = {
	"df": None,
	"edc": 'edc_sulfate.txt',
	"edml": 'edml_sulfate.txt',
	"td": None,
	"wdc": 'wdc_sulfate.txt',
}

PREFERRED_DATA_KEYWORDS: tuple[str, ...] = ("sulfate", "ecm", "dep")
SHOW_VERTICAL_TIE_LINES = True
MAX_DRAWN_TIEPOINTS = 25

SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_SULFATE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = PLOT_SULFATE_DIR.parent
OUT_DIR = PLOT_SULFATE_DIR / "out"
BIG_TABLE_PATH = PROJECT_DIR / "all_tiepoints" / "big_table.csv"

CORE_ALIASES = {
	"df": "DF",
	"edc": "EDC",
	"edml": "EDML",
	"td": "TALDICE",
	"taldice": "TALDICE",
	"wdc": "WDC",
}
CORE_SELECTION_KEYS: tuple[str, ...] = tuple(sorted(CORE_DATA_FILES.keys()))


@dataclass
class CoreSeries:
	core_key: str
	canonical_name: str
	file_path: Path
	data: pd.DataFrame
	value_label: str


Tiepoint = tuple[float, float, str, str]


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


def list_core_data_files(core: str) -> list[Path]:
	core_key = core.strip().lower()
	folder = core_out_folder(core_key)
	txt_files = sorted((p for p in folder.glob("*.txt") if p.is_file()), key=rank_data_file)
	if not txt_files:
		raise FileNotFoundError(f"No .txt data files found in {folder}")
	return txt_files


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
		if isinstance(code, str):
			label = label_from_code(code, row_idx)
		else:
			label = str(row_idx)

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


def build_reference_color_map(all_pairs: list[Tiepoint]) -> dict[str, tuple[float, float, float, float]]:
	unique_references = sorted({reference for _, _, _, reference in all_pairs})
	cmap = plt.get_cmap("tab20")
	return {reference: cmap(idx % cmap.N) for idx, reference in enumerate(unique_references)}


class DropdownSelector:
	def __init__(
		self,
		fig: plt.Figure,
		button_rect: list[float],
		menu_rect: list[float],
		title: str,
		options: list[str],
		initial: str,
		on_select: Callable[[str], None],
		on_toggle: Callable[["DropdownSelector"], None],
	) -> None:
		self.fig = fig
		self.title = title
		self.options = options
		self.menu_rect = menu_rect
		self.on_select = on_select
		self.on_toggle = on_toggle

		if initial not in options:
			raise ValueError(f"Initial option '{initial}' not in options for {title}.")

		self.button_ax = fig.add_axes(button_rect)
		self.button = Button(self.button_ax, self._button_label(initial))
		self.current = initial

		self.option_axes: list[plt.Axes] = []
		self.option_buttons: list[Button] = []
		self._build_option_buttons(options)

		self.button.on_clicked(self._handle_toggle)

	def _button_label(self, value: str) -> str:
		return f"{self.title}: {value}"

	def _build_option_buttons(self, options: list[str]) -> None:
		menu_x, menu_y, menu_w, menu_h = self.menu_rect
		option_height = menu_h / max(len(options), 1)
		for idx, option in enumerate(options):
			ax_y = menu_y + menu_h - option_height * (idx + 1)
			option_ax = self.fig.add_axes([menu_x, ax_y, menu_w, option_height])
			option_button = Button(option_ax, option)
			option_button.on_clicked(lambda _event, selected=option: self._handle_select(selected))
			option_button.set_active(False)
			option_ax.set_visible(False)
			self.option_axes.append(option_ax)
			self.option_buttons.append(option_button)

	def _handle_toggle(self, _event) -> None:
		self.on_toggle(self)

	def _handle_select(self, selected: str) -> None:
		self.current = selected
		self.button.label.set_text(self._button_label(selected))
		self.hide_menu()
		self.on_select(selected)
		self.fig.canvas.draw_idle()

	def show_menu(self) -> None:
		for option_ax, option_button in zip(self.option_axes, self.option_buttons):
			option_ax.set_visible(True)
			option_button.set_active(True)

	def hide_menu(self) -> None:
		for option_ax, option_button in zip(self.option_axes, self.option_buttons):
			option_ax.set_visible(False)
			option_button.set_active(False)

	def toggle_menu(self) -> None:
		if self.is_menu_visible():
			self.hide_menu()
		else:
			self.show_menu()

	def is_menu_visible(self) -> bool:
		if not self.option_axes:
			return False
		return bool(self.option_axes[0].get_visible())

	def set_current(self, selected: str) -> None:
		if selected not in self.options:
			raise ValueError(f"Option '{selected}' not available in selector {self.title}.")
		self.current = selected
		self.button.label.set_text(self._button_label(selected))

	def set_title(self, title: str) -> None:
		self.title = title
		self.button.label.set_text(self._button_label(self.current))

	def set_options(self, options: list[str], initial: str) -> None:
		if initial not in options:
			raise ValueError(f"Initial option '{initial}' not in options for {self.title}.")

		self.hide_menu()
		for option_ax in self.option_axes:
			option_ax.remove()

		self.options = options
		self.option_axes = []
		self.option_buttons = []
		self._build_option_buttons(options)
		self.set_current(initial)


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
) -> tuple[CoreSeries, CoreSeries, list[Tiepoint], tuple[float, float], tuple[float, float], str]:
	pair_prefix, paired = load_tiepoint_pairs_from_big_table(core_a, core_b)
	tie_a = [depth_a for depth_a, _, _, _ in paired]
	tie_b = [depth_b for _, depth_b, _, _ in paired]

	if not paired:
		raise ValueError(f"No tiepoints found in {BIG_TABLE_PATH} for pair prefix {pair_prefix}.")

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

	return series_a, series_b, paired, window_a, window_b, pair_prefix


def plot_pair(core_a: str, core_b: str) -> None:
	series_a, series_b, all_pairs, window_a, window_b, pair_prefix = build_view(core_a, core_b)
	data_bounds_a = (float(series_a.data["depth"].min()), float(series_a.data["depth"].max()))
	data_bounds_b = (float(series_b.data["depth"].min()), float(series_b.data["depth"].max()))
	reference_colors = build_reference_color_map(all_pairs)
	available_files_a = list_core_data_files(core_a)
	available_files_b = list_core_data_files(core_b)
	file_map_a = {path.name: path for path in available_files_a}
	file_map_b = {path.name: path for path in available_files_b}
	current_core_a = core_a.strip().lower()
	current_core_b = core_b.strip().lower()

	fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
	plt.subplots_adjust(left=0.08, right=0.80, top=0.93, bottom=0.22, hspace=0.16)

	state: dict[str, object] = {
		"window_a": window_a,
		"window_b": window_b,
		"base_window_a": window_a,
		"base_window_b": window_b,
		"data_bounds_a": data_bounds_a,
		"data_bounds_b": data_bounds_b,
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

	def synced_window_a_from_window_b(current_window_b: tuple[float, float]) -> tuple[float, float]:
		pairs_in_b = [
			(depth_a, depth_b)
			for depth_a, depth_b, _, _ in all_pairs
			if current_window_b[0] <= depth_b <= current_window_b[1]
		]

		if len(pairs_in_b) < 2:
			raise ValueError(
				"Sync requires at least 2 tiepoints in Core B depth range."
			)

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

		bounds_a = state["data_bounds_a"]
		assert isinstance(bounds_a, tuple)
		return clamp_window((a_min, a_max), bounds_a)

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
		visible = pairs_in_bottom_window(all_pairs, current_window_b)
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
			for depth_a, depth_b, label, reference in visible:
				tie_color = reference_colors.get(reference, "crimson")
				if SHOW_VERTICAL_TIE_LINES:
					ax_a.axvline(depth_a, color="gray", lw=2.5, alpha=0.15, zorder=1)
					ax_b.axvline(depth_b, color="gray", lw=2.5, alpha=0.15, zorder=1)

				ax_a.hlines(
					y=y_line_a,
					xmin=depth_a - seg_half_width_a,
					xmax=depth_a + seg_half_width_a,
					color=tie_color,
					lw=1.2,
					zorder=5,
				)
				ax_b.hlines(
					y=y_line_b,
					xmin=depth_b - seg_half_width_b,
					xmax=depth_b + seg_half_width_b,
					color=tie_color,
					lw=1.2,
					zorder=5,
				)

				ax_a.text(
					depth_a,
					y_text_a,
					str(label),
					color=tie_color,
					fontsize=8,
					ha="center",
					va="bottom",
					zorder=6,
				)
				ax_b.text(
					depth_b,
					y_text_b,
					str(label),
					color=tie_color,
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
					color=tie_color,
					lw=0.8,
					alpha=0.55,
					zorder=4,
				)
				fig.add_artist(connector)
				state["connectors"].append(connector)

			visible_references = sorted({reference for _, _, _, reference in visible})
			if visible_references:
				legend_handles = [
					Line2D([0], [0], color=reference_colors.get(reference, "crimson"), lw=2, label=reference)
					for reference in visible_references
				]
				ncol = 1 if len(legend_handles) <= 10 else 2
				ax_a.legend(
					handles=legend_handles,
					title="Reference",
					loc="upper left",
					bbox_to_anchor=(1.01, 1.0),
					borderaxespad=0.0,
					framealpha=0.9,
					fontsize=8,
					title_fontsize=9,
					ncol=ncol,
				)

			title_suffix = f" ({len(visible)} tiepoints in bottom range)"
		else:
			ax_a.text(
				0.5,
				0.03,
				f"Tiepoints hidden: {len(visible)} in bottom range (> {MAX_DRAWN_TIEPOINTS})",
				transform=ax_a.transAxes,
				ha="center",
				va="bottom",
				color="crimson",
				fontsize=10,
			)
			title_suffix = (
				f" ({len(visible)} tiepoints in bottom range; hidden because > {MAX_DRAWN_TIEPOINTS})"
			)

		fig.suptitle(
			f"{series_a.canonical_name} vs {series_b.canonical_name} tiepoints{title_suffix}",
			fontsize=13,
		)

		fig.canvas.draw_idle()

	def update_core_series(core_side: str, file_name: str) -> None:
		if core_side == "a":
			selected_path = file_map_a[file_name]
			target_series = series_a
			window_key = "window_a"
			bounds_key = "data_bounds_a"
			min_box = box_a_min
			max_box = box_a_max
		else:
			selected_path = file_map_b[file_name]
			target_series = series_b
			window_key = "window_b"
			bounds_key = "data_bounds_b"
			min_box = box_b_min
			max_box = box_b_max

		new_data = load_core_data(selected_path)
		target_series.file_path = selected_path
		target_series.data = new_data
		target_series.value_label = str(new_data.attrs.get("value_label", "value"))

		new_bounds = (float(new_data["depth"].min()), float(new_data["depth"].max()))
		current_window = state[window_key]
		assert isinstance(current_window, tuple)
		state[bounds_key] = new_bounds
		state[window_key] = clamp_window(current_window, new_bounds)

		updated_window = state[window_key]
		assert isinstance(updated_window, tuple)
		min_box.set_val(f"{updated_window[0]:g}")
		max_box.set_val(f"{updated_window[1]:g}")
		sync_boxes()
		redraw()

	dropdowns: list[DropdownSelector] = []
	selector_by_name: dict[str, DropdownSelector] = {}

	def toggle_dropdown(active_dropdown: DropdownSelector) -> None:
		for dropdown in dropdowns:
			if dropdown is active_dropdown:
				dropdown.toggle_menu()
			else:
				dropdown.hide_menu()
		fig.canvas.draw_idle()

	def update_core_pair(new_core_a: str, new_core_b: str) -> bool:
		nonlocal series_a, series_b, all_pairs, pair_prefix
		nonlocal reference_colors, available_files_a, available_files_b, file_map_a, file_map_b
		nonlocal current_core_a, current_core_b

		try:
			new_series_a, new_series_b, new_pairs, new_window_a, new_window_b, new_pair_prefix = build_view(new_core_a, new_core_b)
			new_available_files_a = list_core_data_files(new_core_a)
			new_available_files_b = list_core_data_files(new_core_b)
		except Exception as exc:
			fig.suptitle(str(exc), fontsize=12, color="crimson")
			fig.canvas.draw_idle()
			return False

		series_a = new_series_a
		series_b = new_series_b
		all_pairs = new_pairs
		pair_prefix = new_pair_prefix
		reference_colors = build_reference_color_map(all_pairs)
		available_files_a = new_available_files_a
		available_files_b = new_available_files_b
		file_map_a = {path.name: path for path in available_files_a}
		file_map_b = {path.name: path for path in available_files_b}
		current_core_a = new_core_a
		current_core_b = new_core_b

		new_bounds_a = (float(series_a.data["depth"].min()), float(series_a.data["depth"].max()))
		new_bounds_b = (float(series_b.data["depth"].min()), float(series_b.data["depth"].max()))
		state["data_bounds_a"] = new_bounds_a
		state["data_bounds_b"] = new_bounds_b
		state["window_a"] = new_window_a
		state["window_b"] = new_window_b
		state["base_window_a"] = new_window_a
		state["base_window_b"] = new_window_b

		box_a_min.label.set_text(f"{series_a.canonical_name} min")
		box_a_max.label.set_text(f"{series_a.canonical_name} max")
		box_b_min.label.set_text(f"{series_b.canonical_name} min")
		box_b_max.label.set_text(f"{series_b.canonical_name} max")

		file_selector_a = selector_by_name["file_a"]
		file_selector_b = selector_by_name["file_b"]
		file_selector_a.set_title(f"{series_a.canonical_name} file")
		file_selector_b.set_title(f"{series_b.canonical_name} file")
		file_selector_a.set_options([path.name for path in available_files_a], series_a.file_path.name)
		file_selector_b.set_options([path.name for path in available_files_b], series_b.file_path.name)

		sync_boxes()
		redraw()
		return True

	def on_core_select(core_side: str, selected_core: str) -> None:
		old_core_a = current_core_a
		old_core_b = current_core_b

		if core_side == "a":
			next_core_a = selected_core
			next_core_b = current_core_b
		else:
			next_core_a = current_core_a
			next_core_b = selected_core

		if not update_core_pair(next_core_a, next_core_b):
			if core_side == "a":
				selector_by_name["core_a"].set_current(old_core_a)
			else:
				selector_by_name["core_b"].set_current(old_core_b)
			fig.canvas.draw_idle()

	initial_file_a = series_a.file_path.name
	initial_file_b = series_b.file_path.name
	if initial_file_a not in file_map_a:
		initial_file_a = available_files_a[0].name
	if initial_file_b not in file_map_b:
		initial_file_b = available_files_b[0].name

	dropdown_core_a = DropdownSelector(
		fig=fig,
		button_rect=[0.82, 0.78, 0.16, 0.04],
		menu_rect=[0.82, 0.82, 0.16, 0.16],
		title="Core A",
		options=list(CORE_SELECTION_KEYS),
		initial=current_core_a,
		on_select=lambda selected: on_core_select("a", selected),
		on_toggle=toggle_dropdown,
	)
	dropdown_core_b = DropdownSelector(
		fig=fig,
		button_rect=[0.82, 0.58, 0.16, 0.04],
		menu_rect=[0.82, 0.62, 0.16, 0.16],
		title="Core B",
		options=list(CORE_SELECTION_KEYS),
		initial=current_core_b,
		on_select=lambda selected: on_core_select("b", selected),
		on_toggle=toggle_dropdown,
	)

	dropdown_a = DropdownSelector(
		fig=fig,
		button_rect=[0.82, 0.38, 0.16, 0.04],
		menu_rect=[0.82, 0.42, 0.16, 0.16],
		title=f"{series_a.canonical_name} file",
		options=[path.name for path in available_files_a],
		initial=initial_file_a,
		on_select=lambda selected: update_core_series("a", selected),
		on_toggle=toggle_dropdown,
	)
	dropdown_b = DropdownSelector(
		fig=fig,
		button_rect=[0.82, 0.18, 0.16, 0.04],
		menu_rect=[0.82, 0.22, 0.16, 0.16],
		title=f"{series_b.canonical_name} file",
		options=[path.name for path in available_files_b],
		initial=initial_file_b,
		on_select=lambda selected: update_core_series("b", selected),
		on_toggle=toggle_dropdown,
	)
	dropdowns.extend([dropdown_core_a, dropdown_core_b, dropdown_a, dropdown_b])
	selector_by_name["core_a"] = dropdown_core_a
	selector_by_name["core_b"] = dropdown_core_b
	selector_by_name["file_a"] = dropdown_a
	selector_by_name["file_b"] = dropdown_b

	def apply_windows(_event=None) -> None:
		try:
			bounds_a = state["data_bounds_a"]
			bounds_b = state["data_bounds_b"]
			assert isinstance(bounds_a, tuple)
			assert isinstance(bounds_b, tuple)
			new_window_a = parse_window(box_a_min, box_a_max, bounds_a, series_a.canonical_name)
			new_window_b = parse_window(box_b_min, box_b_max, bounds_b, series_b.canonical_name)
		except ValueError as exc:
			fig.suptitle(str(exc), fontsize=12, color="crimson")
			fig.canvas.draw_idle()
			return

		state["window_a"] = new_window_a
		state["window_b"] = new_window_b
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
	print(f"Tiepoint table: {BIG_TABLE_PATH}")
	print(f"Tiepoint pair prefix: {pair_prefix}")
	print(f"Core A window: [{window_a[0]:g}, {window_a[1]:g}]")
	print(f"Core B window: [{window_b[0]:g}, {window_b[1]:g}]")
	print(f"Tiepoints loaded: {len(all_pairs)}")

	plt.show()


def main() -> None:
	core_a, core_b = CORE_PAIR
	plot_pair(core_a, core_b)


if __name__ == "__main__":
	main()
