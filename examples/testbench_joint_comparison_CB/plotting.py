"""
Reading and plotting of physical-solution CSVs -- shared by main.py and
plot_comparison.py.

This is deliberately the same reader and the same label logic as the pyFBS
testbench_joint_comparison example's plotting.py: the Craig-Bampton runs happen
on the machine that holds the FE data, the FBS runs on another, and neither
repository can reach the other's results. Having the module on both sides means
a comparison folder produces the same curves, the same automatic labels and the
same plot_styles.csv wherever it is opened. Both CSV conventions are read --
``x_rel_*`` (FBS, gap exported directly) and ``A_vp_*``/``B_vp_*`` (Craig-
Bampton, gap formed as VP_A - VP_B) -- so the two pipelines overlay.

Three layers:

  * :func:`read_result` -- one CSV into the four plotted curves plus the CONFIG
    that produced it. Only the needed columns are read (these files get large),
    and the harmonic list comes from the file itself, never from a constant here.

  * :func:`auto_labels` -- legend labels derived from the config entries that
    actually DIFFER across the loaded files, e.g. ``linear+cubic | alpha=1e+08``.

  * :func:`read_style_spec` -- the comparison-folder infrastructure: a folder of
    CSVs plus a ``plot_styles.csv`` whose ROW ORDER is the plot and legend order
    and whose empty cells fall back to auto label / cycled style. The spec is
    written on first use, so a fresh folder works immediately and can be edited
    afterwards. Several such folders coexist side by side.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_T = 1024                          # common dense time grid for the period maxima
TRANS = ("ux", "uy", "uz")          # translational gap DoFs
STYLE_COLUMNS = ("file", "label", "color", "linestyle", "linewidth")
SKIP_CSV = ("manifest.csv",)        # study bookkeeping, not a result branch

DEFAULT_COLORS = ["#2ca02c", "black", "#d62728", "#1f77b4", "#9467bd", "#ff7f0e",
                  "#17becf", "#8c564b", "#e377c2", "#7f7f7f"]
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_LINEWIDTH = 1.4


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_header(path):
    """The leading ``#`` comment lines of a result CSV, without the marker."""
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            lines.append(line[1:].strip())
    return lines


def read_config(path):
    """The CONFIG dict recorded in the ``config_json`` header line, or None for
    files written without one (e.g. the older reference exports)."""
    for line in read_header(path):
        if line.startswith("config_json:"):
            return json.loads(line[len("config_json:"):].strip())
    return None


def _harmonics(config, columns):
    """Exported harmonics: from the file's own config, else from the column
    names. Hard-coding a list here would silently misread any run with a
    different harmonic set."""
    if config and "harmonics" in config:
        return [int(h) for h in config["harmonics"]]
    found = {int(m.group(1)) for m in
             (re.match(r"re_h(\d+)_", c) for c in columns) if m}
    if not found:
        raise KeyError(f"no re_h<h>_<dof> columns found -- not a "
                       f"physical-solution CSV?")
    return sorted(found)


def _output_channel(header, columns, harmonics):
    """Plotted output channel, taken from the header line the exporters write."""
    match = re.search(r"uout \(plotted output channel\) = (\S+)",
                      "\n".join(header))
    for name in ([match.group(1)] if match else []) + ["A_S1X"]:
        if f"re_h{harmonics[0]}_{name}" in columns:
            return name
    raise KeyError("output-channel columns not found; the header names "
                   f"{match.group(1) if match else 'nothing'}")


def _amplitudes(df, dof, harmonics):
    """Complex physical amplitudes a_h of ``dof``, shape (n_points, n_harmonics)."""
    re_ = df[[f"re_h{h}_{dof}" for h in harmonics]].to_numpy()
    im_ = df[[f"im_h{h}_{dof}" for h in harmonics]].to_numpy()
    return re_ + 1j * im_


def _period_signal(amps, harmonics):
    """u(theta) = Re( sum_h a_h e^{i h theta} ) on N_T samples of one period.
    Matmul contracts the trailing harmonic axis: (..., n_h) -> (..., N_T).
    Recomputing on a common grid makes files with different AFT sample counts
    comparable."""
    theta = np.linspace(0.0, 2.0 * np.pi, N_T, endpoint=False)
    return np.real(amps @ np.exp(1j * np.outer(harmonics, theta)))


def read_result(path):
    """
    One physical-solution CSV into the plotted curves.

    :returns: dict with ``f`` [Hz], ``out_max`` = max_t |u_out(t)|, ``out_h1`` =
        |a_1(u_out)|, ``gap_max`` = max_t ||x_rel(t)||_2 and ``gap_h1`` =
        ||a_1(x_rel)||_2 over the three translational gap DoFs, plus ``config``.
    """
    path = Path(path)
    config = read_config(path)
    header = read_header(path)
    columns = list(pd.read_csv(path, comment="#", nrows=0).columns)  # names only
    harmonics = _harmonics(config, columns)
    h0 = harmonics[0]

    out = _output_channel(header, columns, harmonics)
    if f"re_h{h0}_x_rel_{TRANS[0]}" in columns:
        gap_dofs = [f"x_rel_{d}" for d in TRANS]        # gap exported directly
    elif f"re_h{h0}_A_vp_{TRANS[0]}" in columns:
        gap_dofs = [f"{s}_vp_{d}" for s in ("A", "B") for d in TRANS]
    else:
        raise KeyError(f"{path.name}: neither x_rel_* nor A_vp_*/B_vp_* harmonic "
                       f"columns found -- not a physical-solution CSV?")

    wanted = {"freq_hz"} | {f"{p}_h{h}_{d}" for d in [out] + gap_dofs
                            for h in harmonics for p in ("re", "im")}
    df = pd.read_csv(path, comment="#", usecols=lambda c: c in wanted)
    missing = wanted - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} lacks expected columns, e.g. "
                       f"{sorted(missing)[:3]}")

    a_out = _amplitudes(df, out, harmonics)                          # (n, n_h)
    if gap_dofs[0].startswith("x_rel"):
        a_gap = np.stack([_amplitudes(df, f"x_rel_{d}", harmonics) for d in TRANS],
                         axis=1)
    else:                                   # Craig-Bampton style: gap = VP_A - VP_B
        a_gap = np.stack([_amplitudes(df, f"A_vp_{d}", harmonics)
                          - _amplitudes(df, f"B_vp_{d}", harmonics) for d in TRANS],
                         axis=1)                                     # (n, 3, n_h)

    sig_out = _period_signal(a_out, harmonics)                       # (n, N_T)
    sig_gap = _period_signal(a_gap, harmonics)                       # (n, 3, N_T)
    i_h1 = harmonics.index(1) if 1 in harmonics else 0
    return dict(
        f       = df["freq_hz"].to_numpy(),
        out_max = np.abs(sig_out).max(axis=1),
        out_h1  = np.abs(a_out[:, i_h1]),
        gap_max = np.linalg.norm(sig_gap, axis=1).max(axis=1),
        gap_h1  = np.linalg.norm(a_gap[:, :, i_h1], axis=1),
        config  = config,
    )


# ---------------------------------------------------------------------------
# Automatic legend labels
# ---------------------------------------------------------------------------

def _fmt(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _flatten(config):
    """Config as flat ``key -> string`` pairs. Joint fields become dotted keys
    (``linear.k``); a repeated joint type gets a running number (``linear2.k``)
    so two springs of the same type stay distinguishable."""
    flat, seen = {}, {}
    for spec in config.get("joints", []):
        kind = spec.get("type", "?")
        n = seen.get(kind, 0)
        seen[kind] = n + 1
        prefix = kind if n == 0 else f"{kind}{n + 1}"
        for key, value in spec.items():
            if key != "type":
                flat[f"{prefix}.{key}"] = _fmt(value)
    flat["joints"] = "+".join(dict.fromkeys(spec.get("type", "?")
                                            for spec in config.get("joints", [])))
    for key, value in config.items():
        if key != "joints":
            flat[key] = _fmt(value)
    return flat


def auto_labels(configs, fallback_names):
    """
    Legend labels built ONLY from the config entries that differ across the
    loaded files, e.g. ``linear+cubic | alpha=1e+08 | F0=200``. Falls back to
    ``fallback_names`` (normally the file stems) when a config is missing or all
    configs are identical.

    Top-level config keys are compared across ALL files, with ``"-"`` standing
    in where a file has no such key: solvers with different schemas (an FBS run
    has ``frf_source``, a Craig-Bampton run has ``condensation``) are then still
    told apart instead of falling back to the file names. Joint element keys --
    the dotted ones, ``linear.k``, ``cubic2.alpha`` -- keep the all-present rule:
    a missing joint field means the element itself is absent, which the leading
    joint-type slug already says.
    """
    if any(c is None for c in configs) or len(configs) < 2:
        return list(fallback_names)

    flats = [_flatten(c) for c in configs]
    keys = [k for k in dict.fromkeys(k for flat in flats for k in flat)
            if "." not in k or all(k in flat for flat in flats)]
    values = [{k: flat.get(k, "-") for k in keys} for flat in flats]
    differing = [k for k in keys if len({v[k] for v in values}) > 1]
    if not differing:
        return list(fallback_names)
    differing.sort(key=lambda k: (k != "joints", keys.index(k)))

    # show the short field name where it is unambiguous ("alpha", not "cubic.alpha")
    tails = [k.rsplit(".", 1)[-1] for k in differing]
    display = {k: (t if tails.count(t) == 1 else k)
               for k, t in zip(differing, tails)}

    labels = []
    for vals in values:
        parts = [vals[k] if k == "joints" else f"{display[k]}={vals[k]}"
                 for k in differing]
        labels.append(" | ".join(parts))
    return labels


# ---------------------------------------------------------------------------
# Comparison-folder style spec
# ---------------------------------------------------------------------------

def _write_style_spec(spec_path, names, auto):
    with open(spec_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(STYLE_COLUMNS)
        for i, name in enumerate(names):
            writer.writerow([name, auto[name],
                             DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                             DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)],
                             DEFAULT_LINEWIDTH])


def read_style_spec(folder, style_file="plot_styles.csv"):
    """
    Resolve one comparison folder into the curves to plot, in plot order.

    ``<folder>/<style_file>`` has the columns file,label,color,linestyle,linewidth
    and its row order is the plot and legend order; empty cells fall back to the
    auto label and a cycled default style. CSVs in the folder that the spec does
    not name are appended at the end (with a printed note), and a spec row naming
    a file that is not in the folder is an error. If the spec does not exist it
    is written from the folder contents first.

    :returns: list of ``(path, label, style_kwargs)``.
    """
    folder = Path(folder)
    spec_path = folder / style_file
    # Path.glob returns empty on a missing directory instead of raising, which
    # would surface below as a misleading "no result CSVs" message.
    if not folder.is_dir():
        raise NotADirectoryError(
            f"{folder} is not a folder -- COMPARE_DIR names a comparison FOLDER "
            f"inside the example folder, not a CSV file")
    files = sorted(p for p in folder.glob("*.csv")
                   if p.name != style_file and p.name not in SKIP_CSV)
    if not files:
        raise FileNotFoundError(f"no result CSVs in {folder} (looked for *.csv "
                                f"besides {style_file})")

    auto = dict(zip([p.name for p in files],
                    auto_labels([read_config(p) for p in files],
                                [p.stem for p in files])))

    if spec_path.exists():
        with open(spec_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        entries = []
        for line_no, row in enumerate(rows, start=2):      # 1 = header row
            name = (row.get("file") or "").strip()
            if not name:
                continue
            if not (folder / name).exists():
                raise FileNotFoundError(
                    f"{spec_path}, line {line_no}: '{name}' is not in {folder}")
            entries.append((name, row))
        listed = {name for name, _ in entries}
        extra = [p.name for p in files if p.name not in listed]
        if extra:
            print(f"note: not listed in {style_file}, appended at the end: "
                  f"{', '.join(extra)}")
        entries += [(name, {}) for name in extra]
    else:
        entries = [(p.name, {}) for p in files]
        _write_style_spec(spec_path, [name for name, _ in entries], auto)
        print(f"created {spec_path} from the folder contents -- edit it to "
              f"control order, labels and styles")

    out = []
    for i, (name, row) in enumerate(entries):
        width = (row.get("linewidth") or "").strip()
        out.append((
            folder / name,
            (row.get("label") or "").strip() or auto[name],
            dict(color=(row.get("color") or "").strip()
                       or DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                 linestyle=(row.get("linestyle") or "").strip()
                           or DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)],
                 linewidth=float(width) if width else DEFAULT_LINEWIDTH),
        ))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def overlay(curves, labels, styles, key_max, key_h1, title, ylabel_max,
            ylabel_h1, xlim=None):
    """One window, 2x1 shared-x semilogy: period maximum on top, 1st-harmonic
    amplitude below; ``styles[i]`` are the matplotlib line kwargs of curve i."""
    # A legend with many entries is taller than the axes. Drawn inside them,
    # tight_layout honours it by shrinking the plots to slivers, so it goes
    # beside the axes instead and constrained_layout budgets the space; the
    # figure widens and the legend gains columns as the curve count grows.
    ncol = 1 + len(curves) // 14
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9 + 2.4 * ncol, 8),
                                         sharex=True, layout="constrained")
    for curve, label, style in zip(curves, labels, styles):
        ax_top.semilogy(curve["f"], curve[key_max], label=label, **style)
        ax_bot.semilogy(curve["f"], curve[key_h1], label=label, **style)
    ax_top.set_title(title)
    ax_top.set_ylabel(ylabel_max)
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=ncol,
                  fontsize="small", frameon=False)
    ax_bot.set_ylabel(ylabel_h1)
    ax_bot.set_xlabel("Frequency [Hz]")
    if xlim is not None:
        ax_bot.set_xlim(*xlim)
    for ax in (ax_top, ax_bot):
        ax.grid(True, which="both", alpha=0.3)
    return fig


def nfrc_figure(curves, labels, styles, title="Output channel", xlim=None):
    """Forced-response curves of the output channel."""
    return overlay(curves, labels, styles, "out_max", "out_h1", title,
                   "max_t |u_out(t)|  [m]", "|a_1(u_out)|  [m]", xlim)


def gap_figure(curves, labels, styles,
               title="Interface gap x_rel (translations)", xlim=None):
    """Forced-response curves of the translational interface gap the joint acts on."""
    return overlay(curves, labels, styles, "gap_max", "gap_h1", title,
                   "max_t ||x_rel(t)||_2  [m]", "||a_1(x_rel)||_2  [m]", xlim)
