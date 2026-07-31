"""
Overlay every result CSV of one comparison folder -- no solver, no Ansys, no FE data.

Set COMPARE_DIR below; that is the only thing to change when switching between
studies. Several comparison folders live side by side in this example folder
(results/<study>/, compare_pipelines/, ...): create a new one, copy or move the
CSVs to compare into it, and point COMPARE_DIR at it. On the first run a
plot_styles.csv is written from the folder contents with automatic labels; edit
its rows afterwards to control legend text, colour, line style, line width and
-- through the row order -- the plot and legend order. CSVs that the spec does
not list are appended at the end with a printed note.

Craig-Bampton and FBS exports mix freely: the gap comes from A_vp_* - B_vp_* for
the former and from x_rel_* for the latter, and the labels name the pipeline
through the ``solver`` config key. That is the point of having this script here
as well as in the pyFBS example -- the two sets of runs are produced on
different machines, and either one can plot the union.

Two figures: the output channel and the translational interface gap, each as
period maximum over 1st-harmonic amplitude.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

import plotting

COMPARE_DIR = "results/cubic_vs_amplitude_condensation"   # folder inside this example
STYLE_FILE  = "plot_styles.csv"       # style spec inside that folder
XLIM        = (200, 500)              # displayed frequency range [Hz]

HERE = Path(__file__).resolve().parent
folder = HERE / (sys.argv[1] if len(sys.argv) > 1 else COMPARE_DIR)

curves, labels, styles = [], [], []
for path, label, style in plotting.read_style_spec(folder, STYLE_FILE):
    print(f"loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...")
    curve = plotting.read_result(path)
    print(f"  {len(curve['f'])} points, "
          f"{curve['f'].min():.1f}..{curve['f'].max():.1f} Hz  ->  {label}")
    curves.append(curve)
    labels.append(label)
    styles.append(style)

plotting.nfrc_figure(curves, labels, styles,
                     title=f"{folder.name} -- output channel", xlim=XLIM)
plotting.gap_figure(curves, labels, styles,
                    title=f"{folder.name} -- interface gap (translations)",
                    xlim=XLIM)
plt.show()
