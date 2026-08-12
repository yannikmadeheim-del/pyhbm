"""
Derive this example's substructure descriptor from the cubic-spring one.

This example runs on a COLLOCATED copy of coupling_example.xlsx: the Grouping==10
rows of Channels_<X> and Impacts_<X> are both the union of the two sheets' rows,
appended in that order (original channels first in Channels_<X>, original impacts
first in Impacts_<X>). That is what makes Tu and Tf transposes of one another, so
RBE_average and the pyFBS VPT observe and load the virtual point through the same
DoFs -- the set RBE_rigid already ties rigidly -- and the three become comparable.

The descriptor only mirrors those workbook rows, so it needs no Ansys run: the new
``interface_channels`` / ``interface_impacts`` are exactly the old two lists
concatenated, in the order the sheets were edited. ``interface_node_ids`` is the
union either way and stays untouched, which is why "rbe2" / "rbe3" and the
interface_method="descriptor" lookup are unaffected.

Re-run this whenever export_substructure_descriptor.py regenerates the source
descriptor. main.py's check_vpt_rows compares the result against the local
workbook row by row and will fail loudly if the two ever drift apart.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "testbench_cubicSpring_CB" / "substructure_descriptor.json"
TARGET = HERE / "substructure_descriptor.json"

descriptor = json.loads(SOURCE.read_text())
descriptor["example"] = "testbench_joint_comparison (collocated interface rows)"

for name, sub in descriptor["substructures"].items():
    chn, imp = sub["interface_channels"], sub["interface_impacts"]
    sub["interface_channels"] = chn + imp          # Channels_<X>: channels, then impacts
    sub["interface_impacts"] = imp + chn           # Impacts_<X>:  impacts, then channels
    print(f"[{name}] {len(chn)} channels + {len(imp)} impacts -> "
          f"{len(chn) + len(imp)} collocated rows on each side, "
          f"{len(sub['interface_node_ids'])} interface nodes (unchanged)")

TARGET.write_text(json.dumps(descriptor, indent=2))
print(f"\nwrote {TARGET}")
