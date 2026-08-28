"""
Write the pyFBS substructure descriptor of a measurement table in measurements/.

Adding a table to this example is two data files: the workbook itself, and the
``substructure_descriptor_<workbook>.json`` next to main.py that main.py reads
for the virtual-point frame, the excitation/output DoFs and the interface rows.
Drop the .xlsx into measurements/, run this, then run main.py.

The descriptor is NOT built here. pyFBS owns the mesh and the snapper, so it
stays the single source of truth for which FE nodes the interface rows land on;
this script only calls its exporter (testbench_cubicSpring/
export_substructure_descriptor.py) with the paths of this example and writes the
result where main.py looks for it. Everything crossing the boundary -- Ansys node
ids, the VP frame, the I/O DoFs -- is therefore produced by exactly the code that
produced the descriptors already in this folder.

    python make_descriptor.py                                # every table missing one
    python make_descriptor.py collocated_impact_locations    # these, overwriting

main.py's check_vpt_rows compares the result against the local workbook row by
row on every solve and fails loudly if the two ever drift apart.
"""

import json
import sys
from pathlib import Path

import measurements
# main.py owns the descriptor naming convention (and study.py likewise imports
# it), so the pattern is not repeated here
from main import descriptor_path

EXAMPLE = "testbench_joint_comparison"     # label stored in the descriptor


def write_descriptor(workbook):
    """
    Export the descriptor of one workbook and write it next to main.py.

    :param workbook: file name without extension, as in CONFIG["workbook"]
    :return: the path written
    """
    from pyfbs.nonlinearFBS.examples.testbench_cubicSpring \
        import export_substructure_descriptor as exporter

    # the exporter's __main__ would download the data here; that is a 292 MB
    # fetch into the pyFBS clone, so this reports instead of starting one
    if not exporter.FEM_DIR.exists():
        raise SystemExit(
            f"no FE data at {exporter.FEM_DIR} -- the lab_testbench folder is "
            f"gitignored on both sides; copy it in, or run "
            f"pyfbs.io.download_lab_testbench()")

    xlsx = measurements.resolve(workbook)
    descriptor = exporter.build_descriptor(xlsx, f"{EXAMPLE} ({workbook})")

    # The exporter asserts this in its own __main__ block, which calling
    # build_descriptor directly skips -- so it is repeated here rather than
    # dropped. A collocated table lists the SAME interface DoF set in both
    # sheets, which is what makes Tf equal Tu.T and RBE_average comparable to
    # pyFBS's VPT; a non-collocated one would reduce and solve without complaint
    # and only differ in the results.
    for name, sub in descriptor["substructures"].items():
        n_chn, n_imp = len(sub["interface_channels"]), len(sub["interface_impacts"])
        assert n_chn == n_imp, \
            f"[{name}] {n_chn} channel rows vs {n_imp} impact rows -- not collocated"
        print(f"acceptance ok: [{name}] {n_chn} collocated interface rows, "
              f"{len(sub['interface_node_ids'])} nodes")

    path = descriptor_path(workbook)
    path.write_text(json.dumps(descriptor, indent=2))
    return path


if __name__ == "__main__":
    names = sys.argv[1:]
    if names:                                   # named tables: rebuild them
        names = [Path(n).stem for n in names]
    else:                                       # bare call: fill in what is missing
        names = [n for n in measurements.available()
                 if not descriptor_path(n).is_file()]
        if not names:
            raise SystemExit("every table in measurements/ already has a "
                             "descriptor; name one to rebuild it")

    for name in names:
        print(f"\n=== {name} ===")
        print(f"wrote {write_descriptor(name)}")
