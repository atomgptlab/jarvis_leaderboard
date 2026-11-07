import json, warnings; warnings.filterwarnings("ignore")
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data
from alignn.ff.ff import AlignnAtomwiseCalculator
import jarvis.analysis.thermodynamics as thermo, os
refs = json.load(open("/tmp/matpes_unary_refs.json"))
uj = json.load(open(os.path.join(os.path.dirname(thermo.__file__), "unary.json")))
d3 = {e["jid"]: e for e in data("dft_3d")}
if "Sr" in uj and uj["Sr"]["jid"] in d3:
    refs["Sr"] = {"jid": uj["Sr"]["jid"], "atoms": d3[uj["Sr"]["jid"]]["atoms"]}
calc = AlignnAtomwiseCalculator(path="/home/kamalch/alignn2026/matpes_smooth/results",
                                force_mult_natoms=False, force_multiplier=1, stress_wt=1)
unary = {}
for el, r in refs.items():
    at = Atoms.from_dict(r["atoms"]).ase_converter(); at.calc = calc
    unary[el] = {"jid": r["jid"], "energy": round(float(at.get_potential_energy()) / len(at), 6)}
    print(f"  {el}: {unary[el]['energy']}", flush=True)
json.dump(unary, open("/home/kamalch/alignn2026/matpes_smooth_unary.json", "w"), indent=2)
print(f"DONE {len(unary)} elements (single-point)", flush=True)
