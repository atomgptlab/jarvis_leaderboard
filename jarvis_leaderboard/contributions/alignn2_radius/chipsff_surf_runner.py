import sys, json, os, subprocess
d=sys.argv[1]
mill=json.load(open(f"{d}/millers.json"))
RC="/home/kamalch/alignn2026/chipsff/chipsff/run_chipsff.py"
for jid, millers in mill.items():
    wd=f"{d}/{jid}"; os.makedirs(wd, exist_ok=True)
    inp={"jid":jid,"calculator_type":"alignn_ff","chemical_potentials_file":"/home/kamalch/alignn2026/ff_bakeoff/chemical_potentials.json",
         "properties_to_calculate":["relax_structure","calculate_ev_curve","analyze_surfaces"],
         "bulk_relaxation_settings":{"filter_type":"ExpCellFilter","relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":False}},
         "surface_settings":{"indices_list":millers,"layers":4,"vacuum":18,
             "relaxation_settings":{"fmax":0.05,"steps":200,"constant_volume":True},"filter_type":"ExpCellFilter"},
         "use_conventional_cell":True,
         "calculator_settings":{"alignn_ff":{"path":"/home/kamalch/alignn2026/matpes_smooth/results","model_filename":"best_model.pt","stress_wt":1}}}
    json.dump(inp, open(f"{wd}/input.json","w"))
    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", BROWSER="/bin/true", OMP_NUM_THREADS="3")
    subprocess.run(["/home/kamalch/miniforge3/envs/chipsff/bin/python", RC, "--input_file", "input.json"],
                   cwd=wd, env=env, stdout=open(f"{wd}/run.log","w"), stderr=subprocess.STDOUT)
print(f"CHUNK {d} DONE")
