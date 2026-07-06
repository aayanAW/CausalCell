import sys; import numpy as np; from pathlib import Path
ROOT=Path("/Users/aayanalwani/virtual cells"); sys.path.insert(0,str(ROOT))
from scripts.run_eval_local import build_cb_input, run_gies
from scripts.a3_combinations import score
import scripts.a3_mechanism_discriminator as D
import json
P=D.build_panel(); genes=P["genes"]; anch=P["anch"]; add_doubles=P["add_doubles"]
rng=np.random.default_rng(2024); gl=list(genes); seed=42
ci=rng.choice(P["ctrl_raw"].shape[0],size=min(800,P["ctrl_raw"].shape[0]),replace=False)
ctrl_X=P["ctrl_raw"][ci]; rm,rl=[],[]
for d in add_doubles:
    gc=P["dbl_cells_raw"][d]
    if gc.shape[0]>100: gc=gc[rng.choice(gc.shape[0],100,replace=False)]
    rm.append(gc); rl+=[d]*gc.shape[0]
cb=build_cb_input(ctrl_X,rm,rl,genes)
edges=set(run_gies(cb,seed=seed,partition_size=20))
f1=score(edges,anch)["f1"]
floor=[]
for _ in range(50):
    shuf={(a, gl[rng.integers(len(gl))]) for a,b in anch}
    floor.append(score(edges,shuf)["f1"])
out=dict(real_doubles_F1=float(f1), perm_floor_mean=float(np.mean(floor)),
         perm_floor_max=float(np.max(floor)), perm_floor_p95=float(np.percentile(floor,95)),
         n_perm=50, gt_edges=len(anch), real_edges=len(edges))
json.dump(out, open("results/hardening/a3_floor.json","w"), indent=2)
print(json.dumps(out))
