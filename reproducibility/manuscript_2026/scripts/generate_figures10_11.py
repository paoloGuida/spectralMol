from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors

out = Path("/tmp/spectralmol_fig10_11_20260915")
out.mkdir(parents=True, exist_ok=True)

generated = [
    dict(label="G1", role="generated", seed=8, docking_raw=-10.3, qed=0.914, sa=2.014, molwt=307.3, smiles="Cc1cc(C(F)(F)F)ccc1CCc1ccccc1C(N)=O", source="latest Saturn Table 8 run", decode_reason="THETA_GENERATED_HIT_NEIGHBORHOOD:OK:CHEAP_PRESELECT"),
    dict(label="G2", role="generated", seed=7, docking_raw=-10.3, qed=0.764, sa=2.204, molwt=311.7, smiles="N=C(Cl)c1ccccc1CCc1ccc(C(F)(F)F)cc1", source="latest Saturn Table 8 run", decode_reason="THETA_LATE_GENERATED_HIT_SITE_SCAN:OK:CHEAP_PRESELECT"),
    dict(label="G3", role="generated", seed=2, docking_raw=-10.2, qed=0.942, sa=2.103, molwt=294.3, smiles="NC(=O)c1ncccc1CCc1ccc(C(F)(F)F)cc1", source="latest Saturn Table 8 run", decode_reason="THETA_LATE_GENERATED_HIT_SITE_SCAN:OK:CHEAP_PRESELECT"),
    dict(label="G4", role="generated", seed=8, docking_raw=-10.2, qed=0.776, sa=2.153, molwt=291.3, smiles="CC(=N)c1ccccc1CCc1ccc(C(F)(F)F)cc1", source="latest Saturn Table 8 run", decode_reason="THETA_GENERATED_HIT_SITE_SCAN:OK:CHEAP_PRESELECT"),
    dict(label="G5", role="generated", seed=2, docking_raw=-10.1, qed=0.938, sa=2.020, molwt=295.3, smiles="O=C(O)c1ncccc1CCc1ccc(C(F)(F)F)cc1", source="latest Saturn Table 8 run", decode_reason="THETA_LATE_GENERATED_HIT_SITE_SCAN:OK:CHEAP_PRESELECT"),
    dict(label="G6", role="generated", seed=8, docking_raw=-10.1, qed=0.919, sa=1.828, molwt=293.3, smiles="NC(=O)c1ccccc1CCc1ccc(C(F)(F)F)cc1", source="latest Saturn Table 8 run", decode_reason="THETA_GENERATED_HIT_SITE_SCAN:OK:CHEAP_PRESELECT"),
]
references = [
    dict(label="R1", role="reference", name="ONC201", smiles="Cc1ccccc1CN1C(=O)C2=C(CCN(Cc3ccccc3)C2)N2CCN=C12", source="ONC201 / dordaviprone reference"),
    dict(label="R2", role="reference", name="ONC212", smiles="O=C1N(CC2=CC=C(C(F)(F)F)C=C2)C3=NCCN3C4=C1CN(CC5=CC=CC=C5)CC4", source="ONC212 reference"),
    dict(label="R3", role="reference", name="TR-107", smiles="N#Cc1cccc(CN2CC3=C(CC2)N=CN(Cc2ccc(Cl)cc2)C3=O)c1", source="TR-107 reference"),
]

all_mols = generated + references
for m in all_mols:
    mol = Chem.MolFromSmiles(m["smiles"])
    if mol is None:
        raise SystemExit(f"Could not parse SMILES for {m['label']}: {m['smiles']}")
    Chem.rdDepictor.Compute2DCoords(mol)
    m["mol"] = mol
    m["canonical_smiles"] = Chem.MolToSmiles(mol)

legends = []
for m in generated:
    legends.append(f"{m['label']} seed {m['seed']}\ndock {m['docking_raw']:.1f} kcal/mol\nQED {m['qed']:.3f}; SA {m['sa']:.2f}")
for m in references:
    legends.append(f"{m['label']} {m['name']}\nreference ClpP activator")
img = Draw.MolsToGridImage([m["mol"] for m in all_mols], molsPerRow=3, subImgSize=(430, 310), legends=legends, useSVG=False)
img.save(out / "Molecules.png")

features = [
    ("H-bond donor", lambda mol: rdMolDescriptors.CalcNumHBD(mol) > 0),
    ("H-bond acceptor", lambda mol: rdMolDescriptors.CalcNumHBA(mol) > 0),
    ("Aromatic ring", lambda mol: any(atom.GetIsAromatic() for atom in mol.GetAtoms())),
    ("Hydrophobic aryl/alkyl", lambda mol: any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())),
    ("Basic/positive N", lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;!$(NC=O)]")) or any(atom.GetFormalCharge() > 0 for atom in mol.GetAtoms())),
    ("Acidic/negative", lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")) or mol.HasSubstructMatch(Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1,OX1-]")) or any(atom.GetFormalCharge() < 0 for atom in mol.GetAtoms())),
    ("Halogen", lambda mol: mol.HasSubstructMatch(Chem.MolFromSmarts("[F,Cl,Br,I]"))),
]

def feature_vector(mol):
    return [bool(fn(mol)) for _, fn in features]

rows = []
matrix = []
row_labels = []
for g in generated:
    gv = feature_vector(g["mol"])
    for r in references:
        rv = feature_vector(r["mol"])
        union = sum(a or b for a, b in zip(gv, rv))
        inter = sum(a and b for a, b in zip(gv, rv))
        j = inter / union if union else 1.0
        row_labels.append(f"{g['label']} vs {r['name']}\nJ={j:.2f}")
        vals = []
        for (fname, _), gh, rh in zip(features, gv, rv):
            if gh and rh:
                state, val = "shared", 3
            elif gh and not rh:
                state, val = "generated_only", 2
            elif (not gh) and rh:
                state, val = "reference_only", 1
            else:
                state, val = "absent", 0
            vals.append(val)
            rows.append({"generated_label": g["label"], "generated_seed": g["seed"], "generated_smiles": g["canonical_smiles"], "generated_docking_raw": g["docking_raw"], "generated_qed": g["qed"], "generated_sa": g["sa"], "reference_label": r["label"], "reference_name": r["name"], "reference_smiles": r["canonical_smiles"], "feature": fname, "generated_has": int(gh), "reference_has": int(rh), "state": state, "feature_jaccard_for_pair": f"{j:.4f}"})
        matrix.append(vals)

matrix = np.asarray(matrix)
colors = np.array([[0.92,0.92,0.92], [0.56,0.42,0.82], [0.95,0.55,0.24], [0.20,0.63,0.38]])
fig_h = max(7.2, 0.39 * len(row_labels) + 2.0)
fig, ax = plt.subplots(figsize=(10.8, fig_h), dpi=300)
ax.imshow(colors[matrix], aspect="auto")
ax.set_xticks(np.arange(len(features)))
ax.set_xticklabels([f[0] for f in features], rotation=35, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8)
ax.set_title("Feature overlap between SpectralMol candidates and reference ClpP activators", fontsize=12, pad=12)
ax.set_xticks(np.arange(-.5, len(features), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)
handles = [plt.Rectangle((0,0),1,1,color=c) for c in colors]
ax.legend(handles, ["Absent in both", "Reference only", "Generated only", "Shared"], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out / "HeatMap.png", bbox_inches="tight")
plt.close(fig)

with (out / "figure10_selected_molecules.tsv").open("w", newline="") as f:
    fieldnames = ["label", "role", "name", "seed", "docking_raw", "qed", "sa", "molwt", "smiles", "canonical_smiles", "decode_reason", "source"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    for m in all_mols:
        writer.writerow({k: m.get(k, "") for k in fieldnames})
with (out / "figure11_pharmacophore_heatmap.tsv").open("w", newline="") as f:
    fieldnames = ["generated_label", "generated_seed", "generated_smiles", "generated_docking_raw", "generated_qed", "generated_sa", "reference_label", "reference_name", "reference_smiles", "feature", "generated_has", "reference_has", "state", "feature_jaccard_for_pair"]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
print("wrote", out)
for p in ["Molecules.png", "HeatMap.png", "figure10_selected_molecules.tsv", "figure11_pharmacophore_heatmap.tsv"]:
    print(p, (out / p).stat().st_size)
