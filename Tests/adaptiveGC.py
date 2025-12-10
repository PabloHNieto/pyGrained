from pyGrained.models.AdaptiveCG import AdaptiveCG
from pyGrained.models.SBCG import SBCG

pdb_file1 = "./data/lizard_3tower.pdb"  # Replace with your PDB file path
pdb_file2 = "./data/_mcps/6qi5_mcp.pdb"  # Replace with your PDB file path

params = {
    "resolution": 250, # Desired number of beads
    # "nBeads": 20, # Desired number of beads
    "sigma": .20  # Width parameter for Gaussian
}

params2 = {
    "parameters": {"resolution":200, 
                   "steps":1000, 
                   "bondsModel":{"name":"count"},
                   "nativeContactsModel":{"name":"CA", "parameters":{
                       "epsilon":1.0,
                       "D":1.0
                   }},
                   }, 
    "SASA": False
}

model = AdaptiveCG("test", pdb_file1, params=params)
model.view()
# model.write_pdb()
# model = SBCG("test", pdb_file2, params=params2)
# model = AdaptiveCG("test", pdb_file, params=params)
# model.view()

# model = AdaptiveCG(pdb_file, n_beads, sigma)
# R_opt, chi_opt = model.optimize(max_iter=3000)

# model.export_beads_pdb("cg_beads.pdb")
# model.write_chimerax_beads_script(pdb_file, R_opt, out_script="show_beads.cxc", view=True)

# pdb_file = "/home/pablo/Lizard_MD/structures/au_mcps/6qi5_mcp.pdb"  # Replace with your PDB file path
# unique_mols = model.compute_unique_molecules()