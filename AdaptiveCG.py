from .. import CoarseGrainedBase

import numpy as np
from Bio.PDB import PDBParser
from sklearn.cluster import KMeans
from copy import deepcopy

import logging

class ChainAdapticeCG:
    def __init__(self, n_beads:int, 
                 coords:np.ndarray, 
                 masses:np.ndarray, 
                 R_init:np.ndarray | None=None, sigma:float=2.0):
        self.coords = coords  # (N,3)
        self.sigma = sigma
        self.masses = masses  # (N,)

        self.logger = logging.getLogger(f"pyGrained")

        if R_init is not None:
            self.R_init = R_init.copy()
            self.n_beads = R_init.shape[0]
            logging.info(f"Using provided initial bead positions for chain with {self.n_beads} beads.")
        else:
            self.n_beads = n_beads
            self.R_init = self._initialize_beads()
            self.R_init = np.tile(np.mean(self.coords, axis=0), self.n_beads).reshape(-1,3)
            # import pdb;pdb.set_trace()
        
        self.R = self.R_init.copy()
        self.R_opt = None
        self.chi = None
        self.chi_opt = None
        # self.R_opt, self.chi_opt = self.optimize()

    def _initialize_beads(self):
        ## TODO: Probar a que todas sean en las misma cordeenada para ver qué ocurre
        """
        Inicializa las posiciones de los beads usando KMeans.
        Esto proporciona centros razonables para iniciar la iteración.
        """
        kmeans = KMeans(n_clusters=self.n_beads, n_init=10)
        kmeans.fit(self.coords)

        # R tendrá forma (M,3): posiciones iniciales de los beads
        return kmeans.cluster_centers_.astype(float)
    
    def compute_chi(self):
        """
        Calcula χ(r_i) para cada átomo y bead.
        χ_iμ = Δ(r_i - R_μ) / Σ_ν Δ(r_i - R_ν)
        donde Δ es una Gaussiana con desviación sigma.
        """
        diff = self.coords[:, None, :] - self.R[None, :, :]
        dist2 = np.sum(diff**2, axis=2)  # (N,M)

        # Gaussianas (Δ)
        weights = np.exp(-dist2 / (2 * self.sigma**2))

        # Normalización → χ
        chi = weights / np.sum(weights, axis=1, keepdims=True)
        # if np.any(np.isnan(chi)):
        #     import pdb;pdb.set_trace()
        return chi

    def update_R(self, chi):
        """
        Actualiza las posiciones de los beads siguiendo:
        R_μ = Σ_i [m_i r_i χ_iμ] / Σ_i [m_i χ_iμ]
        """
        # weighted = self.coords[:, None, :] * (self.masses[:, None, None] * chi)
        weighted = self.coords[:, None, :] * (self.masses[:, None, None] * chi[:, :, None])
        num = np.sum(weighted, axis=0)                  # (M,3)
        den = np.sum(self.masses[:, None] * chi, axis=0)  # (M,)

        # self.R = num / den[:, None]
        return num / den[:, None]

    def optimize(self, max_iter=100, tol=1e-4, debug=False):
        """
        Itera Voronoi hasta convergencia.
        Converge cuando ningún bead se mueve más de tol.
        """
        # R_old = self.R_init.copy()
        for it in range(max_iter):
            R_old = self.R.copy()

            chi = self.compute_chi()
            self.R = self.update_R(chi)

            # Cálculo del desplazamiento máximo
            shift = np.max(np.linalg.norm(self.R - R_old, axis=1))

            if shift < tol:
                self.logger.info(f"Converged in {it} iterations")
                break
        if debug:
            import pdb;pdb.set_trace()
        
        self.R_opt = self.R.copy() 
        self.chi_opt = self.compute_chi()
        # print(np.abs(self.R_init - self.R_opt))
        self.logger.info(f"Finished optimization after {it+1} iterations")
        # import pdb;pdb.set_trace()
        return self.R, chi

class AdaptiveCG(CoarseGrainedBase):
    """
    Implementación completa del método Soft-Voronoi usando Biopython.
    - Carga un PDB con Bio.PDB.
    - Extrae coordenadas y masas atómicas.
    - Inicializa los beads con KMeans.
    - Itera cálculo de chi y actualización de R hasta convergencia.
    """

    def __init__(self, name:str, 
                 inputPDBfilePath:str, 
                 params:dict, 
                 debug = False):
        
        self.inputPDBfilePath = inputPDBfilePath
        self.resolution = params.get("resolution", 100)
        # self.n_beads = params.get("nBeads", 100)
        self.sigma = params.get("sigma", 2)
        self.R_0 = params.get("R_0", 20.0)

        super().__init__(tpy  = "AdaptiveGC",
                         name = name,
                         inputPDBfilePath = inputPDBfilePath,
                         removeHetatm = True, removeHydrogens = False,removeNucleics  = True,
                         centerInput = params.get("centerInput",True),
                         SASA = params.get("SASA",False),
                         aggregateChains = params.get("aggregateChains",True),
                         debug = debug)
        
        self.logger.info(f"Generating coarse grained model (AdaptiveGC) ...")
        
        # Parse del microestado con BioPython
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("mol", inputPDBfilePath)

        atom_coords = []
        masses = []
        micro_chains = []

        # Extraemos todas las coordenadas y masas
        for atom in structure.get_atoms():
            atom_coords.append(atom.get_coord())
            # BioPython expone la masa como atom.mass
            masses.append(atom.mass)
            full_id = atom.get_full_id()
            micro_chains.append(full_id[2])

        self.micro_coords = np.array(atom_coords)     # (N,3)
        self.micro_masses = np.array(masses)     # (N,)
        self.micro_chains = np.array(micro_chains)     # (N,)
        cg_chain = []
        cg_coords = []
        cb_beads_ids = []

        ## Iterate over each class to make initial CG
        self.classes_beads = {}
        self.chain_beads = {}
        for tmp_class, chain_info in self._classes.items(): ## First calculate for the leader chain
            leader_chain = chain_info['leader']
            self.logger.info(f"Working in class {tmp_class} which leader is {leader_chain}.")
            tmp_coords = self.micro_coords[self.micro_chains == leader_chain]
            ref2orig = np.mean(tmp_coords, axis=0)
            n_beads = int(tmp_coords.shape[0] / self.resolution)
            self.logger.info(f" Chain {leader_chain} has {tmp_coords.shape[0]} atoms and will be represented with {n_beads} beads.")

            tmp_masses = self.micro_masses[self.micro_chains == leader_chain]
            tmp_chain_CG = ChainAdapticeCG(n_beads, tmp_coords, tmp_masses, sigma=self.sigma)
            tmp_chain_CG.optimize(max_iter=1000)
            self.classes_beads[tmp_class] = deepcopy(tmp_chain_CG)
            self.chain_beads[leader_chain] = deepcopy(tmp_chain_CG)
            cg_chain.extend(leader_chain*n_beads)
            cb_beads_ids.extend(list(range(n_beads)))
            cg_coords.extend(tmp_chain_CG.R_opt)
            ## Now propagate to the other chains in the class
            # other_chains = set(chain_info["members"]) - set("P")
            for _, ch, trans_matrix, rot_matrix in chain_info['transformations']:
                if ch == leader_chain:
                    continue
                self.logger.info(f" Propagating to chain {ch}.")
                tmp_coords_other_chain = self.micro_coords[self.micro_chains == ch]
                tmp_masses_other_chain = self.micro_masses[self.micro_chains == ch]
                beads_coords = self.classes_beads[tmp_class].R_opt.copy()
                R_init = (beads_coords - ref2orig) @ rot_matrix.as_matrix().T + ref2orig + trans_matrix 

                cg_other_chain = ChainAdapticeCG(n_beads, tmp_coords_other_chain, tmp_masses_other_chain, sigma=self.sigma, R_init=R_init.copy())
                cg_other_chain.optimize(max_iter=500) 

                self.chain_beads[ch] = deepcopy(cg_other_chain)
                
                cg_chain.extend(ch*n_beads)
                cb_beads_ids.extend(list(range(n_beads)))
                cg_coords.extend(cg_other_chain.R_opt)

        self.cg_chains = np.array(cg_chain)
        self.cg_beads_ids = np.array(cb_beads_ids)
        self.cg_coords = np.array(cg_coords, dtype=np.float32)

        self.logger.info(f"Model generation end")

        self.logger.info(f"Calculating CG distances...")

        bead_distances = self.calculateBeadDistances(self.cg_coords, self.R_0)
        self.bead_distances = bead_distances[0]
        self.bead_distances_indexes = bead_distances[1]
        self.intra_chain_distances = bead_distances[2]
        self.inter_chain_distances = bead_distances[3]
    
    def calculateBeadDistances(self, coords:np.ndarray, R_0:float=20.0):
        from scipy.spatial.distance import pdist
        from itertools import combinations

        # Condensed vector of length N*(N-1)/2
        dcond = pdist(coords, metric='euclidean')
        indexes = np.array(list(combinations(range(len(coords)), 2)))
        chain_indexes = self.cg_chains[indexes]
        beads_indexes = self.cg_beads_ids[indexes]

        # chain_name = np.unique(self.chains)

        ## Intra-chain distances
        # intra_chain_distances = []
        intra_mask = (chain_indexes[:,0] == chain_indexes[:,1]) & (dcond < R_0)
        intra_distances = dcond[intra_mask]
        intra_chain_indexes = chain_indexes[intra_mask]
        intra_beads_indexes = beads_indexes[intra_mask]

        intra_chain_distances = list(zip(
            intra_chain_indexes[:,0].tolist(),
            intra_beads_indexes[:,0].tolist(),
            intra_chain_indexes[:,1].tolist(),
            intra_beads_indexes[:,1].tolist(),
            intra_distances.tolist(),
        ))

        ## Inter-chain distances
        inter_mask = (chain_indexes[:,0] != chain_indexes[:,1]) & (dcond < R_0)
        inter_distances = dcond[inter_mask]
        inter_chain_indexes = chain_indexes[inter_mask]
        inter_beads_indexes = beads_indexes[inter_mask]

        ## TODO: just [[bead_idx_1, bead_idx_2, distance], [...]]
        inter_chain_distances = list(zip(
            inter_chain_indexes[:,0].tolist(),
            inter_beads_indexes[:,0].tolist(),
            inter_chain_indexes[:,1].tolist(),
            inter_beads_indexes[:,1].tolist(),
            inter_distances.tolist(),
        ))

        return dcond, indexes, intra_chain_distances, inter_chain_distances
    
    def write_pdb(self):
        # from moleculekit.molecule import Molecule

        # mol = Molecule()
        # mol.coords = self.cg_coords.reshape(-1,3,1)
        # mol.chain = self.cg_chains
        # # mol.masses = 
        # mol.name = np.repeat("C", mol.chain.shape[0])
        # mol.occupancy = np.repeat(1, mol.chain.shape[0])
        # mol.beta = np.repeat(1, mol.chain.shape[0])
        # mol.write("/tmp/my_mol.pdb")

        """
        Exporta los beads en un archivo PDB sencillo.
        """
        filename = "/tmp/my_mol.pdb"
        with open(filename, "w") as f:
            for i, (x, y, z) in enumerate(self.cg_coords):
                line = (
                    f"ATOM  {i+1:5d}  BEA BDS A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           X\n"
                )
                f.write(line)
        import os
        os.system("chimerax /tmp/my_mol.pdb")
        # return filename

    def view(self, min_radius = 2.0, max_radius = 8.0, bead_radius=2.0, out_script="/tmp/show_beads.cxc", view=True):
        """
        Visualiza los beads y la estructura original en ChimeraX.
        """
        with open(out_script, "w") as f:

            # escala lineal de masas a radios
            # Create a new molecule for beads (fake atoms)
            f.write("# Creating bead pseudo-atoms\n")
            f.write("close all\n")     # hide everything first
            f.write("# ChimeraX script to visualize beads and original structure\n")
            f.write(f"open {self.inputPDBfilePath}\n\n")
            f.write("show #1\n\n")       # show original PDB

            # Add spheres at bead positions
            for idx, (tmp_chain, cg) in enumerate(self.chain_beads.items()):
                # leader_chain = self._classes[tmp_class]['leader']
                tmp_masses = tmp_masses = self.micro_masses[self.micro_chains == tmp_chain]
                bead_masses = np.sum(tmp_masses[:, None] * cg.chi_opt , axis=0)
                radius = min_radius + (bead_masses - bead_masses.min()) / (bead_masses.max() - bead_masses.min()) * (max_radius - min_radius)
                # import pdb;pdb.set_trace()
                for i, (x, y, z) in enumerate(cg.R_init):
                    # color fijo o puedes crear un array de colores si quieres variar
                    f.write(f"shape sphere center {x:.3f},{y:.3f},{z:.3f} radius 1 mesh false color #4079bf96 model #{idx+2}.1.{i+1}\n")
                    # f.write(f"shape sphere center {x:.3f},{y:.3f},{z:.3f} radius {radius[i]} mesh false color #4079bf96 model #{idx+2}.1.{i+1}\n")
                    f.write("\n# Final view tweaks\n")
                for i, (x, y, z) in enumerate(cg.R_opt):
                    # color fijo o puedes crear un array de colores si quieres variar
                    f.write(f"shape sphere center {x:.3f},{y:.3f},{z:.3f} radius 3 mesh false color #bf404077 model #{idx+2}.2.{i+1}\n")
                    # f.write(f"shape sphere center {x:.3f},{y:.3f},{z:.3f} radius {radius[i]} mesh false color #bf404077 model #{idx+2}.2.{i+1}\n")
                f.write(f"rename #{idx+2}.1 CG_init\n")
                f.write(f"rename #{idx+2}.2 CG_opt\n")
                f.write(f"rename #{idx+2} {tmp_chain}\n")
                # f.write(f"rename #{idx+2} CG_opt\n")
            last_chain_idx = idx + 2 
            
            for idx, (ch_a, b_a, ch_b, b_b, dist) in enumerate(self.intra_chain_distances):
                coords_a = self.cg_coords[(self.cg_chains == ch_a) & (self.cg_beads_ids == b_a)][0]
                coords_b = self.cg_coords[(self.cg_chains == ch_b) & (self.cg_beads_ids == b_b)][0]
                f.write(f"shape cylinder radius 0.25 fromPoint {coords_a[0]:.3f},{coords_a[1]:.3f},{coords_a[2]:.3f} toPoint {coords_b[0]:.3f},{coords_b[1]:.3f},{coords_b[2]:.3f} color green model #{last_chain_idx+1}.{idx+1} name ch{ch_a}_{b_a}__ch{ch_b}_{b_b}__{round(dist, 2)}\n")   

            f.write(f"rename #{last_chain_idx+1} intraContacts\n")
            
            for idx, (ch_a, b_a, ch_b, b_b, dist) in enumerate(self.inter_chain_distances):
                coords_a = self.cg_coords[(self.cg_chains == ch_a) & (self.cg_beads_ids == b_a)][0]
                coords_b = self.cg_coords[(self.cg_chains == ch_b) & (self.cg_beads_ids == b_b)][0]
                f.write(f"shape cylinder radius 0.25 fromPoint {coords_a[0]:.3f},{coords_a[1]:.3f},{coords_a[2]:.3f} toPoint {coords_b[0]:.3f},{coords_b[1]:.3f},{coords_b[2]:.3f} color yellow model #{last_chain_idx+2}.{idx+1} name ch{ch_a}_{b_a}__ch{ch_b}_{b_b}__{round(dist, 2)}\n")   

            f.write(f"rename #{last_chain_idx+2} interContacts\n")
            # f.write("show #1 #2\n")
            # f.write("transparency #2 30\n")
            # f.write("hide #*.1\n")  # hide initial beads
            # f.write("hide #2-14\n")  # hide initial beads
            f.write("hide atoms\n")  # hide initial beads
            f.write("show cartoons\n")  # hide initial beads
            f.write("lighting depthCue false\n")  # hide initial beads
            # f.write("hide #1/A-M cartoon\n")  # hide initial beads
            f.write("zoom\n")

        print(f"CXC script written to {out_script}")
        if view:
            import os
            os.system(f"chimerax {out_script} &")

