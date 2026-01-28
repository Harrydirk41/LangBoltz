import torch
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import mdtraj as md
import Geometry
import PeptideBuilder
import Bio.PDB
from esm.pretrained import ESM3_structure_encoder_v0
AA_TO_N = {
    "S": 1,
    "C": 1,
    "V": 1,
    "I": 2,
    "L": 2,
    "T": 1,
    "R": 4,
    "K": 4,
    "D": 2,
    "N": 2,
    "E": 3,
    "Q": 3,
    "M": 3,
    "H": 2,
    "F": 2,
    "Y": 2,
    "W": 2,
}

three_to_one = {
            "UNK": 'X', #3
            'LEU': 'L',  # 4
            'ALA': 'A',  # 5
            'GLY': 'G',  # 6
            'VAL': 'V',  # 7
            'SER': 'S',  # 8
            'GLU': 'E',  # 9
            'ARG': 'R',  # 10
            'THR': 'T',  # 11
            'ILE': 'I',  # 12
            'ASP': 'D',  # 13
            'PRO': 'P',  # 14
            'LYS': 'K',  # 15
            'GLN': 'Q',  # 16
            'ASN': 'N',  # 17
            'PHE': 'F',  # 18
            'TYR': 'Y',  # 19
            'MET': 'M',  # 20
            'HIS': 'H',  # 21
            'TRP': 'W',  # 22
            'CYS': 'C',  # 23
        }
def remove_first_last_residue(traj: md.Trajectory) -> md.Trajectory:
    n_res = traj.topology.n_residues
    if n_res <= 2:
        raise ValueError("Trajectory must have more than 2 residues.")
    
    atom_indices = [
        atom.index
        for atom in traj.topology.atoms
        if 0 < atom.residue.index < n_res - 1
    ]
    
    return traj.atom_slice(atom_indices)

def RegressionHead(
    d_model: int, output_dim: int, hidden_dim: int | None = None
) -> nn.Module:
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )

class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2
def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)
def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )
def gelu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    hidden_dim = int(expansion_ratio * d_model)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, hidden_dim, bias=bias),
        nn.GELU(),
        nn.Linear(hidden_dim, d_model, bias=bias),
    )

def tokens_to_torsion_deg(tokens, n_bins=36, sample_uniform=True):
    bin_width = 360.0 / n_bins
    deg = torch.empty_like(tokens, dtype=torch.float32)
    mask = tokens < n_bins
    if sample_uniform:
        u = torch.rand_like(deg)
        deg[mask] = (tokens[mask].float() + u[mask]) * bin_width
    else:
        deg[mask] = (tokens[mask].float() + 0.5) * bin_width
    deg[~mask] = float('nan')
    deg = ((deg + 180.0) % 360.0) - 180.0
    return deg

def build_geo(aa,chi,phi,psi):
    #geo:Geo
    #torsion:4
    N = 0
    geo = Geometry.geometry(aa)
    if aa == "S":
        geo.N_CA_CB_OG_diangle = chi[0]
        N = 1
    if aa == "C":
        geo.N_CA_CB_SG_diangle = chi[0]
        N = 1
    if aa == "V":
        geo.N_CA_CB_CG1_diangle = chi[0]
        geo.N_CA_CB_CG2_dianlge = (chi[0] + 120 + 180.0) % 360.0 - 180.0
        N = 1
    if aa =="I":
        geo.N_CA_CB_CG1_diangle = chi[0]
        geo.N_CA_CB_CG2_diangle = (chi[0] + 120 + 180.0) % 360.0 - 180.0
        geo.CA_CB_CG1_CD1_diangle = chi[1]
        N = 2
    if aa == "L":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD1_diangle = chi[1]
        geo.CA_CB_CG_CD2_diangle = (chi[1] - 108.2 + 180) % 360 - 180
        N = 2
    if aa == "T":
        #question
        geo.N_CA_CB_OG1_diangle = chi[0]
        geo.N_CA_CB_OG2_diangle = (chi[0] + 120 + 180.0) % 360.0 - 180.0
        N = 1
    if aa == "R":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD_diangle = chi[1]
        geo.CB_CG_CD_NE_diangle = chi[2]
        geo.CG_CD_NE_CZ_diangle = chi[3]
        N = 4
    if aa == "K":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD_diangle = chi[1]
        geo.CB_CG_CD_CE_diangle = chi[2]
        geo.CG_CD_CE_NZ_diangle = chi[3]
        N = 4
    if aa == "D":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_OD1_diangle = chi[1]
        if chi[1] > 0:
            geo.CA_CB_CG_OD2_diangle = chi[1] - 180.0
        else:
            geo.CA_CB_CG_OD2_diangle = chi[1] + 180.0
        N = 2
    if aa =="N":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_OD1_diangle = chi[1]
        if chi[1] > 0:
            geo.CA_CB_CG_ND2_diangle = chi[1] - 180.0
        else:
            geo.CA_CB_CG_ND2_diangle = chi[1] + 180.0
        N = 2
    if aa == "E":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD_diangle = chi[1]
        geo.CB_CG_CD_OE1_diangle = chi[2]
        if geo.CB_CG_CD_OE1_diangle > 0:
            geo.CB_CG_CD_OE2_diangle = chi[2] - 180.0
        else:
            geo.CB_CG_CD_OE2_diangle = chi[2] + 180.0
        N = 3
    if aa == "Q":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD_diangle = chi[1]
        geo.CB_CG_CD_OE1_diangle = chi[2]
        if geo.CB_CG_CD_OE1_diangle > 0:
            geo.CB_CG_CD_NE2_diangle = chi[2] - 180.0
        else:
            geo.CB_CG_CD_NE2_diangle = chi[2] + 180.0
        N = 3
    if aa == "M":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_SD_diangle = chi[1]
        geo.CB_CG_SD_CE_diangle = chi[2]
        N = 3
    if aa == "H":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_ND1_diangle = chi[1]
        if geo.CA_CB_CG_ND1_diangle > 0:
            geo.CA_CB_CG_CD2_diangle = chi[1] - 180.0
        else:
            geo.CA_CB_CG_CD2_diangle = chi[1] + 180.0
    if aa == "F":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD1_diangle = chi[1]
        if geo.CA_CB_CG_CD1_diangle > 0:
            geo.CA_CB_CG_CD2_diangle = chi[1] - 180.0
        else:
            geo.CA_CB_CG_CD2_diangle = chi[1] + 180.0
        N = 2
    if aa == "Y":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD1_diangle = chi[1]
        if geo.CA_CB_CG_CD1_diangle > 0:
            geo.CA_CB_CG_CD2_diangle = chi[1] - 180.0
        else:
            geo.CA_CB_CG_CD2_diangle = chi[1] + 180.0
        N = 2
    if aa == "W":
        geo.N_CA_CB_CG_diangle = chi[0]
        geo.CA_CB_CG_CD1_diangle = chi[1]
        if geo.CA_CB_CG_CD1_diangle > 0:
            geo.CA_CB_CG_CD2_diangle = chi[1] - 180.0
        else:
            geo.CA_CB_CG_CD2_diangle = chi[1] + 180.0
        N = 2

    geo.phi = phi
    geo.psi_im1 = psi
    return geo

def encode_pdb(traj,device):
    struc_encoder = ESM3_structure_encoder_v0(device)
    npy = get_coord(traj)
    coords = torch.zeros(npy.shape[0],npy.shape[1],37,3)
    coords[:,:,:3,:] = npy[:,:,:3,:]
    coords = coords.to(device)
    diff = coords[:, 1:,1, :] - coords[:, :-1,1, :]
    dist = torch.norm(diff, dim=-1)
    mean_consecutive_dist = dist.mean()
    if mean_consecutive_dist < 1:
        coords = coords * 10
    coords[:, :, 3:, :] = float('nan')
    with torch.no_grad():
        _, structure_tokens = struc_encoder.encode(coords)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0).detach().cpu()
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097
    return structure_tokens.detach().cpu(),coords[:,:,:3,:]

def get_coord(traj):
    atom_names = ['N', 'CA', 'C']
    top = traj.topology
    residues = list(top.residues)
    N_res = len(residues)
    N_frame = traj.n_frames
    indices = np.array([
        [next(a.index for a in res.atoms if a.name == name) for name in atom_names]
        for res in residues
    ])      
    coords = traj.xyz[:, indices, :]
    return torch.from_numpy(coords)