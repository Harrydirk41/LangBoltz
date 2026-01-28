import sys
import os
import re
import glob
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from esm.pretrained import ESM3_sm_open_v0,ESM3_structure_decoder_v0
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.utils import encoding
import mdtraj as md
import functools
import einops
from custom_rotary import RotaryEmbedding
from huggingface_hub import login
#login(token = "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")
import Geometry
import PeptideBuilder
import Bio.PDB
import math
from huggingface_hub import hf_hub_download
from utils import remove_first_last_residue, AA_TO_N, three_to_one,tokens_to_torsion_deg,build_geo,encode_pdb
from utils import RegressionHead,SwiGLU,swiglu_correction_fn,swiglu_ln_ffn,gelu_ln_ffn

class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,ffn_type="gelu",
        expansion_ratio: float = 4.0,scaling_factor = 1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,n_heads,bias,qk_layernorm)
        self.scaling_factor = scaling_factor
        if ffn_type == "swiglu":
            self.ffn_seq = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn_seq = gelu_ln_ffn(d_model, expansion_ratio, bias)
    def forward(self,x_seq,mask):
        r1_seq = self.attn(x_seq,mask)
        x_seq = x_seq + r1_seq / self.scaling_factor
        
        r3_seq = self.ffn_seq(x_seq)
        x_seq = x_seq + r3_seq / self.scaling_factor
        return x_seq
class Transformerstack(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,n_layers=24):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,bias,qk_layernorm) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model, bias=False)
    def forward(self,x_seq,mask):
        for block in self.blocks:
            x_seq = block(x_seq,mask)
        return self.norm(x_seq), x_seq

class ESMDynamics(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,n_layers=24):
        super().__init__()
        self.transformer = Transformerstack(d_model,n_heads,bias,qk_layernorm,n_layers)
        self.torsion_head = RegressionHead(d_model, 40*4)
        self.seq_embed = nn.Embedding(64, d_model)
        self.struc_embed = nn.Embedding(4096 + 5, d_model)
    def forward(self,seq_tokens,struc_tokens,mask):
        N_seq = seq_tokens.shape[1]
        N_struc = int(struc_tokens.shape[1] / seq_tokens.shape[1])
        x_seq = self.seq_embed(seq_tokens)
        x_struc = self.struc_embed(struc_tokens)
        x_input_seq = x_seq + x_struc
        x_torsion_out,x_torsion_out_embed = self.transformer(x_input_seq,mask)
        torsion_logits = self.torsion_head(x_torsion_out)
        return torsion_logits
class MultiHeadAttention(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, bias: bool = False, qk_layernorm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv_seq = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj_seq = nn.Linear(d_model, d_model, bias=bias)
        if qk_layernorm:
            self.q_ln_seq = nn.LayerNorm(d_model, bias=bias)
            self.k_ln_seq = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln_seq = nn.Identity()
            self.k_ln_seq = nn.Identity()
        self.rotary_seq = RotaryEmbedding(d_model // n_heads)
    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary_seq(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k
    def forward(self, x_seq,mask):
        N_seq = x_seq.shape[1]
        qkv_BLD3_seq = self.layernorm_qkv_seq(x_seq)
        query_BLD_seq, key_BLD_seq, value_BLD_seq = torch.chunk(qkv_BLD3_seq, 3, dim=-1)
        query_BLD_seq, key_BLD_seq = (
            self.q_ln_seq(query_BLD_seq).to(query_BLD_seq.dtype),
            self.k_ln_seq(key_BLD_seq).to(query_BLD_seq.dtype),
        )
        query_BLD_seq, key_BLD_seq = self._apply_rotary(query_BLD_seq, key_BLD_seq)
        query_BLD_input_struc = query_BLD_seq
        key_BLD_input_struc = key_BLD_seq
        value_BLD_input_struc = value_BLD_seq
        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads)
        query_BHLD_input_struc, key_BHLD_input_struc, value_BHLD_input_struc = map(reshaper, (query_BLD_input_struc,key_BLD_input_struc, value_BLD_input_struc))
        context_BHLD_struc = F.scaled_dot_product_attention(
            query_BHLD_input_struc, key_BHLD_input_struc, value_BHLD_input_struc, mask.bool()
        )
        context_BHLD_struc = einops.rearrange(context_BHLD_struc, "b h s d -> b s (h d)")
        out_seq = self.out_proj_seq(context_BHLD_struc)
        return out_seq

class ESM3LightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, seq_tokens, struc_tokens,mask):
        torsion_logits = self.model(seq_tokens=seq_tokens, struc_tokens=struc_tokens,mask = mask)
        return torsion_logits

    def distance_attention_mask(self,coords, min_dist=1.0, max_dist=12.0):
        diff = coords[:, :, None, :] - coords[:, None, :, :]
        dist = torch.norm(diff, dim=-1)
        mask = (dist > min_dist) & (dist < max_dist)
        eye = torch.eye(dist.shape[1], dtype=torch.bool, device=coords.device)
        mask = mask | eye.unsqueeze(0)
        return mask.unsqueeze(1)
    def predict_step(self, batch, batch_idx):
        folder = self.folder
        merge_folder = self.merge_folder
        sequence_tokens = self.sequence_tokens.to(self.device)
        structure_tokens = self.structure_tokens.to(self.device)
        threeD = self.threeD.to(self.device)
        sequence = self.sequence
        N_batch = structure_tokens.shape[0]
        bb_coords = (
            self.decoder.decode(
                structure_tokens,
                torch.ones_like(sequence_tokens),
                torch.zeros_like(sequence_tokens),
            )["bb_pred"].detach().cpu())
        for i in range(N_batch):
            chain = ProteinChain.from_backbone_atom_coordinates(bb_coords[i:i+1], sequence="X" + sequence + "X")
            chain.infer_oxygen().to_pdb(f"{folder}/sample_{batch_idx}_{i}.pdb")
        sequence_tokens = sequence_tokens[:,1:-1]
        structure_tokens = structure_tokens[:,1:-1]
        threeD = threeD[:,:,1]
        seq_mask = (sequence_tokens != 1).to(structure_tokens.device)
        seq_attn_mask = (seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)).unsqueeze(1)
        struc_attn_mask = self.distance_attention_mask(threeD) 
        struc_attn_mask = struc_attn_mask * seq_attn_mask
        logits = self.forward(seq_tokens=sequence_tokens, 
                              struc_tokens=structure_tokens,
                             mask = struc_attn_mask)
        logits = logits.reshape(logits.shape[0], logits.shape[1], 4, 40)
        probs = torch.softmax(logits[:, :, :, :36], dim=-1)
        torsion_tokens = torch.multinomial(probs.view(-1, 36), 1).view(*logits.shape[:-1])
        torsion = tokens_to_torsion_deg(torsion_tokens,n_bins = 36).detach().cpu().numpy()
        torsion = torsion.reshape(torsion.shape[0],-1,4)
        for i in range(N_batch):
            traj = md.load(f"{folder}/sample_{batch_idx}_{i}.pdb")
            traj = remove_first_last_residue(traj)
            _,phi = md.compute_phi(traj)
            _,psi = md.compute_psi(traj)
            _,omega = md.compute_omega(traj)
            phi = np.degrees(phi)
            psi = np.degrees(psi)
            for j in range(len(sequence)):
                if j == 0:
                    geo = build_geo(sequence[j],torsion[i,j],-60,40)
                    structure = PeptideBuilder.initialize_res(geo)
                else:
                    geo = build_geo(sequence[j],torsion[i,j],phi[0,j-1],psi[0,j-1])
                    PeptideBuilder.add_residue(structure, geo)
            
            out = Bio.PDB.PDBIO()
            out.set_structure(structure)
            out.save(f"{folder}/sample_{batch_idx}_{i}.pdb")

def sample(lightning_model,trainer,protein_name=None,device="cuda"):
    base = os.getcwd()
    pdb_dir = os.path.join(base, "stepone_merge",protein_name,"sample.pdb")
    xtc_dir = os.path.join(base, "stepone_merge",protein_name,"sample.xtc")
    pdb_old_dir = os.path.join(base, "stepone_merge",protein_name,"sample.pdb")
    xtc_old_dir = os.path.join(base, "stepone_merge",protein_name,"sample.xtc")
    model = ESM3_sm_open_v0("cpu").train()
    tokenizer_sequence = model.tokenizers.sequence
    pdb = md.load(xtc_dir,top = pdb_dir)
    pdb_old = md.load(xtc_old_dir,top = pdb_old_dir)
    sequence = ''.join(three_to_one.get(res.name, 'X') for res in pdb_old.topology.residues)
    seq_len = len(sequence)
    sequence_tokens = encoding.tokenize_sequence(sequence, tokenizer_sequence, add_special_tokens=True)
    structure_tokens,coords = encode_pdb(pdb,device)
    sequence_tokens = sequence_tokens.unsqueeze(0).repeat(structure_tokens.shape[0],1)
    num_sample = structure_tokens.shape[0]
    folder = f"steptwo_sample/{protein_name}"
    merge_folder = f"all_atom_MLM/{protein_name}"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(merge_folder, exist_ok=True)
    
    lightning_model.name = protein_name
    lightning_model.merge_folder= merge_folder
    lightning_model.folder = folder
    lightning_model.sequence_tokens = sequence_tokens
    lightning_model.structure_tokens = structure_tokens
    lightning_model.threeD = coords
    lightning_model.sequence = sequence
    if seq_len > 300:
        batch_size = 64
    elif seq_len > 100:
        batch_size = 128
    elif seq_len < 50:
        batch_size = 500
    else:
        batch_size = 256
    dummy = torch.zeros(structure_tokens.shape[0], 1)  # shape doesn't matter if you ignore it
    dummy_loader = DataLoader(TensorDataset(dummy), batch_size=batch_size)
    trainer.predict(lightning_model, dummy_loader)

    pdb_files = []
    num_batches = math.ceil(structure_tokens.shape[0] / batch_size)
        
    for batch_idx in range(num_batches):
        for j in range(batch_size):
            sample_idx = batch_idx * batch_size + j
            if sample_idx >= num_sample:
                break  # no more samples to cover
            pdb_files.append(f"{folder}/sample_{batch_idx}_{j}.pdb")

    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {folder}")
    traj_list = [md.load(pdb) for pdb in pdb_files]
    merged_traj = md.join(traj_list)
    xtc_path = os.path.join(merge_folder, "sample.xtc")
    merged_traj.save_xtc(xtc_path)
    pdb_path = os.path.join(merge_folder, "sample.pdb")
    merged_traj[0].save_pdb(pdb_path)
    print(f"Merged {len(pdb_files)} PDB files into {xtc_path}")
    print(f"Saved first frame as {pdb_path}")
    print(f"Deleted original {len(pdb_files)} PDB files from {folder}")

def run_experiment(protein_name = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESM3_sm_open_v0("cpu").train()
    backbone_model = ESMDynamics(1536,12,n_layers=8)
    backbone_model.seq_embed = model.encoder.sequence_embed
    backbone_model.struc_embed = model.encoder.structure_tokens_embed
    ckpt_path = hf_hub_download(
        repo_id="harrydirk41/LangBoltz",
        filename="LangBoltz_sidechain.ckpt",
    )
    lightning_model = ESM3LightningModule.load_from_checkpoint(ckpt_path,model = backbone_model)
    lightning_model.decoder = ESM3_structure_decoder_v0("cpu")
    del model
    trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None
        )
    sample(lightning_model,trainer,protein_name=protein_name,device=device)
    del lightning_model


    

    