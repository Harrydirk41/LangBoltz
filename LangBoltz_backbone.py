import sys,os,re
import mdtraj as md
import random
from tqdm import tqdm
import numpy as np
import time
import glob

import functools
import einops

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F
import torch.nn as nn


from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.utils import encoding
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from custom_rotary import RotaryEmbedding
from huggingface_hub import login
login(token = "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")
import math
import copy
from huggingface_hub import hf_hub_download

from utils import remove_first_last_residue, AA_TO_N, three_to_one,encode_pdb
from utils import RegressionHead,SwiGLU,swiglu_correction_fn,swiglu_ln_ffn,gelu_ln_ffn
class TransformerBlock(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,ffn_type="gelu",
        expansion_ratio: float = 4.0,scaling_factor = 1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,n_heads,bias,qk_layernorm)
        self.scaling_factor = scaling_factor
        if ffn_type == "swiglu":
            self.ffn_seq = swiglu_ln_ffn(d_model, expansion_ratio, bias)
            self.ffn_struc = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        elif ffn_type == "gelu":
            self.ffn_seq = gelu_ln_ffn(d_model, expansion_ratio, bias)
            self.ffn_struc = gelu_ln_ffn(d_model, expansion_ratio, bias)
    def forward(self,x_seq,x_struc,mask,temp, T = 1,N_seg = 2,index = 0):
        if index == 0:
            r1_seq,r1_struc = self.attn(x_seq, x_struc,mask,temp, T,N_seg = N_seg,index = index)
            x_seq = x_seq + r1_seq / self.scaling_factor
            x_struc = x_struc + r1_struc / self.scaling_factor
            r3_seq = self.ffn_seq(x_seq)
            x_seq = x_seq + r3_seq / self.scaling_factor
            r3_struc = self.ffn_struc(x_struc) 
            x_struc = x_struc + r3_struc / self.scaling_factor
        else:
            r1_seq,r1_struc = self.attn(x_seq, x_struc,mask,temp, T,N_seg = N_seg,index = index)
            x_struc = x_struc + r1_struc / self.scaling_factor
            r3_struc = self.ffn_struc(x_struc) 
            x_struc = x_struc + r3_struc / self.scaling_factor
        return x_seq,x_struc
class Transformerstack(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,n_layers=24):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model,n_heads,bias,qk_layernorm) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model, bias=False)
    def forward(self,x_seq,x_struc,mask,temp, T = 1,N_seg = 2,index = 0):
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x_seq,x_struc = block(x_seq,x_struc,mask,temp, T,N_seg = N_seg,index = index)
        return self.norm(x_struc), x_struc

class ESMDynamics(nn.Module):
    def __init__(self,d_model,n_heads,bias = False,qk_layernorm=True,n_layers=24):
        super().__init__()
        self.transformer = Transformerstack(d_model,n_heads,bias,qk_layernorm,n_layers)
        self.structure_head = RegressionHead(d_model, 4096)
        self.seq_embed = nn.Embedding(64, d_model)
        self.struc_embed = nn.Embedding(4096 + 5, d_model)

    def forward(self,seq_tokens,struc_tokens,mask,temp, T = 1,N_seg = 2,index = 1,input_ids = None):
        N_seq = seq_tokens.shape[1]
        x_seq = self.seq_embed(seq_tokens)
        x_struc = self.struc_embed(struc_tokens)
        x_struc_out,x_struc_out_embed = self.transformer(x_seq,x_struc,mask,temp,T,N_seg = N_seg,index = index)
        structure_logits = self.structure_head(x_struc_out)
        return structure_logits
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
        self.layernorm_qkv_struc = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 3, bias=bias)
        )
        self.out_proj_seq = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj_struc = nn.Linear(d_model, d_model, bias=bias)
        if qk_layernorm:
            self.q_ln_seq = nn.LayerNorm(d_model, bias=bias)
            self.k_ln_seq = nn.LayerNorm(d_model, bias=bias)
            self.q_ln_struc = nn.LayerNorm(d_model, bias=bias)
            self.k_ln_struc = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln_seq = nn.Identity()
            self.k_ln_seq = nn.Identity()
            self.q_ln_struc = nn.Identity()
            self.k_ln_struc = nn.Identity()
        self.rotary_seq = RotaryEmbedding(d_model // n_heads)
        self.rotary_struc = RotaryEmbedding(d_model // n_heads)
        self.rotary_time = RotaryEmbedding(d_model // n_heads)
        self.temp_embed = nn.Sequential(
                    nn.Linear(1, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model)
                )
        self.T_embed = nn.Sequential(
                    nn.Linear(256, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model)
                )
    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor,use_struc = True,N_seq = None,N_seg = None,index = 0):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        if use_struc:
            position_res = torch.arange(N_seq).to(q.device)
            position_res = position_res.repeat(N_seg)
            q, k = self.rotary_struc(q, k,positions=position_res[index:index + 1])
        else:
            q, k = self.rotary_seq(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k
    def get_timestep_embedding(self,timesteps, embedding_dim = 256,max_len=10000):
        assert len(timesteps.shape) == 1
        timesteps = timesteps * max_len
        half_dim = embedding_dim // 2
        emb = math.log(max_len) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        assert emb.shape == (timesteps.shape[0], embedding_dim)
        return emb
    def make_autoregressive_attn_mask(self,batch_mask: torch.Tensor, N_seg: int, *, as_float: bool = True):
        B, N_res = batch_mask.shape
        L = (N_seg + 1) * N_res
        device = batch_mask.device
        causal = torch.ones((L, L), dtype=torch.bool, device=device).tril().view(1,1,L,L)
        tiled = batch_mask.repeat(1, N_seg + 1).bool()         
        key_valid = tiled.view(B, 1, 1, L)                     
        q_valid   = tiled.view(B, 1, L, 1)                     
        allowed = causal & key_valid & q_valid                 
        first_edit = batch_mask[:,None,:] * batch_mask[:,:,None]
        allowed[:,0,:N_res,:N_res] = first_edit.bool()
        return allowed
    def forward(self, x_seq, x_struc,mask,temp,T = 1,N_seg = 2,index = 0):
        N_seq = x_seq.shape[1]
        position_T = torch.arange(N_seg).to(x_struc.device) * T / 100
        T_input = position_T.repeat_interleave(N_seq)[index:index + 1]
        if index == 0:
            qkv_BLD3_seq = self.layernorm_qkv_seq(x_seq)
            query_BLD_seq, key_BLD_seq, value_BLD_seq = torch.chunk(qkv_BLD3_seq, 3, dim=-1)
            query_BLD_seq, key_BLD_seq = (
                self.q_ln_seq(query_BLD_seq).to(query_BLD_seq.dtype),
                self.k_ln_seq(key_BLD_seq).to(query_BLD_seq.dtype),
            )
            query_BLD_seq, key_BLD_seq = self._apply_rotary(query_BLD_seq, key_BLD_seq,use_struc = False)
            self.k_cache[:,:N_seq] = key_BLD_seq
            self.v_cache[:,:N_seq] = value_BLD_seq
        temp_embed = self.temp_embed(temp)
        T_embed = self.T_embed(self.get_timestep_embedding(T_input))
        x_norm = self.layernorm_qkv_struc[0](x_struc)
        x_mod = x_norm  + temp_embed + T_embed[None,:,:]
        qkv_BLD3_struc = self.layernorm_qkv_struc[1](x_mod)
        query_BLD_struc, key_BLD_struc, value_BLD_struc = torch.chunk(qkv_BLD3_struc, 3, dim=-1)
        query_BLD_struc, key_BLD_struc = (
            self.q_ln_struc(query_BLD_struc).to(query_BLD_struc.dtype),
            self.k_ln_struc(key_BLD_struc).to(query_BLD_struc.dtype)
        )
        query_BLD_struc, key_BLD_struc = self._apply_rotary(query_BLD_struc, key_BLD_struc,use_struc = True,N_seq = N_seq,N_seg = N_seg,index = index)
        self.k_cache[:,N_seq + index:N_seq + index + 1] = key_BLD_struc
        self.v_cache[:,N_seq + index:N_seq + index + 1] = value_BLD_struc
        key_BLD_input_struc = self.k_cache[:,:N_seq + index + 1]
        value_BLD_input_struc = self.v_cache[:,:N_seq + index + 1]
        if index == 0:
            query_BLD_input_struc = torch.cat([query_BLD_seq,query_BLD_struc],dim = 1)
        else:
            query_BLD_input_struc = query_BLD_struc
        reshaper = functools.partial(einops.rearrange, pattern="b s (h d) -> b h s d", h=self.n_heads)
        query_BHLD_input_struc, key_BHLD_input_struc, value_BHLD_input_struc = map(reshaper, (query_BLD_input_struc,key_BLD_input_struc, value_BLD_input_struc))
        mask_BHLL_struc = self.make_autoregressive_attn_mask(mask, N_seg, as_float=True).bool()
        if index == 0:
            mask_BHLL_struc = mask_BHLL_struc[:,:,:N_seq + index + 1,:N_seq + index + 1]
        else:
            mask_BHLL_struc = mask_BHLL_struc[:,:,N_seq + index:N_seq + index + 1,:N_seq + index + 1]
        context_BHLD_struc = F.scaled_dot_product_attention(
            query_BHLD_input_struc, key_BHLD_input_struc, value_BHLD_input_struc, mask_BHLL_struc
        )
        context_BHLD_struc = einops.rearrange(context_BHLD_struc, "b h s d -> b s (h d)")
        if index == 0:
            out_struc = self.out_proj_struc(context_BHLD_struc[:,N_seq:])
            out_seq = self.out_proj_seq(context_BHLD_struc[:,:N_seq])
        else:
            out_struc = self.out_proj_struc(context_BHLD_struc)
            out_seq = None
        return out_seq,out_struc

class ESM3LightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, seq_tokens, struc_tokens,mask,temp,T = 1,N_seg = 1,index = 1):
        structure_logits = self.model(seq_tokens=seq_tokens, struc_tokens=struc_tokens,mask = mask,temp = temp, T = T,N_seg = N_seg,index = index)
        return structure_logits
    
    def sample_i_with_temperature(self,logits, i = 0, temperature=1.0):
        logits_i = logits[:, i, :]
        scaled_logits = logits_i / temperature
        probs = F.softmax(scaled_logits, dim=-1)         
        log_probs = F.log_softmax(scaled_logits, dim=-1)   
        sampled_idx = torch.multinomial(probs, num_samples=1).squeeze(-1) 
        p_sampled = probs.gather(1, sampled_idx.unsqueeze(-1)).squeeze(-1)        
        logp_sampled = log_probs.gather(1, sampled_idx.unsqueeze(-1)).squeeze(-1)
        return sampled_idx, p_sampled, logp_sampled

    def enforce_peptide_bond(self,pos, target_dist=1.33):
        pos = pos.clone()
        N_batch, N_res, _, _ = pos.shape
        for i in range(1, N_res):
            c_prev = pos[:, i - 1, 2, :]       
            n_curr = pos[:, i, 0, :]            
            v = n_curr - c_prev                 
            d = torch.norm(v, dim=-1, keepdim=True) 
            direction = v / (d + 1e-8)
            shift = (target_dist - d) * direction
            pos[:, i:, :, :] += shift[:, None, None, :]
        return pos

    def predict_step(self, batch, batch_idx):
        folder = self.folder
        merge_folder = self.merge_folder
        sequence = self.sequence
        N_batch = self.num_sample
        protein_name = self.protein_name
        N_seq = len(sequence) + 2
        sequence_tokens = encoding.tokenize_sequence(sequence, self.tokenizer_sequence, add_special_tokens=True)
        sequence_tokens = sequence_tokens.unsqueeze(0).repeat(N_batch,1).to(self.device)
        structure_tokens = torch.zeros_like(sequence_tokens)
        structure_tokens[:,0] = 4098
        structure_tokens[:,-1] = 4097
        temp = torch.ones(structure_tokens.shape[0],1).to(self.device).float()
        mask = (sequence_tokens != 1).to(self.device)
        prob = 1
        logprob = 0
        for block in self.model.transformer.blocks:
            block.attn.k_cache = torch.zeros(N_batch,N_seq * 2,1536).to(self.device)
            block.attn.v_cache = torch.zeros(N_batch,N_seq * 2,1536).to(self.device)
        for j in range(N_seq):
            logits = self.forward(seq_tokens=sequence_tokens,
                                  struc_tokens=structure_tokens[:,j:j + 1],
                                  temp = temp.unsqueeze(-1),
                                  mask = mask,
                                  T = 1,
                                  N_seg = 1,
                                  index = j)
            if j < N_seq - 2:
                sample_token,prob_token,logprob_token = self.sample_i_with_temperature(logits)
                prob *= prob_token
                logprob += logprob_token
                structure_tokens[:,j + 1] = sample_token
        structure_tokens[:,:N_seq] = (
            structure_tokens[:,:N_seq].where(sequence_tokens != 0, 4098)  # BOS
            .where(sequence_tokens != 2, 4097)  # EOS
            .where(sequence_tokens != 31, 4100)  # Chainbreak
        )
        bb_coords = (
            self.decoder.decode(
                structure_tokens,
                torch.ones_like(sequence_tokens),
                torch.zeros_like(sequence_tokens),
            )["bb_pred"].detach().cpu())
        bb_coords = self.enforce_peptide_bond(bb_coords)
        for k in range(N_batch):
            chain = ProteinChain.from_backbone_atom_coordinates(
            bb_coords[k:k+1], sequence="X" + sequence + "X")
            chain.infer_oxygen().to_pdb(f"{folder}/{protein_name}_{batch_idx}_{k}.pdb")
        np.save(f"{folder}/prob_{protein_name}_{batch_idx}.npy",prob.detach().cpu().numpy())
        np.save(f"{folder}/logprob_{protein_name}_{batch_idx}.npy",logprob.detach().cpu().numpy())

def sample(lightning_model,trainer,num_sample = 1000,sequence = None,batch_size = 250,protein_name = None):
    if protein_name is None:
        protein_name = "test"
    
    base = os.getcwd()   # current working directory
    folder = os.path.join(base, "stepone_merge",protein_name)
    merge_folder = os.path.join(base, "stepone_merge",protein_name)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(merge_folder, exist_ok=True)
    lightning_model.sequence = sequence
    lightning_model.num_sample = batch_size
    lightning_model.folder = folder
    lightning_model.merge_folder = merge_folder
    lightning_model.protein_name = protein_name
    dummy = torch.zeros(num_sample, 1)  # shape doesn't matter if you ignore it
    dummy_loader = DataLoader(TensorDataset(dummy), batch_size=batch_size)
    
    trainer.predict(lightning_model, dummy_loader)
    
    all_prob = []
    all_logprob = []
    pdb_files = []
    num_batches = math.ceil(num_sample / batch_size)
        
    for batch_idx in range(num_batches):
        for j in range(batch_size):
            sample_idx = batch_idx * batch_size + j
            if sample_idx >= num_sample:
                break  # no more samples to cover
            pdb_files.append(f"{folder}/{protein_name}_{batch_idx}_{j}.pdb")
        prob_path = f"{folder}/prob_{protein_name}_{batch_idx}.npy"
        logprob_path = f"{folder}/logprob_{protein_name}_{batch_idx}.npy"
        if not os.path.exists(prob_path):
            print(f"Skipping missing file: {prob_path}")
            continue
        prob = np.load(prob_path)         
        logprob = np.load(logprob_path) 
        all_prob.append(prob)
        all_logprob.append(logprob)
        merged_prob = np.concatenate(all_prob, axis=0)
        merged_logprob = np.concatenate(all_logprob, axis=0)
    prob_path = os.path.join(merge_folder, "sample_prob.npy")
    logprob_path = os.path.join(merge_folder, "sample_logprob.npy")
    np.save(prob_path,merged_prob)
    np.save(logprob_path,merged_logprob)
    if not pdb_files:
        raise FileNotFoundError(f"No .pdb files found in {folder}")
    traj_list = [md.load(pdb) for pdb in pdb_files]
    merged_traj = md.join(traj_list)
    merged_traj = remove_first_last_residue(merged_traj)
    xtc_path = os.path.join(merge_folder, "sample.xtc")
    merged_traj.save_xtc(xtc_path)
    pdb_path = os.path.join(merge_folder, "sample.pdb")
    merged_traj[0].save_pdb(pdb_path)   
    for pdb_file in pdb_files:
        os.remove(pdb_file)
    print(f"Merged {len(pdb_files)} PDB files into {xtc_path}")
    print(f"Saved first frame as {pdb_path}")
    print(f"Deleted original {len(pdb_files)} PDB files from {folder}")

def run_experiment(num_sample = 500,batch_size = 500,sequence = None,protein_name = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESM3_sm_open_v0("cpu").train()
    backbone_model = ESMDynamics(1536,24)
    backbone_model.seq_embed = model.encoder.sequence_embed
    backbone_model.struc_embed = model.encoder.structure_tokens_embed
    ckpt_path = hf_hub_download(
        repo_id="harrydirk41/LangBoltz",
        filename="LangBoltz_backbone.ckpt",
    )
    #lightning_model = ESM3LightningModule.load_from_checkpoint("LangBoltz_backbone.ckpt",model = backbone_model)
    lightning_model = ESM3LightningModule.load_from_checkpoint(ckpt_path,model = backbone_model)
    lightning_model.decoder = ESM3_structure_decoder_v0("cpu")
    lightning_model.struc_encoder = ESM3_structure_encoder_v0("cpu")
    lightning_model.tokenizer_sequence = model.tokenizers.sequence
    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None
    )
    sample(lightning_model,trainer,num_sample = num_sample,batch_size = batch_size,sequence = sequence,protein_name = protein_name)

