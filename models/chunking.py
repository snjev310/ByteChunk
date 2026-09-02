import torch
import torch.nn as nn
from typing import List, Tuple

class DynamicChunking(nn.Module):
    """
    Converts a byte sequence + hard boundaries into chunk-level representations
    by selecting the boundary-marked embedding for each chunk.

    Returns per-item lists (variable number of chunks per sequence).
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        h: torch.Tensor,
        b: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> dict:
        """
        Returns dict with:
          chunk_embs   : List[Tensor[C_i, D]]   — one per batch item
          chunk_ids    : List[Tensor[T_i]]       — chunk index per byte position
          chunk_sizes  : List[Tensor[C_i]]       — bytes per chunk
          boundary_pos : List[Tensor[C_i]]       — byte position of each chunk boundary
        """
        
        B, T, D = h.shape
        chunk_embs_batch = []
        chunk_ids_batch = []
        chunk_sizes_batch = []
        boundary_pos_batch = []
        
        for i in range(B):
            # Get the embeddings and boundaries for this item
            h_i = h[i]
            b_i = b[i]
            
            # Padding to ensure we capture the last chunk if it ends at the end of the sequence
            if attention_mask is not None:
                valid_length = int(attention_mask[i].sum().item())
            else:
                valid_length = T
            
            h_i = h_i[:valid_length]
            b_i = b_i[:valid_length].float()
            
            # Assign chunk IDs via cumulative sum of hard boundaries
            # b[0] may be 0 or 1; if b[0] is 1, it starts a new chunk immediately
            # b_hard = (b_i > 0.3).long()  # Convert to binary
            b_hard = (b_i > 0.3).long()
            chunk_ids = torch.cumsum(b_hard, dim=0)
            
            # Normalize chunk IDs to start from 0
            chunk_ids = chunk_ids - chunk_ids[0].item()
            
            num_chunks = int(chunk_ids.max().item()) + 1
            
            # Select the boundary token of each chunks
            # for chunk c: the boundary position is the first token where chunk_ids == c
            # i.e where b_t == 1 caused the transition into chunk c
            chunk_embs = []
            chunk_sizes = []
            boundary_pos = []
            
            for c in range(num_chunks):
                positions = (chunk_ids == c).nonzero(as_tuple=False).squeeze(1)
                if positions.numel() == 0:
                    continue
                
                # Select the First position in this chunk (the boundary token)
                bpos = positions[0].item()
                chunk_embs.append(h_i[bpos])
                chunk_sizes.append(positions.numel())
                boundary_pos.append(bpos)
            
            chunk_embs_batch.append(torch.stack(chunk_embs, dim=0))
            chunk_ids_batch.append(chunk_ids)
            chunk_sizes_batch.append(torch.tensor(chunk_sizes, dtype=torch.long, device=h.device))
            boundary_pos_batch.append(torch.tensor(boundary_pos, dtype=torch.long, device=h.device))
        return {
            "chunk_embs": chunk_embs_batch,
            "chunk_ids": chunk_ids_batch,
            "chunk_sizes": chunk_sizes_batch,
            "boundary_pos": boundary_pos_batch,
        }

def pad_chunks(
    chunk_embs_batch: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of [C_i, D] tensors into [B, C_max, D] with an attention mask.

    Args:
        chunk_embs_batch: List of [C_i, D] float tensors

    Returns:
        padded : [B, C_max, D]
        mask   : [B, C_max]  — 1 for real chunks, 0 for padding
    """
    
    B = len(chunk_embs_batch)
    D = chunk_embs_batch[0].shape[1]
    C_max = max(c.shape[0] for c in chunk_embs_batch)
    
    device = chunk_embs_batch[0].device
    dtype = chunk_embs_batch[0].dtype
    
    padded = torch.zeros((B, C_max, D), device=device, dtype=dtype)
    mask = torch.zeros((B, C_max), device=device, dtype=torch.long)
    
    for i, c in enumerate(chunk_embs_batch):
        n = c.shape[0]
        padded[i, :n] = c
        mask[i, :n] = 1
    
    return padded, mask