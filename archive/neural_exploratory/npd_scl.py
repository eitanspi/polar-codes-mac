"""
npd_scl.py — SCL decoder for NPDTree.

L paths through the recursive tree. At each info leaf, paths fork.
The embedding state is carried through the recursion via e_block
(batch dim = L). Path forking only affects decisions and path metrics,
not the embeddings — because all paths see the same channel output.
The only path-dependent input is cw_top (the decisions), which
enters the bitnode to produce different e_bot per path.
"""
import math
import numpy as np
import torch
from polar.encoder import bit_reversal_perm


def _crc8_compute(msg_bits):
    msg_int = 0
    for b in msg_bits:
        msg_int = (msg_int << 1) | int(b)
    msg_int <<= 8
    for i in range(len(msg_bits)):
        if msg_int & (1 << (len(msg_bits) + 7 - i)):
            msg_int ^= (0x107 << (len(msg_bits) - 1 - i))
    return msg_int & 0xFF


def _crc8_check(msg_bits, crc_bits):
    computed = _crc8_compute(msg_bits)
    crc_rx = 0
    for b in crc_bits:
        crc_rx = (crc_rx << 1) | int(b)
    return computed == crc_rx


class NPDListDecoder:
    def __init__(self, tree, L=4, crc_bits=0):
        self.tree = tree
        self.L = L
        self.crc_bits = crc_bits

    @torch.no_grad()
    def decode(self, emb, frozen_set, info_positions=None):
        """
        emb : (1, N, d) in NPD tree order
        frozen_set : set of 0-indexed natural positions
        Returns: (1, N) best path
        """
        assert emb.shape[0] == 1
        N = emb.shape[1]
        n = int(math.log2(N))
        br = bit_reversal_perm(n)
        L = self.L

        # State: L paths, each with decisions and path metric
        u_hat = torch.zeros(L, N, dtype=torch.long)
        pm = torch.full((L,), float('inf'))
        pm[0] = 0.0  # only 1 active path initially
        active = 1

        leaf_idx = [0]

        def _fork_and_prune(logit, nat_idx):
            """Fork active paths at an info leaf, prune to L best."""
            nonlocal u_hat, pm, active

            # Build 2*active candidates
            n_cand = 2 * active
            cand_pm = torch.zeros(n_cand)
            cand_parent = torch.zeros(n_cand, dtype=torch.long)
            cand_bit = torch.zeros(n_cand, dtype=torch.long)

            for i in range(active):
                llr = logit[i].item()
                # Convention: logit > 0 → model prefers bit=1
                # bit=0: penalty if logit > 0
                cand_pm[2*i] = pm[i] + (abs(llr) if llr > 0 else 0)
                cand_parent[2*i] = i
                cand_bit[2*i] = 0
                # bit=1: penalty if logit <= 0
                cand_pm[2*i+1] = pm[i] + (abs(llr) if llr <= 0 else 0)
                cand_parent[2*i+1] = i
                cand_bit[2*i+1] = 1

            # Keep best min(n_cand, L)
            new_L = min(n_cand, L)
            _, top_idx = cand_pm[:n_cand].topk(new_L, largest=False)
            top_idx = top_idx.sort().values

            new_u = torch.zeros(L, N, dtype=torch.long)
            new_pm = torch.full((L,), float('inf'))

            for j in range(new_L):
                idx = top_idx[j].item()
                parent = cand_parent[idx].item()
                bit = cand_bit[idx].item()
                new_u[j] = u_hat[parent].clone()
                new_u[j, nat_idx] = bit
                new_pm[j] = cand_pm[idx]

            u_hat = new_u
            pm = new_pm
            active = new_L

            # Return which parent each survivor came from (for reindexing e_block)
            reindex = torch.zeros(L, dtype=torch.long)
            for j in range(new_L):
                idx = top_idx[j].item()
                reindex[j] = cand_parent[idx]
            return reindex

        def _decode_scl(e_block):
            """
            e_block: (L, block_size, d)
            Returns: (L, block_size) codeword bits
            """
            nonlocal active
            block_size = e_block.shape[1]

            if block_size == 1:
                logit = self.tree.emb2llr(e_block[:, 0, :]).squeeze(-1)  # (L,)
                idx = leaf_idx[0]
                leaf_idx[0] += 1
                nat_idx = int(br[idx])

                if nat_idx in frozen_set:
                    # Frozen: all decide 0
                    # Convention: logit > 0 → model prefers bit=1
                    # Frozen forces bit=0, so penalty when logit > 0
                    dec = torch.zeros(L, dtype=torch.long)
                    for i in range(active):
                        if logit[i].item() > 0:
                            pm[i] += abs(logit[i].item())
                    u_hat[:active, nat_idx] = 0
                    return dec.unsqueeze(1)
                else:
                    # Info: fork and prune
                    reindex = _fork_and_prune(logit, nat_idx)
                    # No need to reindex e_block at leaves (size 1, already processed)
                    dec = torch.zeros(L, dtype=torch.long)
                    dec[:active] = u_hat[:active, nat_idx]
                    return dec.unsqueeze(1)

            e_odd = e_block[:, 0::2, :]
            e_even = e_block[:, 1::2, :]

            e_top = self.tree.checknode(torch.cat([e_odd, e_even], dim=-1))
            cw_top = _decode_scl(e_top)

            # CRITICAL: after left subtree, paths may have been reordered.
            # e_odd and e_even are from the CHANNEL (same for all paths
            # that share the same channel observation). Since we started
            # with L copies of the same embedding, all paths have the
            # same e_odd/e_even. So no reindexing needed for these.
            # BUT cw_top IS path-dependent (each path made different decisions).
            # bitnode(e_odd, e_even, cw_top) uses cw_top to differentiate paths.

            e_bot = self.tree.bitnode(e_odd, e_even, cw_top)
            cw_bot = _decode_scl(e_bot)

            cw = torch.zeros(L, block_size, dtype=torch.long)
            cw[:, 0::2] = cw_top ^ cw_bot
            cw[:, 1::2] = cw_bot
            return cw

        # Run with L copies of embedding
        emb_L = emb.expand(L, -1, -1).clone()
        _decode_scl(emb_L)

        # Select best path (optional CRC)
        if self.crc_bits > 0 and info_positions is not None:
            for i in sorted(range(active), key=lambda x: pm[x].item()):
                info_bits = [int(u_hat[i, p-1].item()) for p in info_positions]
                msg = info_bits[:-self.crc_bits]
                crc_rx = info_bits[-self.crc_bits:]
                if _crc8_check(msg, crc_rx):
                    return u_hat[i:i+1]

        best = pm[:active].argmin().item()
        return u_hat[best:best+1]


def eval_npd_scl(model_stage, channel, N, Au, Av, frozen_u_set,
                 n_cw=500, L=4, crc_bits=0, seed=999):
    from polar.encoder import polar_encode_batch
    br = bit_reversal_perm(int(math.log2(N)))
    br_t = torch.from_numpy(br.copy()).long()
    decoder = NPDListDecoder(model_stage.tree, L=L, crc_bits=crc_bits)
    model_stage.eval()
    rng = np.random.default_rng(seed)
    errs = 0
    for cw in range(n_cw):
        u = np.zeros(N, dtype=int); v = rng.integers(0, 2, N).astype(int)
        for p in Au: u[p-1] = rng.integers(0, 2)
        for p in Av: v[p-1] = rng.integers(0, 2)
        x = polar_encode_batch(u.reshape(1,-1))[0]
        y = polar_encode_batch(v.reshape(1,-1))[0]
        z = torch.from_numpy(channel.sample_batch(x.reshape(1,-1), y.reshape(1,-1))).float()
        emb = model_stage.encode_channel(z)
        emb_npd = emb[:, br_t, :]
        u_hat = decoder.decode(emb_npd, frozen_u_set,
                               info_positions=Au if crc_bits > 0 else None)
        if any(int(u_hat[0, p-1].item()) != int(u[p-1]) for p in Au):
            errs += 1
    return errs / n_cw
