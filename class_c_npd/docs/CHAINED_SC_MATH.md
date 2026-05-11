---
title: "Chained SC Decoding for MAC Polar Codes at Corner Rate Points"
date: "2026-04-13"
geometry: margin=1in
fontsize: 11pt
header-includes:
  - \usepackage{amsmath,amssymb}
---

# 1. System Model

Two users transmit binary codewords $\mathbf{x}, \mathbf{y} \in \{0,1\}^N$ over a multiple access channel (MAC). The channel output is

$$Z_i = f(X_i, Y_i, W_i), \quad i = 1, \ldots, N$$

where $W_i$ is the channel noise. For the Gaussian MAC (GMAC):

$$Z_i = (1 - 2X_i) + (1 - 2Y_i) + W_i, \quad W_i \sim \mathcal{N}(0, \sigma^2)$$

Each user encodes independently with a polar code: $\mathbf{x} = \mathbf{u} B_N F^{\otimes n}$ and $\mathbf{y} = \mathbf{v} B_N F^{\otimes n}$, where $\mathbf{u}, \mathbf{v} \in \{0,1\}^N$ are the message vectors, $F = \bigl[\begin{smallmatrix} 1 & 0 \\ 1 & 1 \end{smallmatrix}\bigr]$, and $B_N$ is the bit-reversal permutation.

# 2. Monotone Chain Paths and Corner Rates

The MAC capacity region is achieved by decoding the $2N$ message bits $(u_1, \ldots, u_N, v_1, \ldots, v_N)$ in a specific order, defined by a **path** $\mathbf{b} \in \{0, 1\}^{2N}$, where $b_t = 0$ means "decode a $U$-bit" and $b_t = 1$ means "decode a $V$-bit."

A monotone chain path has the form $\mathbf{b} = 0^i \, 1^N \, 0^{N-i}$ for some $0 \leq i \leq N$. The **corner rate points** are:

- **Class A** ($i = 0$): path $1^N 0^N$ — decode all of $V$ first, then all of $U$
- **Class C** ($i = N$): path $0^N 1^N$ — decode all of $U$ first, then all of $V$

# 3. Class C Decomposition

For Class C (path $0^N 1^N$), all $N$ bits of $U$ are decoded before any bit of $V$. This creates a natural two-stage decomposition:

**Stage 1 — Decode $U$ on the marginal channel:**  
When decoding $U$, no information about $V$ is available. The decoder sees the MAC output $\mathbf{z}$ and treats $Y$ as noise. The effective channel for $X$ is:

$$P(Z | X) = \sum_{Y \in \{0,1\}} P(Z | X, Y) \cdot P(Y) = \frac{1}{2} P(Z | X, Y{=}0) + \frac{1}{2} P(Z | X, Y{=}1)$$

This is a **binary-input mixture channel**. For the GMAC:

$$Z | X \sim \frac{1}{2}\mathcal{N}(1 - 2X + 1, \sigma^2) + \frac{1}{2}\mathcal{N}(1 - 2X - 1, \sigma^2)$$

Stage 1 applies standard single-user SC decoding of $\mathbf{u}$ on this marginal channel. The frozen set $\mathcal{F}_U$ is designed for this channel.

**Stage 2 — Decode $V$ on the clean channel:**  
After Stage 1, we have $\hat{\mathbf{u}}$ and can reconstruct $\hat{\mathbf{x}} = \hat{\mathbf{u}} B_N F^{\otimes n}$. Now the decoder sees both $\mathbf{z}$ and $\hat{\mathbf{x}}$. The effective channel for $Y$ is:

$$P(Z | Y, \hat{X}) = P(Z | \hat{X}, Y) \quad \text{(assuming } \hat{X} = X \text{)}$$

If Stage 1 decoded correctly ($\hat{\mathbf{x}} = \mathbf{x}$), this is a **clean single-user channel** — the interference from $X$ is perfectly subtracted. For the GMAC:

$$Z - (1 - 2\hat{X}) = (1 - 2Y) + W \quad \Rightarrow \quad \tilde{Z} | Y \sim \mathcal{N}(1 - 2Y, \sigma^2)$$

This is just a standard BPSK-AWGN channel. Stage 2 applies single-user SC decoding of $\mathbf{v}$ on this channel.

# 4. Chained SC Decoder

The full decoder chains the two stages:

\begin{enumerate}
\item \textbf{Compute channel features:} From $\mathbf{z}$, compute per-position LLRs for the mixture channel:
$$\Lambda_i^{(1)} = \log \frac{P(Z_i | X_i = 0)}{P(Z_i | X_i = 1)} = \log \frac{P(Z_i | X{=}0, Y{=}0) + P(Z_i | X{=}0, Y{=}1)}{P(Z_i | X{=}1, Y{=}0) + P(Z_i | X{=}1, Y{=}1)}$$
\item \textbf{Stage 1 SC decode:} Feed $\boldsymbol{\Lambda}^{(1)}$ to single-user SC decoder $\rightarrow \hat{\mathbf{u}}$
\item \textbf{Reconstruct:} $\hat{\mathbf{x}} = \hat{\mathbf{u}} B_N F^{\otimes n}$
\item \textbf{Compute clean LLRs:}
$$\Lambda_i^{(2)} = \log \frac{P(Z_i | \hat{X}_i, Y_i = 0)}{P(Z_i | \hat{X}_i, Y_i = 1)}$$
For GMAC: $\Lambda_i^{(2)} = \frac{2(Z_i - (1 - 2\hat{X}_i))}{\sigma^2}$
\item \textbf{Stage 2 SC decode:} Feed $\boldsymbol{\Lambda}^{(2)}$ to single-user SC decoder $\rightarrow \hat{\mathbf{v}}$
\end{enumerate}

# 5. Error Analysis

The total block error rate is bounded by:

$$P_e^{\text{total}} \leq P_e^{(1)} + P_e^{(2)} + P_e^{\text{cascade}}$$

where $P_e^{(1)}$ is the Stage 1 error rate on the mixture channel, $P_e^{(2)}$ is the Stage 2 error rate on the clean channel (conditioned on correct Stage 1), and $P_e^{\text{cascade}}$ accounts for error propagation when $\hat{\mathbf{x}} \neq \mathbf{x}$.

In practice, $P_e^{(2)} \ll P_e^{(1)}$ because the clean channel (with interference removed) is much better than the mixture channel. The dominant term is $P_e^{(1)}$.

# 6. Neural Chained Decoder

The chained NPD replaces both SC decoders with Neural Polar Decoders (NPDs), trained independently:

- **Stage 1 NPD:** Trained on the mixture channel with fast parallel cross-entropy (fast\_ce), $O(\log N)$ gradient depth
- **Stage 2 NPD:** Trained on the clean channel (given correct $\hat{\mathbf{x}}$), same fast\_ce training

Each NPD uses the same architecture: $z$-encoder $\rightarrow$ recursive CheckNode/BitNode tree $\rightarrow$ leaf decisions. The key advantage is **NPD-guided code design**: train at rate 1, measure per-position mutual information, select information positions where the NPD performs best (which may differ from the SC-optimal positions).
