# Literature Survey: Neural Decoders for Channel Coding and Multiple Access Channels

**Prepared for:** Research paper on "Neural SC Decoding of Polar Codes for the Two-User MAC"
**Date:** April 2026

---

## 1. Neural Decoders for Single-User Polar Codes

### 1.1 Foundational Work: Deep Learning for Channel Decoding

The application of deep learning to channel decoding was initiated by several seminal works in 2017.

**Gruber et al. (2017)** — "On Deep Learning-Based Channel Decoding"
- *Venue:* 51st Annual Conference on Information Sciences and Systems (CISS), IEEE, 2017
- *Contribution:* Revisited using deep neural networks for one-shot decoding of random and structured codes, including polar codes. Demonstrated that structured codes are easier to learn than random codes, and that neural networks can generalize to unseen codewords for structured codes, providing evidence that NNs learn a form of decoding algorithm rather than memorizing a lookup table.
- *Architecture:* Fully connected DNN trained on received vectors to output estimated codewords.
- *Limitation:* Scalability — training complexity grows exponentially with block length due to the size of the codebook.

**Cammerer et al. (2017)** — "Scaling Deep Learning-Based Decoding of Polar Codes via Partitioning"
- *Authors:* S. Cammerer, T. Gruber, J. Hoydis, S. ten Brink
- *Venue:* arXiv:1702.06901, 2017
- *Contribution:* Addressed the exponential scaling problem by partitioning the polar code encoding graph into smaller sub-blocks, training neural networks on each sub-block to approximate MAP decoding, and connecting them via conventional belief propagation (BP) stages. This was among the first works to demonstrate that hybrid neural/conventional architectures could scale to practical block lengths.
- *Channel:* AWGN
- *Key result:* NN-augmented BP decoder outperforms standard BP while maintaining manageable training complexity.

**O'Shea and Hoydis (2017)** — "An Introduction to Deep Learning for the Physical Layer"
- *Venue:* IEEE Transactions on Cognitive Communications and Networking, vol. 3, pp. 563-575, 2017
- *Contribution:* Proposed interpreting an entire communications system as an autoencoder, jointly optimizing transmitter and receiver as an end-to-end reconstruction task. Extended the concept to multiple transmitter/receiver networks. This paper established the autoencoder paradigm that many subsequent works in both single-user and multi-user settings have followed.

### 1.2 Neural Belief Propagation Decoders

**Nachmani et al. (2018)** — "Deep Learning Methods for Improved Decoding of Linear Codes"
- *Authors:* E. Nachmani, E. Marciano, L. Lugosch, W. Gross, D. Burshtein, Y. Be'ery
- *Venue:* IEEE J. Sel. Top. Signal Process., 2018
- *Contribution:* Showed that a standard BP decoder for linear codes (including polar codes) can be improved by assigning learnable weights to the edges of the Tanner graph, effectively parameterizing the BP iterations as a neural network. Tying parameters across iterations forms a recurrent neural network (RNN) architecture that requires significantly fewer parameters with comparable performance. This "weighted BP" approach became a foundation for model-based deep learning in channel coding.
- *Architecture:* Weighted factor graph / neural BP with learnable edge weights.

### 1.3 Neural Successive Cancellation (NSC) Decoders

**Doan and Bhatt (2018)** — "Neural Successive Cancellation Decoding of Polar Codes"
- *Venue:* IEEE 19th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), Kalamata, Greece, 2018
- *Contribution:* Proposed a neural successive cancellation (NSC) decoder that partitions the polar code into sub-blocks and replaces each sub-decoder with a small neural network, connected via the SC decoding framework. At the same error performance, the NSC decoder reduces decoding latency by up to 89% compared to SC, BP, and partitioned NN (PNN) decoders. For Polar(128,64), the NSC decoder matches PNN performance while reducing latency by 42.5%.
- *Architecture:* Constituent NN decoders connected via SC structure.
- *Relevance to our work:* This is among the earliest works to embed neural networks inside the SC tree structure, though it replaces entire sub-decoders rather than individual tensor operations (CalcLeft/CalcRight/CalcParent) as we do.

### 1.4 Neural Polar Decoder (NPD) — Aharoni et al.

This line of work by Aharoni, Pfister, Permuter, and Huleihel is the most directly related to our approach, as both replace SC decoder operations with neural networks.

**Aharoni et al. (2023/2024)** — "Data-Driven Neural Polar Codes for Unknown Channels With and Without Memory"
- *Authors:* Ziv Aharoni, Bashar Huleihel, Henry D. Pfister, Haim H. Permuter
- *Venue:* IEEE ISIT 2024 (arXiv:2309.03148, September 2023); extended version in IEEE Open Journal of Communications Society, 2024
- *Channel model:* Arbitrary memoryless channels and channels with memory (finite-state channels), treated as black boxes.
- *Architecture:* The Neural SC (NSC) decoder uses **three neural networks** that replace the core elements of the SC decoder:
  1. **Check-node NN** (replaces the f-function / CalcLeft operation)
  2. **Bit-node NN** (replaces the g-function / CalcRight operation)
  3. **Soft-decision NN** (replaces the hard/soft decision at leaves)
  
  A fourth **channel embedding NN** maps raw channel outputs into the SC decoder's input space, enabling the decoder to work without an explicit channel model.
- *Key results:* Provides theoretical consistency guarantees for the NSC decoder. Computational complexity is O(AN log N) where A is a user-defined budget independent of channel memory — a dramatic improvement over O(|S|^3 N log N) for SC trellis decoders on channels with memory of state-space size |S|.
- *Difference from our work:* The NPD operates on single-user channels and replaces SC operations with NNs trained per-channel. Our work applies a similar philosophy to the **two-user MAC**, where the tensor operations involve joint distributions over both users' bits, requiring fundamentally different NN architectures and training procedures. Moreover, we operate on the MAC-specific SC tree structure (with its monotone chain path structure and five extremal types) rather than the standard binary SC tree.

**Aharoni and Pfister (2025)** — "Neural Polar Decoders for Deletion Channels"
- *Venue:* arXiv:2507.12329, July 2025
- *Channel:* Deletion channels with constant deletion rate delta in {0.01, 0.1}
- *Contribution:* Extended NPD to deletion channels, reducing complexity from O(N^4) (existing trellis-based polar decoders for deletion channels) to O(AN log N). Only one of the four NNs needed modification to handle deletions. Enabled list decoding for deletion channels for the first time.
- *Relevance:* Demonstrates the generality of the NPD framework across channel types, though still single-user.

**Hirsch, Aharoni et al. (2025)** — "A Study of Neural Polar Decoders for Communication"
- *Authors:* Rom Hirsch, Ziv Aharoni, Henry D. Pfister, Haim H. Permuter
- *Venue:* NeurIPS 2025 (arXiv:2510.03069)
- *Contribution:* Extended NPDs to practical 5G systems including OFDM and single-carrier systems. Supports variable code lengths via rate matching, higher-order modulations, and pilotless decoding that exploits channel memory. NPDs consistently outperformed the 5G polar decoder in BER, BLER, and throughput, with largest gains for low-rate, short-block configurations common in 5G control channels.

**Aharoni et al. (2025)** — "Code Rate Optimization via Neural Polar Decoders"
- *Authors:* Ziv Aharoni, Bashar Huleihel, Henry D. Pfister, Haim H. Permuter
- *Venue:* IEEE ISIT 2025 (arXiv:2506.15836)
- *Contribution:* Used NPDs to simultaneously optimize code rates and input distributions by estimating mutual information of effective channels. Integrated the Honda-Yamamoto scheme for non-uniform input distributions. Demonstrated significant MI and BER improvements when capacity-achieving distributions are non-uniform.

**Aharoni et al. (2025)** — "Neural Polar Decoders for DNA Data Storage"
- *Venue:* arXiv:2506.17076, 2025
- *Contribution:* Applied NPDs to DNA storage channels.

### 1.5 Curriculum-Based and Transformer-Based Neural Polar Decoders

**Hebbar et al. (2023)** — "CRISP: Curriculum Based Sequential Neural Decoders for Polar Code Family"
- *Authors:* S. Ashwin Hebbar, Viraj Nadkarni, Ashok Vardhan Makkuva, Suma Bhat, Sewoong Oh, Pramod Viswanath
- *Venue:* ICML 2023
- *Contribution:* Designed a curriculum-based sequential neural decoder guided by information-theoretic insights. CRISP outperforms SC decoding and approaches near-optimal reliability for Polar(32,16) and Polar(64,22). Extended to Polarization-Adjusted Convolutional (PAC) codes, constructing the first data-driven PAC decoder with near-optimal performance on PAC(32,16).
- *Architecture:* Sequential neural decoder trained with curriculum learning — easier decoding tasks are presented first, gradually increasing difficulty.
- *Channel:* AWGN

**Hebbar et al. (2024)** — "DeepPolar: Inventing Nonlinear Large-Kernel Polar Codes via Deep Learning"
- *Authors:* S. Ashwin Hebbar et al.
- *Venue:* ICML 2024
- *Contribution:* Generalized polar codes by expanding the 2x2 Arikan kernel to a larger l x l kernel, parameterized by learnable MLPs. Both the nonlinear kernels and matched decoders are learned jointly. Setting kernel size l = sqrt(N) yields the best performance, outperforming both conventional polar codes and existing neural codes.
- *Architecture:* MLP-parameterized kernels in a polar coding framework.
- *Channel:* AWGN
- *Difference from our work:* DeepPolar modifies the code construction itself (nonlinear kernels) rather than the decoder for an existing code. Our work keeps standard polar codes and replaces SC decoder operations with NNs.

**Ankireddy et al. (2024)** — "Nested Construction of Polar Codes via Transformers"
- *Authors:* Sravan Kumar Ankireddy, S. Ashwin Hebbar, Heping Wan, Joonyoung Cho, Charlie Zhang
- *Venue:* IEEE ISIT 2024
- *Contribution:* Used self-attention transformers to iteratively construct the polar code reliability sequence (one information position at a time), exploiting the nested structure of polar codes. Transformer-designed codes outperform both 5G-NR and Density Evolution approaches for AWGN and Rayleigh fading channels.
- *Relevance:* Applies deep learning to code design rather than decoding, complementary to our decoder-focused approach.

### 1.6 Other Neural Polar Decoding Approaches

**Ebada et al. (2019)** — "Deep Learning-Based Polar Code Design"
- *Authors:* M. Ebada, S. Cammerer, A. Elkelesh, S. ten Brink
- *Venue:* 57th Annual Allerton Conference, 2019
- *Contribution:* Represented information/frozen bit indices as a trainable binary vector, relaxed to soft values for gradient-based optimization. Learns optimal frozen-set selection via deep learning.

**DL-Aided SCL Decoders (2020-2023):**
- Song et al. proposed a DL-aided adaptive SCL (DL-ASCL) decoder using an ANN predictor to select the list size at each decoding stage, achieving optimal error correction with 56% complexity reduction.
- Wang et al. proposed DL-aided SC-Flip decoders using LSTM networks to identify error bits, combining supervised and reinforcement learning for improved flipping decisions.
- Deep learning has been used for shifted-pruning in SCL decoding, reducing computational complexity while maintaining performance (IEEE, 2023).

### 1.7 Survey Papers

**Matsumine and Ochiai (2024)** — "Recent Advances in Deep Learning for Channel Coding: A Survey"
- *Venue:* IEEE Open Journal of Communications Society, vol. 5, pp. 6443-6481, 2024
- *Scope:* Comprehensive survey covering model-free and model-based deep learning for LDPC and polar codes. Covers DL-based code design, BP decoding, SC decoding, and end-to-end learned codes. Notably, the survey does not cover MAC or multi-user scenarios, highlighting the gap our work addresses.

---

## 2. Neural Decoders for Multiple Access Channels

### 2.1 The Gap in the Literature

Despite the extensive work on neural decoders for single-user polar codes (Section 1), **there is a striking absence of work on neural decoders for polar-coded MAC channels.** The polar code MAC literature (Sasoglu et al. 2013, Onay 2013, Ren et al. 2025) uses conventional algebraic/analytic decoders, while the neural decoder literature focuses exclusively on point-to-point channels. Our work bridges this gap.

The key challenge in MAC decoding that distinguishes it from the single-user case:
- The SC decoder operates on **joint distributions** p(u_i^A, u_j^B | y^N) over both users' bits
- The tree structure follows **monotone chain paths** through a 2D grid of (i,j) positions, not a simple binary tree
- Five types of extremal channels arise (frozen-frozen, frozen-free, free-frozen, free-free with XOR constraint, and free-free independent), compared to just two (frozen/free) in the single-user case
- CalcLeft/CalcRight/CalcParent operations act on 4-element tensors (joint distributions over two binary variables) rather than scalar LLRs

### 2.2 Conventional Polar MAC Decoders

For context, the conventional approaches to polar-coded MAC decoding include:

**Sasoglu, Abbe, and Telatar (2013)** — "Polar Codes for the Two-User Multiple-Access Channel"
- *Venue:* IEEE Trans. Inform. Theory, 2013
- *Contribution:* Original proof that polar codes achieve the capacity region of two-user MACs, with O(N log N) encoding and SC decoding complexity.

**Onay (2013)** — Reference SC decoder for polar-coded MAC
- *Contribution:* O(N^2) SC decoder operating on the full joint distribution space.

**Ren, Bhatt, and Mondelli (2025)** — Efficient O(N log N) MAC SC decoder for all monotone chain paths
- *Contribution:* The interleaved decoder we use as baseline, achieving O(N log N) complexity for arbitrary paths through the MAC capacity region.

### 2.3 Deep Learning for Multi-User Detection (Without Channel Coding)

While no prior work addresses neural decoding of polar-coded MACs, there is substantial work on neural multi-user detection (MUD) without explicit channel coding structure:

**Ye, Li, and Juang (2017)** — "Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems"
- *Venue:* IEEE Wireless Communications Letters, 2017
- *Contribution:* Demonstrated that DNNs can implicitly learn channel state information and directly recover transmitted symbols in OFDM, bypassing explicit channel estimation. While focused on single-user OFDM, this established the paradigm of using NNs to replace model-based signal processing that was later applied to multi-user settings.

**Farsad and Goldsmith (2018)** — "Neural Network Detection of Data Sequences in Communication Systems"
- *Venue:* IEEE Trans. Signal Processing, 2018
- *Contribution:* Proposed sliding bidirectional RNN (SBRNN) for sequence detection that works without knowledge of the underlying channel model or CSI. Demonstrated on Poisson channels applicable to optical and molecular communications.

**DNN-based MIMO-NOMA detection (2019-2024):**
Multiple works have proposed DNNs to replace or augment conventional SIC receivers for NOMA multi-user detection:
- A deep learning approach for MIMO-NOMA downlink signal detection (Sensors, 2019) showed that DNNs can learn the nonlinear mapping from received signals to individual users' data.
- Feedback DNN (FDNN) receivers have been proposed to replace SIC, mitigating error propagation (Electronics, 2024).
- Bi-LSTM based joint detection for NOMA-OFDM systems (Sensors, 2022) showed superior SER compared to traditional methods and CNNs.
- Graph Neural Networks for joint detection-decoding in specialized channels (Electronics, 2024).

*Difference from our work:* These approaches treat detection as a black-box classification problem, without exploiting the polar code structure. Our approach embeds neural networks inside the structured SC tree-walk, preserving the O(N log N) complexity guarantees and interpretability of the SC framework while gaining the ability to operate on channels without closed-form expressions.

---

## 3. Deep Learning for NOMA

### 3.1 End-to-End DL-NOMA Systems

**Gui et al. (2018)** — "Deep Learning for an Effective Non-Orthogonal Multiple Access Scheme"
- *Authors:* G. Gui, H. Huang, Y. Song, H. Sari
- *Venue:* IEEE Transactions on Vehicular Technology, vol. 67, no. 9, pp. 8440-8450, 2018
- *Contribution:* Proposed a hybrid-cascaded DNN architecture with trainable modules that optimizes the entire two-user NOMA system in an end-to-end manner. The DNN learns the mapping between base-station inputs and user outputs to combat channel fading and noise, without requiring explicit channel estimation.
- *Architecture:* Nine trainable DNN modules in a cascaded structure.
- *Difference from our work:* Operates at the modulation/detection level without structured channel codes; does not exploit polar code structure.

**Deep Learning-Based SIC for NOMA (2020):**
- CNN-based SIC schemes have been proposed to improve NOMA system performance, effectively mitigating losses from imperfect SIC. The CNN learns to cancel interference more robustly than model-based SIC.

### 3.2 Polar-Coded NOMA with Deep Learning

**PC-NOMA with Deep Learning (2021):**
- *Venue:* Springer, Lecture Notes in Electrical Engineering, 2021 ("A Study on the Adaptability of Deep Learning-Based Polar-Coded NOMA in Ultra-Reliable Low-Latency Communications")
- *Contribution:* Studied the combination of polar codes with NOMA in URLLC scenarios, using deep learning for decoding. Found that the combination is promising for IoT environments but primarily focused on the NOMA detection layer, using conventional polar decoders after SIC-based user separation.
- *Difference from our work:* Separates multi-user detection (via DL) from channel decoding (via conventional SC/SCL), treating them as independent stages. Our approach performs joint decoding of both users' polar codes in a single SC tree-walk.

### 3.3 DL for SCMA (Sparse Code Multiple Access)

**DL-SCMA and AE-SCMA (2019):**
- *Venue:* Neurocomputing, 2019 (arXiv:1906.03169)
- *Contribution:* Proposed DNN-based SCMA decoders (DL-SCMA) and autoencoder-based SCMA codebook design (AE-SCMA). The DNN SCMA decoder significantly outperformed conventional Message Passing Algorithm (MPA) in BER, SER, and complexity. The AE-SCMA jointly designed codebooks and decoders via end-to-end learning.
- *Architecture:* DNN for decoding; autoencoder for joint codebook/decoder design with sparse connectivity and weight sharing.

**Residual Learning-Aided Convolutional Autoencoder for SCMA (2023):**
- *Venue:* IEEE, 2023
- *Contribution:* Combined CNNs with residual networks for SCMA encoding/decoding, with multitask learning to improve decoding accuracy.

### 3.4 Survey on Deep Learning for NOMA

**Ahsan et al. (2023)** — "A Survey of Deep Learning Based NOMA: State of the Art, Key Aspects, Open Challenges and Future Trends"
- *Venue:* Sensors, vol. 23, no. 6, 2023
- *Scope:* Comprehensive survey covering DL applications to SIC, CSI estimation, power allocation, resource allocation, user pairing, and transceiver design in NOMA systems. Emphasizes that DL-based NOMA can improve throughput, BER, latency, and resource efficiency. Covers integration with IRS, MEC, SWIPT, OFDM, and MIMO.
- *Notable finding:* The survey identifies that most DL-NOMA work focuses on the physical-layer detection problem (replacing SIC) or resource allocation, with very limited work on joint coding-detection optimization. This gap aligns with our contribution.

---

## 4. Autoencoder-Based Joint Coding for MAC

### 4.1 Foundational Autoencoder Approaches

**O'Shea and Hoydis (2017)** — (See Section 1.1)
- Extended the autoencoder concept to networks of multiple transmitters and receivers, laying the foundation for autoencoder-based MAC design.

**Aoudia and Hoydis (2019)** — "Model-Free Training of End-to-End Communication Systems"
- *Venue:* IEEE J. Sel. Areas Commun., vol. 37, no. 11, pp. 2503-2516, 2019
- *Contribution:* Proposed model-free training of autoencoder-based systems using policy gradient methods from reinforcement learning, eliminating the need for a differentiable channel model. This enabled training over actual channels, critical for MAC scenarios where channel models may be complex or unknown.

### 4.2 Autoencoder-Based MAC Constellation Design

**Gorelenkov and Vaezi (2025)** — "Deep Autoencoder-Based Constellation Design in Multiple Access Channels"
- *Venue:* ISIT 2025 (arXiv:2505.00868, May 2025)
- *Channel:* K-user MAC with additive noise
- *Contribution:* Proposed DAE-based constellation design for MAC that creates interference-aware constellations, reducing SER and enhancing constellation-constrained sum capacity. Demonstrated that DAE-designed constellations match or outperform analytically derived constellations, and extend to >2 users where no analytical solutions exist.
- *Architecture:* Multiple encoder NNs (one per user) and a shared decoder NN, trained end-to-end.
- *Difference from our work:* Focuses on uncoded constellation design (modulation level) rather than channel coding. Does not use polar codes or exploit algebraic code structure.

### 4.3 End-to-End Learned Codes

**Kim et al. (2018)** — "Deepcode: Feedback Codes via Deep Learning"
- *Authors:* Hyeji Kim, Yihan Jiang, Sreeram Kannan, Sewoong Oh, Pramod Viswanath
- *Venue:* NeurIPS 2018
- *Contribution:* Constructed the first family of deep learning-based codes that significantly outperform state-of-the-art codes for AWGN channels with feedback (3 orders of magnitude in reliability, 3 dB gain). Used RNN-based encoders and decoders trained jointly.
- *Relevance:* Demonstrated that deep learning can discover codes that outperform decades of mathematical design, motivating neural approaches to coding problems including MAC coding.

**Jiang et al. (2019)** — "Turbo Autoencoder: Deep Learning Based Channel Codes for Point-to-Point Communication Channels"
- *Venue:* NeurIPS 2019
- *Contribution:* Fully end-to-end jointly trained neural encoder and decoder (TurboAE) that approaches state-of-the-art performance under canonical channels and outperforms under non-canonical settings. Demonstrated that channel coding design can be automated via deep learning.
- *Architecture:* CNN-based encoder with interleaver structure inspired by turbo codes; RNN/CNN-based decoder.

### 4.4 Joint Source-Channel Coding for MAC

**Bourtsoulatze et al. (2019)** — "Deep Joint Source-Channel Coding for Wireless Image Transmission"
- *Venue:* IEEE Trans. Cognitive Commun. Networking, 2019
- *Contribution:* Proposed deep JSCC using CNN-based encoder/decoder, directly mapping pixel values to channel symbols without explicit source/channel coding separation. Outperforms JPEG/JPEG2000 + capacity-achieving channel code at low SNR.
- *Extensions to MAC:* Subsequent work (2024-2025) extended DeepJSCC to MAC scenarios using VAE-based distributed source coding over AWGN MAC, exploiting regularization to enhance reconstruction quality.

### 4.5 Autoencoder for Interference Channels

**End-to-End Autoencoder with Interference Suppression (2022):**
- *Venue:* arXiv:2201.01388
- *Contribution:* Designed autoencoder-based transceivers that jointly optimize coding and modulation for interference channels, learning to suppress multi-user interference without explicit interference cancellation.

---

## 5. Summary and Positioning of Our Work

### 5.1 Literature Landscape Summary

The literature on neural approaches to channel coding and multi-user communications can be organized along two axes:

| | **Single-User** | **Multi-User (MAC/NOMA)** |
|---|---|---|
| **Neural decoder for specific code** | NPD (Aharoni et al.), NSC (Doan), CRISP, DL-aided SC/SCL/BP | **[Gap — our work fills this]** |
| **End-to-end learned code** | DeepCode, TurboAE, DeepPolar | MAC autoencoders, DL-SCMA |
| **Neural detection (no coding)** | Farsad & Goldsmith | DNN-SIC, MIMO-NOMA detection |

### 5.2 Key Differentiators of Our Approach

Our work — neural SC decoding of polar codes for the two-user MAC — is distinguished from all prior work in several fundamental ways:

1. **First neural decoder for structured MAC codes.** While Aharoni et al.'s NPD replaces SC operations with NNs for single-user channels, no prior work has applied this approach to the MAC, where the decoder must jointly decode two users' polar codewords by traversing monotone chain paths through a 2D grid of joint bit decisions.

2. **Tensor-level neural replacement, not black-box.** Unlike DNN-based NOMA detectors that treat the entire detection problem as a black-box, we preserve the SC tree structure and replace only the CalcLeft, CalcRight, and CalcParent tensor operations with MLPs. This maintains the O(N log N) complexity of the SC decoder, provides interpretability, and allows the decoder to generalize across different frozen-set configurations.

3. **Joint distributions over two users.** The NPD operates on scalar LLRs (or soft values for a single bit). Our tensor operations process 4-element tensors representing joint distributions p(u_A, u_B) over both users' bits. This requires fundamentally different NN input/output structures and training procedures.

4. **MAC-specific tree structure.** The single-user SC decoder traverses a binary tree of depth log N. The MAC SC decoder traverses a path through a 2D grid, interleaving decisions about User A and User B bits according to a monotone chain path. The CalcLeft/CalcRight operations combine tensors from different positions in this interleaved structure, and CalcParent must handle the five extremal types (FF, FI, IF, II-XOR, II-independent).

5. **Channel-model-free MAC decoding.** Like the NPD, our approach does not require an explicit channel model — it learns the channel implicitly from training data. This is particularly valuable for MAC channels where computing exact transition probabilities may be intractable (e.g., Gaussian MAC with specific SNR and power constraints).

6. **Comparison to end-to-end approaches.** Unlike autoencoder-based MAC systems that learn both encoding and decoding from scratch, we leverage the proven capacity-achieving properties of polar codes for MACs (Sasoglu et al. 2013) and only replace the decoder's internal operations. This provides the benefits of structured codes (proven capacity achievement, known encoding complexity, rate flexibility) while gaining the adaptability of neural networks.

### 5.3 Closest Related Works

| Paper | Similarity | Key Difference |
|---|---|---|
| Aharoni et al. (2024) — NPD | Same philosophy: replace SC operations with NNs | Single-user only; scalar LLRs not joint tensors |
| Doan (2018) — NSC | Neural networks inside SC structure | Single-user; replaces sub-decoders not individual operations |
| CRISP (Hebbar 2023) | Sequential neural decoder for polar codes | Single-user; curriculum-based training differs from our MLP replacement |
| DeepPolar (Hebbar 2024) | Neural parameterization of polar code components | Modifies the code itself (nonlinear kernels), not the decoder for standard codes |
| DNN-NOMA (Gui 2018) | Neural network for multi-user channels | No channel coding structure; detection-level only |
| PC-NOMA + DL (2021) | Polar codes + NOMA + deep learning | Separates detection and decoding; uses conventional polar decoder |
| MAC autoencoder (Gorelenkov 2025) | Autoencoder for MAC | Uncoded constellation design, not structured channel coding |

### 5.4 Open Problems Our Work Addresses

1. **Neural decoding for structured MAC codes** — no prior work exists.
2. **Scalability of MAC decoders to channels without closed-form models** — conventional MAC SC decoders require exact channel transition probabilities.
3. **Complexity reduction for MAC decoding** — by learning compact NN representations of the tensor operations, we can potentially reduce the constant factors in O(N log N) decoding complexity.
4. **Robustness to channel model mismatch** — neural decoders trained on real channel data can adapt to conditions not captured by idealized models.

---

## References

### Neural Decoders for Polar Codes (Single-User)
- [Gruber et al., "On Deep Learning-Based Channel Decoding," CISS, 2017](https://arxiv.org/abs/1701.07738)
- [Cammerer et al., "Scaling Deep Learning-based Decoding of Polar Codes via Partitioning," 2017](https://arxiv.org/abs/1702.06901)
- [Nachmani et al., "Deep Learning Methods for Improved Decoding of Linear Codes," IEEE JSTSP, 2018](https://arxiv.org/abs/1706.07043)
- [Doan and Bhatt, "Neural Successive Cancellation Decoding of Polar Codes," IEEE SPAWC, 2018](https://ieeexplore.ieee.org/document/8445986/)
- [Ebada et al., "Deep Learning-Based Polar Code Design," Allerton, 2019](https://arxiv.org/abs/1909.12035)
- [Hebbar et al., "CRISP: Curriculum Based Sequential Neural Decoders for Polar Code Family," ICML, 2023](https://proceedings.mlr.press/v202/hebbar23a.html)
- [Hebbar et al., "DeepPolar: Inventing Nonlinear Large-Kernel Polar Codes via Deep Learning," ICML, 2024](https://arxiv.org/abs/2402.08864)
- [Ankireddy et al., "Nested Construction of Polar Codes via Transformers," IEEE ISIT, 2024](https://arxiv.org/abs/2401.17188)

### Neural Polar Decoder (NPD) — Aharoni et al.
- [Aharoni et al., "Data-Driven Neural Polar Codes for Unknown Channels With and Without Memory," IEEE ISIT, 2024](https://arxiv.org/abs/2309.03148)
- [Hirsch, Aharoni et al., "A Study of Neural Polar Decoders for Communication," NeurIPS, 2025](https://arxiv.org/abs/2510.03069)
- [Aharoni et al., "Code Rate Optimization via Neural Polar Decoders," IEEE ISIT, 2025](https://arxiv.org/abs/2506.15836)
- [Aharoni and Pfister, "Neural Polar Decoders for Deletion Channels," 2025](https://arxiv.org/abs/2507.12329)
- [Aharoni et al., "Neural Polar Decoders for DNA Data Storage," 2025](https://arxiv.org/abs/2506.17076)

### Autoencoder / End-to-End Learning
- [O'Shea and Hoydis, "An Introduction to Deep Learning for the Physical Layer," IEEE TCCN, 2017](https://arxiv.org/abs/1702.00832)
- [Kim et al., "Deepcode: Feedback Codes via Deep Learning," NeurIPS, 2018](https://papers.nips.cc/paper/8154-deepcode-feedback-codes-via-deep-learning)
- [Jiang et al., "Turbo Autoencoder," NeurIPS, 2019](https://arxiv.org/abs/1911.03038)
- [Aoudia and Hoydis, "Model-Free Training of End-to-End Communication Systems," IEEE JSAC, 2019](https://arxiv.org/abs/1804.02276)
- [Gorelenkov and Vaezi, "Deep Autoencoder-Based Constellation Design in Multiple Access Channels," ISIT, 2025](https://arxiv.org/abs/2505.00868)

### Deep Learning for NOMA / Multi-User Detection
- [Gui et al., "Deep Learning for an Effective Non-Orthogonal Multiple Access Scheme," IEEE TVT, 2018](https://ieeexplore.ieee.org/document/8387468/)
- [Ahsan et al., "A Survey of Deep Learning Based NOMA," Sensors, 2023](https://www.mdpi.com/1424-8220/23/6/2946)
- [DL-SCMA, "A Novel Deep Neural Network Based Approach for Sparse Code Multiple Access," Neurocomputing, 2019](https://arxiv.org/abs/1906.03169)

### Neural Detection and Channel Estimation
- [Farsad and Goldsmith, "Neural Network Detection of Data Sequences in Communication Systems," IEEE TSP, 2018](https://arxiv.org/abs/1802.02046)
- [Ye, Li, Juang, "Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems," IEEE WCL, 2017](https://arxiv.org/abs/1708.08514)

### Polar Codes for MAC (Conventional)
- [Sasoglu, Abbe, Telatar, "Polar Codes for the Two-User Multiple-Access Channel," IEEE TIT, 2013](https://arxiv.org/abs/1006.4255)

### Surveys
- [Matsumine and Ochiai, "Recent Advances in Deep Learning for Channel Coding: A Survey," IEEE OJCOMS, 2024](https://arxiv.org/abs/2406.19664)

### Joint Source-Channel Coding
- [Bourtsoulatze et al., "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE TCCN, 2019](https://arxiv.org/abs/1809.01733)
