# Literature Survey: Neural Decoders for Multiple Access Channels and Polar Codes

**Prepared for:** Research paper on "Neural SC Decoding of Polar Codes for the Two-User MAC"
**Date:** April 2026
**Last updated:** April 3, 2026

---

## 1. Neural Decoders for MAC

### 1.1 The Gap in the Literature

Despite extensive work on neural decoders for single-user polar codes (Section 3) and on deep learning for multi-user detection (Section 1.3), **there is a striking absence of work on neural decoders for polar-coded MAC channels.** The polar code MAC literature (Sasoglu et al. 2013, Onay 2013, Ren et al. 2025) uses conventional algebraic/analytic decoders, while the neural decoder literature focuses exclusively on point-to-point channels. Our work bridges this gap.

The key challenge in MAC decoding that distinguishes it from the single-user case:
- The SC decoder operates on **joint distributions** p(u_i^A, u_j^B | y^N) over both users' bits
- The tree structure follows **monotone chain paths** through a 2D grid of (i,j) positions, not a simple binary tree
- Five types of extremal channels arise (frozen-frozen, frozen-free, free-frozen, free-free with XOR constraint, and free-free independent), compared to just two (frozen/free) in the single-user case
- CalcLeft/CalcRight/CalcParent operations act on 4-element tensors (joint distributions over two binary variables) rather than scalar LLRs

### 1.2 Deep Learning for Multi-User Detection (Without Channel Coding)

While no prior work addresses neural decoding of polar-coded MACs, there is substantial work on neural multi-user detection (MUD) without explicit channel coding structure:

**Ye, Li, and Juang (2017)** -- "Power of Deep Learning for Channel Estimation and Signal Detection in OFDM Systems"
- *Venue:* IEEE Wireless Communications Letters, 2017
- *Contribution:* Demonstrated that DNNs can implicitly learn channel state information and directly recover transmitted symbols in OFDM, bypassing explicit channel estimation. While focused on single-user OFDM, this established the paradigm of using NNs to replace model-based signal processing that was later applied to multi-user settings.

**Farsad and Goldsmith (2018)** -- "Neural Network Detection of Data Sequences in Communication Systems"
- *Venue:* IEEE Trans. Signal Processing, 2018
- *Contribution:* Proposed sliding bidirectional RNN (SBRNN) for sequence detection that works without knowledge of the underlying channel model or CSI. Demonstrated on Poisson channels applicable to optical and molecular communications.

**DNN-based MIMO-NOMA detection (2019--2024):**
Multiple works have proposed DNNs to replace or augment conventional SIC receivers for NOMA multi-user detection:
- A deep learning approach for MIMO-NOMA downlink signal detection (Sensors, 2019) showed that DNNs can learn the nonlinear mapping from received signals to individual users' data.
- Feedback DNN (FDNN) receivers have been proposed to replace SIC, mitigating error propagation (Electronics, 2024).
- Bi-LSTM based joint detection for NOMA-OFDM systems (Sensors, 2022) showed superior SER compared to traditional methods and CNNs.

*Difference from our work:* These approaches treat detection as a black-box classification problem, without exploiting the polar code structure. Our approach embeds neural networks inside the structured SC tree-walk, preserving the O(N log N) complexity guarantees and interpretability of the SC framework while gaining the ability to operate on channels without closed-form expressions.

### 1.3 Deep Learning for Joint Detection and Decoding

**Xu et al. (2019)** -- "Deep Learning for Joint MIMO Detection and Channel Decoding"
- *Venue:* IEEE PIMRC, 2019 (arXiv:1901.05647)
- *Contribution:* Proposed a deep learning framework for joint MIMO detection and LDPC channel decoding, replacing the conventional separated detection-then-decoding pipeline with an end-to-end neural approach. The joint optimization outperforms the separated approach.
- *Architecture:* DNN that combines detection and decoding into a single network.
- *Difference from our work:* Targets single-user MIMO with LDPC codes; does not exploit polar code structure or address the MAC setting.

**Deep Learning Based Joint Detection and Decoding of NOMA (2019)**
- *Venue:* IEEE Globecom, 2019
- *Contribution:* Constructed neural network detectors and decoders based on message passing and belief propagation algorithms for NOMA systems. The decoder is cascaded with the detector to form a larger neural network using weighted edges on factor graphs.
- *Difference from our work:* Operates on generic factor graphs for NOMA detection, not on the polar code SC tree structure. Does not exploit the specific algebraic structure of polar codes for MAC.

### 1.4 Deep Learning for NOMA

**Gui et al. (2018)** -- "Deep Learning for an Effective Non-Orthogonal Multiple Access Scheme"
- *Authors:* G. Gui, H. Huang, Y. Song, H. Sari
- *Venue:* IEEE Transactions on Vehicular Technology, vol. 67, no. 9, pp. 8440-8450, 2018
- *Contribution:* Proposed a hybrid-cascaded DNN architecture with nine trainable modules that optimizes the entire two-user NOMA system in an end-to-end manner. The DNN learns the mapping between base-station inputs and user outputs to combat channel fading and noise, without requiring explicit channel estimation.
- *Difference from our work:* Operates at the modulation/detection level without structured channel codes; does not exploit polar code structure.

**Deep Learning-Based SIC for NOMA (2020--2025):**
- CNN-based SIC schemes have been proposed to improve NOMA system performance, effectively mitigating losses from imperfect SIC (Energies, 2020).
- DL-aided SIC for MIMO-NOMA (IEEE ICC, 2021) using deep learning to enhance successive interference cancellation.
- DL-based decoding in NOMA for high altitude platform systems (HAPS) using CNN (Telecommunication Systems, 2025).
- ResNet50V2 and InceptionResNetV2 models trained for BER improvement in NOMA-VLC systems (Optical and Quantum Electronics, 2023).

**PC-NOMA with Deep Learning (2021):**
- *Venue:* Springer, Lecture Notes in Electrical Engineering, 2021
- *Contribution:* Studied the combination of polar codes with NOMA in URLLC scenarios, using deep learning for decoding. Found that the combination is promising for IoT environments but primarily focused on the NOMA detection layer, using conventional polar decoders after SIC-based user separation.
- *Difference from our work:* Separates multi-user detection (via DL) from channel decoding (via conventional SC/SCL), treating them as independent stages. Our approach performs joint decoding of both users' polar codes in a single SC tree-walk.

**DL for SCMA (Sparse Code Multiple Access) (2019--2023):**
- DNN-based SCMA decoders (DL-SCMA) and autoencoder-based SCMA codebook design (AE-SCMA) proposed in Neurocomputing, 2019 (arXiv:1906.03169). The DNN SCMA decoder significantly outperformed conventional Message Passing Algorithm (MPA).
- Residual learning-aided convolutional autoencoder for SCMA (IEEE, 2023) with multitask learning.

**NOMANet (2025)** -- "A Graph Neural Network Enabled Power Allocation Scheme for NOMA"
- *Venue:* arXiv:2502.05592, February 2025
- *Contribution:* GNN-based power allocation for NOMA using multi-head attention and residual/dense connections. Trained unsupervised, achieves near-optimal performance with 700x faster inference than successive convex approximation.
- *Difference from our work:* Focuses on resource allocation (power control), not decoding.

**Ahsan et al. (2023)** -- "A Survey of Deep Learning Based NOMA: State of the Art, Key Aspects, Open Challenges and Future Trends"
- *Venue:* Sensors, vol. 23, no. 6, 2023
- *Scope:* Comprehensive survey covering DL applications to SIC, CSI estimation, power allocation, resource allocation, user pairing, and transceiver design in NOMA systems. Emphasizes that DL-based NOMA can improve throughput, BER, latency, and resource efficiency.
- *Notable finding:* The survey identifies that most DL-NOMA work focuses on the physical-layer detection problem (replacing SIC) or resource allocation, with very limited work on joint coding-detection optimization. This gap aligns with our contribution.

### 1.5 Polar-Coded NOMA with Iterative Detection-Decoding

**Ebert et al. (2020)** -- "Iterative Detection and Decoding of Finite-Length Polar Codes in Gaussian Multiple Access Channels"
- *Venue:* IEEE ICC 2021 (arXiv:2012.01075, December 2020)
- *Channel:* Gaussian MAC with finite number of users
- *Contribution:* Implemented iterative detection and decoding NOMA receivers based on the interleave-division multiple-access (IDMA) concept with BP-based polar decoders. Demonstrated with 5G polar codes (N=512) for two-user and four-user GMAC.
- *Key result:* The IDMA-style iterative approach scales almost linearly with the number of active users while maintaining good performance.
- *Difference from our work:* Uses conventional BP decoders within an iterative detection-decoding loop; treats detection and decoding as separate (though iteratively coupled) stages. Our approach performs truly joint SC decoding over the MAC tree in a single pass.

**Polar Coding and Early SIC Decoding for Uplink Heterogeneous NOMA (2025)**
- *Venue:* PMC/MDPI, 2025
- *Contribution:* Proposes a novel interleaver design for polar codes that enables early decoding in SIC frameworks for heterogeneous NOMA, with minimal modification to the 5G NR polar coding scheme.
- *Difference from our work:* Uses conventional polar decoders with modified interleaving; does not employ neural networks.

---

## 2. Polar Codes for MAC (Analytical)

### 2.1 Foundational Theory

**Sasoglu, Abbe, and Telatar (2013)** -- "Polar Codes for the Two-User Multiple-Access Channel"
- *Venue:* IEEE Trans. Inform. Theory, vol. 59, no. 10, pp. 6583-6592, October 2013 (arXiv:1006.4255)
- *Contribution:* Original proof that polar codes achieve the capacity region of two-user MACs with binary inputs. When two users independently apply the Arikan polar transform, the resulting synthetic channels polarize to one of five extremal types, on each of which uncoded transmission is optimal. Encoding and SC decoding both have O(N log N) complexity, and block error probability decays as o(exp(-N^{1/2-epsilon})).
- *Key concepts:* Monotone chain rule expansions, five extremal channel types (FF, FI, IF, II-XOR, II-independent), joint frozen set design.

**Sasoglu and Vardy (2013)** -- "Polar Codes for Arbitrary DMCs and Arbitrary MACs"
- *Venue:* arXiv:1311.3123, November 2013
- *Contribution:* Extended polar coding to arbitrary (non-binary) discrete memoryless channels and MACs with arbitrary number of users. Showed that polarization occurs for general alphabets, achieving any point in the MAC capacity region.

### 2.2 SC Decoding for MAC

**Onay (2013)** -- "Successive Cancellation Decoding of Polar Codes for the Two-User Binary-Input MAC"
- *Venue:* IEEE International Symposium on Information Theory (ISIT), 2013
- *Contribution:* Presented a successive cancellation decoder for polar codes on the two-user binary-input MAC that achieves the full admissible rate region. The decoder follows monotone chain rule expansions of mutual information terms.
- *Complexity:* O(N^2) due to operating on the full joint distribution space.
- *Relevance:* This is the "reference decoder" in our project, against which we benchmark.

**Ren, Bhatt, and Mondelli (2025)** -- "Successive Cancellation Decoding for General Monotone Chain Polar Codes"
- *Venue:* arXiv:2509.03128, September 2025
- *Contribution:* Presented a comprehensive decoding strategy for monotone chain polar codes that handles arbitrary numbers of terminals, non-binary alphabets, and decoding along arbitrary monotone chains. Formulated SC decoding as a series of inference subtasks over the polar transform, proposing a computational graph framework based on probability propagation principles.
- *Complexity:* O(N log N) for extremal paths (Class A/C), varying up to O(N^2) for general interleaved paths depending on the specific chain structure. Introduced a constant-time decoder forking strategy for list decoding.
- *Relevance:* This is the "interleaved decoder" we use as our analytical baseline. Our neural decoder aims to match or exceed its performance while potentially offering advantages on channels without closed-form transition probabilities.

### 2.3 Gaussian MAC Design and Decoding

**Marshakov, Balitskiy, Andreev, and Frolov (2019)** -- "Design and Decoding of Polar Codes for the Gaussian Multiple Access Channel"
- *Venue:* arXiv:1901.07297, January 2019
- *Contribution:* Proposed efficient design algorithms for polar codes on the GMAC, optimizing frozen bit selection. Compared joint successive cancellation and joint iterative decoding algorithms. Demonstrated that polar-coded NOMA outperforms LDPC-based solutions by approximately 1 dB and approaches the achievability bound.
- *Channel:* Gaussian MAC (AWGN with superposition)
- *Application:* Massive machine-type communications (mMTC) for 5G.

**Polar Codes with Dynamic Frozen Bits for GMAC (2022)**
- *Venue:* IEEE, 2022
- *Contribution:* Proposed dynamic frozen bits for polar codes on the MAC, with an algorithm that optimizes the search of dynamic frozen positions based on properties of joint SC decoding combined with list decoding. Experiments showed improved practical performance in multi-user scenarios.
- *Relevance:* Complementary to our work; we use standard frozen sets but replace the decoder operations with neural networks.

**Polar-Coded GMAC with Physical-Layer Network Coding (2024)**
- *Authors:* Published in IEEE Trans. Vehicular Technology, 2024
- *Contribution:* Proposed a two-user polar-coded physical-layer network coding (TU-PC-PNC) scheme for the GMAC. Derived concatenated codeword structure enabling XOR-based PNC via code construction. Introduced a joint optimization framework using Monte Carlo methods for polar code and power allocation optimization.
- *Key result:* TU-PC-PNC outperforms benchmark schemes by more than 1 dB on the GMAC.
- *Difference from our work:* Uses conventional polar decoders with PNC structure; our approach replaces the decoder's internal operations with neural networks.

### 2.4 Polar Codes for MAC -- Other Notable Work

**Abbe and Telatar (2010)** -- "Polar Codes for the m-User MAC"
- *Venue:* arXiv:1002.0777
- *Contribution:* Extended polar coding theory to m-user MACs, establishing connections with matroid theory.

**Polar Coding in Networks (2012)**
- *Authors:* Sasoglu
- *Venue:* IEEE Information Theory Workshop, 2012
- *Contribution:* Surveyed known results and open directions for polar codes in network settings including MACs, broadcast channels, and relay channels.

---

## 3. Neural Polar Decoders (Single User)

### 3.1 Foundational Work: Deep Learning for Channel Decoding

**Gruber et al. (2017)** -- "On Deep Learning-Based Channel Decoding"
- *Venue:* 51st Annual Conference on Information Sciences and Systems (CISS), IEEE, 2017
- *Contribution:* Revisited using deep neural networks for one-shot decoding of random and structured codes, including polar codes. Demonstrated that structured codes are easier to learn than random codes, and that neural networks can generalize to unseen codewords.
- *Architecture:* Fully connected DNN trained on received vectors.
- *Limitation:* Scalability -- training complexity grows exponentially with block length.

**Cammerer et al. (2017)** -- "Scaling Deep Learning-Based Decoding of Polar Codes via Partitioning"
- *Authors:* S. Cammerer, T. Gruber, J. Hoydis, S. ten Brink
- *Venue:* arXiv:1702.06901, 2017
- *Contribution:* Addressed the exponential scaling problem by partitioning the polar code encoding graph into smaller sub-blocks, training neural networks on each sub-block, and connecting them via conventional BP stages.
- *Channel:* AWGN
- *Key result:* NN-augmented BP decoder outperforms standard BP while maintaining manageable training complexity.

**O'Shea and Hoydis (2017)** -- "An Introduction to Deep Learning for the Physical Layer"
- *Venue:* IEEE Transactions on Cognitive Communications and Networking, vol. 3, pp. 563-575, 2017
- *Contribution:* Proposed interpreting an entire communications system as an autoencoder, jointly optimizing transmitter and receiver. Extended the concept to multiple transmitter/receiver networks.

### 3.2 Neural Belief Propagation Decoders

**Nachmani et al. (2018)** -- "Deep Learning Methods for Improved Decoding of Linear Codes"
- *Authors:* E. Nachmani, E. Marciano, L. Lugosch, W. Gross, D. Burshtein, Y. Be'ery
- *Venue:* IEEE J. Sel. Top. Signal Process., 2018
- *Contribution:* Showed that a standard BP decoder for linear codes can be improved by assigning learnable weights to the edges of the Tanner graph (weighted BP). Tying parameters across iterations forms an RNN architecture requiring significantly fewer parameters.

### 3.3 Neural Successive Cancellation (NSC) Decoders

**Doan and Bhatt (2018)** -- "Neural Successive Cancellation Decoding of Polar Codes"
- *Venue:* IEEE SPAWC, Kalamata, Greece, 2018
- *Contribution:* Proposed a neural successive cancellation (NSC) decoder that partitions the polar code into sub-blocks and replaces each sub-decoder with a small neural network connected via the SC framework. At the same error performance, NSC reduces decoding latency by up to 89% compared to SC, BP, and partitioned NN decoders. For Polar(128,64), NSC matches PNN while reducing latency by 42.5%.
- *Architecture:* Constituent NN decoders connected via SC structure.
- *Relevance to our work:* Among the earliest works to embed neural networks inside the SC tree structure, though it replaces entire sub-decoders rather than individual tensor operations (CalcLeft/CalcRight/CalcParent) as we do.

### 3.4 Neural Polar Decoder (NPD) -- Aharoni et al.

This line of work by Aharoni, Pfister, Permuter, and Huleihel is the most directly related to our approach, as both replace SC decoder operations with neural networks.

**Aharoni et al. (2023/2024)** -- "Data-Driven Neural Polar Codes for Unknown Channels With and Without Memory"
- *Authors:* Ziv Aharoni, Bashar Huleihel, Henry D. Pfister, Haim H. Permuter
- *Venue:* IEEE ISIT 2024 (arXiv:2309.03148, September 2023); extended version in IEEE Open Journal of Communications Society, 2024
- *Channel model:* Arbitrary memoryless channels and channels with memory (finite-state channels), treated as black boxes.
- *Architecture:* The Neural SC (NSC) decoder uses **three neural networks** that replace the core elements of the SC decoder:
  1. **Check-node NN** (replaces the f-function / CalcLeft operation)
  2. **Bit-node NN** (replaces the g-function / CalcRight operation)
  3. **Soft-decision NN** (replaces the hard/soft decision at leaves)
  A fourth **channel embedding NN** maps raw channel outputs into the SC decoder's input space, enabling the decoder to work without an explicit channel model.
- *Training:* Level-by-level parallel cross-entropy loss (fast_ce), training all tree levels simultaneously.
- *Key results:* Provides theoretical consistency guarantees. Complexity is O(AN log N) where A is a user-defined budget independent of channel memory -- a dramatic improvement over O(|S|^3 N log N) for SC trellis decoders on channels with memory of state-space size |S|.
- *Difference from our work:* The NPD operates on single-user channels and replaces SC operations with NNs trained per-channel. Our work applies a similar philosophy to the **two-user MAC**, where the tensor operations involve joint distributions over both users' bits, requiring fundamentally different NN architectures (4-element tensor inputs/outputs vs. scalar LLRs) and training procedures. Moreover, we operate on the MAC-specific SC tree structure (with its monotone chain path structure and five extremal types) rather than the standard binary SC tree.

**Aharoni and Pfister (2025)** -- "Neural Polar Decoders for Deletion Channels"
- *Venue:* arXiv:2507.12329, July 2025
- *Channel:* Deletion channels with constant deletion rate delta in {0.01, 0.1}
- *Contribution:* Extended NPD to deletion channels, reducing complexity from O(N^4) to O(AN log N). Only one of the four NNs needed modification. Enabled list decoding for deletion channels for the first time.

**Hirsch, Aharoni et al. (2025)** -- "A Study of Neural Polar Decoders for Communication"
- *Authors:* Rom Hirsch, Ziv Aharoni, Henry D. Pfister, Haim H. Permuter
- *Venue:* NeurIPS 2025 (arXiv:2510.03069)
- *Contribution:* Extended NPDs to practical 5G systems including OFDM and single-carrier systems. Supports variable code lengths via rate matching, higher-order modulations, and pilotless decoding that exploits channel memory. NPDs consistently outperformed the 5G polar decoder in BER, BLER, and throughput, with largest gains for low-rate, short-block configurations common in 5G control channels.

**Aharoni et al. (2025)** -- "Code Rate Optimization via Neural Polar Decoders"
- *Venue:* IEEE ISIT 2025 (arXiv:2506.15836)
- *Contribution:* Used NPDs to simultaneously optimize code rates and input distributions by estimating mutual information of effective channels. Integrated the Honda-Yamamoto scheme for non-uniform input distributions.

**Aharoni et al. (2025)** -- "Neural Polar Decoders for DNA Data Storage"
- *Venue:* arXiv:2506.17076, 2025
- *Contribution:* Applied NPDs to DNA storage channels with synchronization errors.

**Aharoni, Huleihel, Pfister, and Permuter (2026)** -- "Optimized Polar Codes via Mutual Information Maximization with Neural Polar Decoders"
- *Venue:* IEEE Transactions on Communications, 2026
- *Contribution:* Proposes maximizing the rate of reliable communication for polar codes on channels with memory by optimizing a neural polar decoder. Enables simultaneous optimization of code rate over the input distribution and practical coding scheme design within the polar code framework.

### 3.5 Curriculum-Based and Transformer-Based Neural Polar Decoders

**Hebbar et al. (2023)** -- "CRISP: Curriculum Based Sequential Neural Decoders for Polar Code Family"
- *Authors:* S. Ashwin Hebbar, Viraj Nadkarni, Ashok Vardhan Makkuva, Suma Bhat, Sewoong Oh, Pramod Viswanath
- *Venue:* ICML 2023
- *Contribution:* Designed a curriculum-based sequential neural decoder guided by information-theoretic insights. CRISP outperforms SC decoding and approaches near-optimal reliability for Polar(32,16) and Polar(64,22). Extended to PAC codes.
- *Architecture:* Sequential neural decoder trained with curriculum learning.
- *Channel:* AWGN

**Hebbar et al. (2024)** -- "DeepPolar: Inventing Nonlinear Large-Kernel Polar Codes via Deep Learning"
- *Venue:* ICML 2024
- *Contribution:* Generalized polar codes by expanding the 2x2 Arikan kernel to a larger l x l kernel parameterized by learnable MLPs. Both nonlinear kernels and matched decoders are learned jointly. Setting kernel size l = sqrt(N) yields best performance.
- *Difference from our work:* DeepPolar modifies the code construction itself (nonlinear kernels) rather than the decoder for an existing code.

**Ankireddy et al. (2024)** -- "Nested Construction of Polar Codes via Transformers"
- *Venue:* IEEE ISIT 2024
- *Contribution:* Used self-attention transformers to iteratively construct the polar code reliability sequence, exploiting the nested structure. Transformer-designed codes outperform both 5G-NR and Density Evolution approaches.
- *Relevance:* Applies deep learning to code design rather than decoding.

### 3.6 TransCoder and Other Recent Approaches

**TransCoder (2025)** -- "A Neural-Enhancement Framework for Channel Codes"
- *Venue:* arXiv:2511.22539, November 2025
- *Contribution:* Employs the transformer architecture to improve the reliability of existing error-correcting codes (LDPC, BCH, Polar, Turbo) as a code-adaptive neural module at either transmitter, receiver, or both. Demonstrates significant BLER improvements across various codes and channel conditions while maintaining complexity comparable to traditional decoders. Particularly effective for longer codes (N>64) and low code rates.
- *Difference from our work:* A general-purpose neural enhancement layer, not specific to polar SC structure or MAC channels.

**5G LDPC Linear Transformer (2025)**
- *Venue:* arXiv:2501.14102, January 2025
- *Contribution:* Novel fully differentiable linear-time complexity transformer decoder for 5G NR LDPC codes with O(n) complexity.

### 3.7 Other Neural Polar Decoding Approaches

**Ebada et al. (2019)** -- "Deep Learning-Based Polar Code Design"
- *Venue:* 57th Annual Allerton Conference, 2019
- *Contribution:* Represented information/frozen bit indices as a trainable binary vector for gradient-based frozen-set optimization.

**DL-Aided SCL Decoders (2020--2024):**
- Song et al. proposed DL-aided adaptive SCL (DL-ASCL) using an ANN predictor to select list size, achieving 56% complexity reduction.
- Wang et al. proposed DL-aided SC-Flip decoders using LSTM networks to identify error bits.
- Deep learning has been used for shifted-pruning in SCL decoding (IEEE, 2023).
- A dynamic adaptive SCL decoder exploiting check properties of frozen bits (IEEE, 2024).

**Deep Learning-Enabled Polar Code Decoders for 5G and Beyond (2024)**
- *Venue:* ScienceDirect, 2024
- *Contribution:* Survey of deep learning-enabled polar code decoders, emphasizing their role in achieving high reliability and low latency for 5G/6G.

### 3.8 Survey Papers

**Matsumine and Ochiai (2024)** -- "Recent Advances in Deep Learning for Channel Coding: A Survey"
- *Venue:* IEEE Open Journal of Communications Society, vol. 5, pp. 6443-6481, 2024 (arXiv:2406.19664)
- *Scope:* Comprehensive survey covering model-free and model-based deep learning for LDPC and polar codes. Covers DL-based code design, BP decoding, SC decoding, and end-to-end learned codes. Notably, the survey does not cover MAC or multi-user scenarios, highlighting the gap our work addresses.

---

## 4. Autoencoders for MAC

### 4.1 Foundational Autoencoder Approaches

**O'Shea and Hoydis (2017)** -- (See Section 3.1)
- Extended the autoencoder concept to networks of multiple transmitters and receivers, laying the foundation for autoencoder-based MAC design.

**Aoudia and Hoydis (2019)** -- "Model-Free Training of End-to-End Communication Systems"
- *Venue:* IEEE J. Sel. Areas Commun., vol. 37, no. 11, pp. 2503-2516, 2019
- *Contribution:* Proposed model-free training of autoencoder-based systems using policy gradient methods from reinforcement learning, eliminating the need for a differentiable channel model. Critical for MAC scenarios where channel models may be complex or unknown.

### 4.2 Autoencoder-Based MAC Constellation Design

**Gorelenkov and Vaezi (2025)** -- "Deep Autoencoder-Based Constellation Design in Multiple Access Channels"
- *Venue:* ISIT 2025 (arXiv:2505.00868, May 2025)
- *Channel:* K-user MAC with additive noise
- *Contribution:* Proposed DAE-based constellation design for MAC that creates interference-aware constellations, reducing SER and enhancing constellation-constrained sum capacity. DAE-designed constellations match or outperform analytically derived ones, and extend to >2 users where no analytical solutions exist.
- *Architecture:* Multiple encoder NNs (one per user) and a shared decoder NN, trained end-to-end.
- *Difference from our work:* Focuses on uncoded constellation design (modulation level) rather than channel coding. Does not use polar codes or exploit algebraic code structure.

### 4.3 End-to-End Learned Codes

**Kim et al. (2018)** -- "Deepcode: Feedback Codes via Deep Learning"
- *Venue:* NeurIPS 2018
- *Contribution:* Constructed the first family of DL-based codes that significantly outperform state-of-the-art for AWGN channels with feedback (3 orders of magnitude in reliability, 3 dB gain). Used RNN-based encoders and decoders.

**Jiang et al. (2019)** -- "Turbo Autoencoder: Deep Learning Based Channel Codes for Point-to-Point Communication Channels"
- *Venue:* NeurIPS 2019
- *Contribution:* Fully end-to-end jointly trained neural encoder and decoder (TurboAE) that approaches state-of-the-art under canonical channels and outperforms under non-canonical settings.
- *Architecture:* CNN-based encoder with interleaver structure; RNN/CNN-based decoder.

### 4.4 Joint Source-Channel Coding for MAC

**Bourtsoulatze et al. (2019)** -- "Deep Joint Source-Channel Coding for Wireless Image Transmission"
- *Venue:* IEEE Trans. Cognitive Commun. Networking, 2019
- *Contribution:* Proposed deep JSCC using CNN-based encoder/decoder, directly mapping pixel values to channel symbols. Outperforms JPEG/JPEG2000 + capacity-achieving channel code at low SNR.

**Tung and Gunduz (2022)** -- "Distributed Deep Joint Source-Channel Coding over a Multiple Access Channel"
- *Venue:* IEEE ICC 2023 (arXiv:2211.09920, November 2022)
- *Channel:* Noisy multiple access channel (AWGN MAC)
- *Contribution:* Introduced non-orthogonal joint source-channel coding for distributed image transmission over a MAC using DeepJSCC. Multiple devices send compressed image representations non-orthogonally. Motivated by the fact that Shannon's separation theorem is suboptimal in the finite block length regime for MAC.
- *Architecture:* CNN-based encoders (one per device) and a shared decoder.
- *Key result:* Non-orthogonal JSCC over MAC outperforms orthogonal schemes at practical block lengths.
- *Difference from our work:* Focuses on image transmission (source coding + channel coding jointly learned), not on channel coding alone. Does not use structured polar codes.
- *Code available:* https://github.com/ipc-lab/deepjscc-noma

**Distributed JSCC with Decoder-Only Side Information (2023)**
- *Venue:* arXiv:2310.04311, October 2023
- *Contribution:* Extended distributed JSCC to the Wyner-Ziv scenario where correlated side information is available only at the receiver.

**In-Context Learning for Deep JSCC over MIMO Channels (2025)**
- *Venue:* arXiv:2512.01567, December 2025
- *Contribution:* Applied in-context learning with attention modules for deep JSCC over MIMO channels at IEEE VTC 2024.

### 4.5 Autoencoder for Interference Channels

**End-to-End Autoencoder with Interference Suppression (2022):**
- *Venue:* arXiv:2201.01388
- *Contribution:* Designed autoencoder-based transceivers that jointly optimize coding and modulation for interference channels, learning to suppress multi-user interference.

### 4.6 Surveys on Autoencoders for Communication

**Zou and Yang (2021)** -- "Channel Autoencoder for Wireless Communication: State of the Art, Challenges, and Trends"
- *Venue:* IEEE, 2021
- *Scope:* State-of-the-art survey on channel autoencoders covering single-user and multi-user settings.

**Review on Deep Learning Autoencoder in Next-Generation Communication Systems (2024)**
- *Venue:* arXiv:2412.13843, December 2024
- *Scope:* Comprehensive review drawing on 120+ recent studies on autoencoders in communication systems, including end-to-end optimization of transmitters and receivers.

---

## 5. Our Position (How Our Work Fits In)

### 5.1 Literature Landscape Summary

The literature on neural approaches to channel coding and multi-user communications can be organized along two axes:

| | **Single-User** | **Multi-User (MAC/NOMA)** |
|---|---|---|
| **Neural decoder for specific code** | NPD (Aharoni et al.), NSC (Doan), CRISP, DL-aided SC/SCL/BP, TransCoder | **[Gap -- our work fills this]** |
| **End-to-end learned code** | DeepCode, TurboAE, DeepPolar | MAC autoencoders, DL-SCMA, DeepJSCC-NOMA |
| **Neural detection (no coding)** | Farsad & Goldsmith | DNN-SIC, MIMO-NOMA detection, NOMANet |

### 5.2 Key Differentiators of Our Approach

Our work -- neural SC decoding of polar codes for the two-user MAC -- is distinguished from all prior work in several fundamental ways:

1. **First neural decoder for structured MAC codes.** While Aharoni et al.'s NPD replaces SC operations with NNs for single-user channels, no prior work has applied this approach to the MAC, where the decoder must jointly decode two users' polar codewords by traversing monotone chain paths through a 2D grid of joint bit decisions.

2. **Tensor-level neural replacement, not black-box.** Unlike DNN-based NOMA detectors that treat the entire detection problem as a black-box, we preserve the SC tree structure and replace only the CalcLeft, CalcRight, and CalcParent tensor operations with MLPs. This maintains the O(N log N) complexity of the SC decoder, provides interpretability, and allows the decoder to generalize across different frozen-set configurations.

3. **Joint distributions over two users.** The NPD operates on scalar LLRs (or soft values for a single bit). Our tensor operations process 4-element tensors representing joint distributions p(u_A, u_B) over both users' bits. This requires fundamentally different NN input/output structures and training procedures.

4. **MAC-specific tree structure.** The single-user SC decoder traverses a binary tree of depth log N. The MAC SC decoder traverses a path through a 2D grid, interleaving decisions about User A and User B bits according to a monotone chain path. The CalcLeft/CalcRight operations combine tensors from different positions in this interleaved structure, and CalcParent must handle the five extremal types (FF, FI, IF, II-XOR, II-independent).

5. **Channel-model-free MAC decoding.** Like the NPD, our approach does not require an explicit channel model -- it learns the channel implicitly from training data. This is particularly valuable for MAC channels where computing exact transition probabilities may be intractable (e.g., Gaussian MAC with specific SNR and power constraints).

6. **Comparison to end-to-end approaches.** Unlike autoencoder-based MAC systems that learn both encoding and decoding from scratch, we leverage the proven capacity-achieving properties of polar codes for MACs (Sasoglu et al. 2013) and only replace the decoder's internal operations. This provides the benefits of structured codes (proven capacity achievement, known encoding complexity, rate flexibility) while gaining the adaptability of neural networks.

### 5.3 Closest Related Works

| Paper | Similarity | Key Difference |
|---|---|---|
| Aharoni et al. (2024) -- NPD | Same philosophy: replace SC operations with NNs | Single-user only; scalar LLRs not joint tensors |
| Hirsch et al. (2025) -- NPD for 5G | NPD in practical systems | Single-user OFDM/SC; no MAC structure |
| Doan (2018) -- NSC | Neural networks inside SC structure | Single-user; replaces sub-decoders not individual operations |
| CRISP (Hebbar 2023) | Sequential neural decoder for polar codes | Single-user; curriculum-based training differs from our MLP replacement |
| DeepPolar (Hebbar 2024) | Neural parameterization of polar code components | Modifies the code itself (nonlinear kernels), not the decoder for standard codes |
| DNN-NOMA (Gui 2018) | Neural network for multi-user channels | No channel coding structure; detection-level only |
| PC-NOMA + DL (2021) | Polar codes + NOMA + deep learning | Separates detection and decoding; uses conventional polar decoder |
| Ebert et al. (2020) | Polar codes on GMAC with iterative decoding | Uses conventional BP decoders; detection/decoding separated |
| MAC autoencoder (Gorelenkov 2025) | Autoencoder for MAC | Uncoded constellation design, not structured channel coding |
| DeepJSCC-NOMA (Tung 2022) | Neural coding over MAC | Source+channel coding jointly learned; no polar code structure |
| TransCoder (2025) | Neural enhancement for polar codes | General-purpose enhancement; single-user; not MAC-specific |
| Ren et al. (2025) | SC decoding for MAC polar codes | Analytical (non-neural) decoder; our baseline |

### 5.4 Open Problems Our Work Addresses

1. **Neural decoding for structured MAC codes** -- no prior work exists.
2. **Scalability of MAC decoders to channels without closed-form models** -- conventional MAC SC decoders require exact channel transition probabilities.
3. **Complexity reduction for MAC decoding** -- by learning compact NN representations of the tensor operations, we can potentially reduce the constant factors in O(N log N) decoding complexity.
4. **Robustness to channel model mismatch** -- neural decoders trained on real channel data can adapt to conditions not captured by idealized models.
5. **Neural SCL for MAC** -- we demonstrate that neural SC can be combined with list decoding (Neural SCL), beating analytical SCL(L=4) at short block lengths (N=32, 64).

### 5.5 Summary of Our Key Results

| Setting | N | SC | NN-SC | NN-SCL(L=4) | SCL(L=4) |
|---------|---|-----|-------|-------------|----------|
| GMAC 6dB Class B | 32 | 0.046 | 0.045 | **0.022** | 0.026 |
| GMAC 6dB Class B | 64 | 0.025 | 0.028 | **0.013** | 0.013 |
| BEMAC Class B | 64 | 0.006 | 0.003 | **0.0007** | -- |

Our Neural SCL(L=4) beats analytical SCL(L=4) at N<=64, demonstrating that neural tensor operations can capture information that the analytical decoder misses, particularly at short block lengths where the finite-length penalty is significant.

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
- [TransCoder, "A Neural-Enhancement Framework for Channel Codes," 2025](https://arxiv.org/abs/2511.22539)

### Neural Polar Decoder (NPD) -- Aharoni et al.
- [Aharoni et al., "Data-Driven Neural Polar Codes for Unknown Channels With and Without Memory," IEEE ISIT, 2024](https://arxiv.org/abs/2309.03148)
- [Hirsch, Aharoni et al., "A Study of Neural Polar Decoders for Communication," NeurIPS, 2025](https://arxiv.org/abs/2510.03069)
- [Aharoni et al., "Code Rate Optimization via Neural Polar Decoders," IEEE ISIT, 2025](https://arxiv.org/abs/2506.15836)
- [Aharoni and Pfister, "Neural Polar Decoders for Deletion Channels," 2025](https://arxiv.org/abs/2507.12329)
- [Aharoni et al., "Neural Polar Decoders for DNA Data Storage," 2025](https://arxiv.org/abs/2506.17076)
- [Aharoni et al., "Optimized Polar Codes via MI Maximization with NPDs," IEEE Trans. Commun., 2026](https://cris.bgu.ac.il/en/publications/optimized-polar-codes-via-mutual-information-maximization-with-ne/)

### Polar Codes for MAC (Analytical)
- [Sasoglu, Abbe, Telatar, "Polar Codes for the Two-User Multiple-Access Channel," IEEE TIT, 2013](https://arxiv.org/abs/1006.4255)
- [Sasoglu and Vardy, "Polar Codes for Arbitrary DMCs and Arbitrary MACs," 2013](https://arxiv.org/abs/1311.3123)
- [Onay, "Successive Cancellation Decoding of Polar Codes for the Two-User Binary-Input MAC," IEEE ISIT, 2013](https://ieeexplore.ieee.org/document/6620401)
- [Marshakov et al., "Design and Decoding of Polar Codes for the Gaussian Multiple Access Channel," 2019](https://arxiv.org/abs/1901.07297)
- [Ren, Bhatt, Mondelli, "Successive Cancellation Decoding for General Monotone Chain Polar Codes," 2025](https://arxiv.org/abs/2509.03128)
- [Polar Codes with Dynamic Frozen Bits for GMAC, IEEE, 2022](https://ieeexplore.ieee.org/document/10016939/)
- [Polar-Coded GMAC with Physical-Layer Network Coding, IEEE TVT, 2024](https://ieeexplore.ieee.org/document/10388470/)

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
- [NOMANet, "A GNN Enabled Power Allocation Scheme for NOMA," 2025](https://arxiv.org/abs/2502.05592)
- [DL-Based Decoding in NOMA for HAPS, Telecommunication Systems, 2025](https://link.springer.com/article/10.1007/s11235-025-01301-2)

### Joint Detection and Decoding
- [Xu et al., "Deep Learning for Joint MIMO Detection and Channel Decoding," IEEE PIMRC, 2019](https://arxiv.org/abs/1901.05647)
- [DL-Based Joint Detection and Decoding of NOMA, IEEE Globecom, 2019](https://ieeexplore.ieee.org/document/8644090/)
- [Ebert et al., "Iterative Detection and Decoding of Polar Codes in GMAC," IEEE ICC, 2021](https://arxiv.org/abs/2012.01075)

### Neural Detection and Channel Estimation
- [Farsad and Goldsmith, "Neural Network Detection of Data Sequences," IEEE TSP, 2018](https://arxiv.org/abs/1802.02046)
- [Ye, Li, Juang, "Power of Deep Learning for Channel Estimation and Signal Detection in OFDM," IEEE WCL, 2017](https://arxiv.org/abs/1708.08514)

### Joint Source-Channel Coding for MAC
- [Bourtsoulatze et al., "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE TCCN, 2019](https://arxiv.org/abs/1809.01733)
- [Tung and Gunduz, "Distributed Deep JSCC over a Multiple Access Channel," IEEE ICC, 2023](https://arxiv.org/abs/2211.09920)

### Surveys
- [Matsumine and Ochiai, "Recent Advances in Deep Learning for Channel Coding: A Survey," IEEE OJCOMS, 2024](https://arxiv.org/abs/2406.19664)
- [Zou and Yang, "Channel Autoencoder for Wireless Communication: State of the Art," IEEE, 2021](https://ieeexplore.ieee.org/document/9446711)
