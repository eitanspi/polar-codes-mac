"""
polar -- Polar codes for the two-user binary-input MAC.

Implements successive cancellation (SC) and SC list (SCL) decoding
of polar codes for the two-user binary-input multiple access channel,
reproducing results from Onay (ISIT 2013).

Modules:
    encoder             -- Polar encoder (bit-reversal + XOR butterfly)
    channels            -- BEMAC, ABNMAC, GaussianMAC channel models
    design              -- Analytical Bhattacharyya + GA polar code design
    design_mc           -- Monte Carlo genie-aided polar code design
    decoder             -- Unified SC MAC decoder (auto-dispatch LLR/tensor)
    decoder_scl         -- SC List (SCL) decoder for MAC polar codes
    decoder_interleaved -- O(N log N) SC decoder for all monotone chain paths
    efficient_decoder   -- O(N log N) SC decoder for extreme paths only
    eval                -- BER / BLER Monte Carlo evaluation pipeline
"""

from .encoder import polar_encode, build_message
from .channels import BEMAC, ABNMAC, GaussianMAC
from .design import design_bemac, make_path
from .design_mc import design_bemac_mc
from .decoder import decode_single, decode_batch
from .decoder_scl import decode_single_list, decode_batch_list
