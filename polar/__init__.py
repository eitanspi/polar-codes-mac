"""
polar — Polar codes for the two-user binary-input MAC.

Implements successive cancellation (SC) and SC list (SCL) decoding
of polar codes for the two-user binary-input multiple access channel,
reproducing results from Önay (ISIT 2013).
"""

from .encoder import polar_encode, build_message
from .channels import BEMAC, ABNMAC
from .design import design_bemac, make_path
from .design_mc import design_bemac_mc
from .decoder import decode_single, decode_batch
from .decoder_scl import decode_single_list, decode_batch_list
