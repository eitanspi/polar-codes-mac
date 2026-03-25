"""
neural — Neural SC decoder for 2-user MAC polar codes.

Modules:
    neural_comp_graph  — Baseline NCG decoder (27K params, O(N log N))
    ncg_pure_neural    — Pure Neural CalcParent decoder (38K params, O(md) per op, zero analytical ops)
    train_pure_neural  — Distillation training for Pure Neural CalcParent (3-phase curriculum)
    scale_pure_neural  — Curriculum scaling N=32/64/128 for Pure Neural decoder
    scale_large        — Memory-lean curriculum scaling N=256/512/1024
    ncg_gmac           — GMAC decoder with continuous z_encoder (MLP)
"""
