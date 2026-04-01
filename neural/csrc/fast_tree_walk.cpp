/*
 * fast_tree_walk.cpp — C++ PyTorch extension for fast sequential tree walk.
 *
 * Eliminates Python interpreter overhead by running the entire tree walk
 * (1500+ MLP calls) in C++. The MLPs are evaluated using raw tensor ops
 * (mm, addmm, elu) without Python dispatch overhead.
 *
 * The schedule (list of operations) is pre-computed in Python and passed in.
 * MLP weights are passed as tensor parameters.
 */
#include <torch/extension.h>
#include <vector>
#include <cmath>

using namespace torch;

// Apply a 2-layer MLP: Linear(in, h) -> ELU -> Linear(h, h) -> ELU -> Linear(h, out)
// weights: [w0, b0, w1, b1, w2, b2]
static Tensor apply_mlp(
    const Tensor& input,
    const Tensor& w0, const Tensor& b0,
    const Tensor& w1, const Tensor& b1,
    const Tensor& w2, const Tensor& b2
) {
    auto h = torch::elu(torch::addmm(b0, input, w0.t()));
    h = torch::elu(torch::addmm(b1, h, w1.t()));
    return torch::addmm(b2, h, w2.t());
}

// Gated residual CalcParent: gate * candidate + (1-gate) * (left+right)/2
static Tensor apply_calc_parent(
    const Tensor& left,   // (B, L, d)
    const Tensor& right,  // (B, L, d)
    const Tensor& gate_w0, const Tensor& gate_b0,
    const Tensor& gate_w1, const Tensor& gate_b1,
    const Tensor& cand_w0, const Tensor& cand_b0,
    const Tensor& cand_w1, const Tensor& cand_b1,
    const Tensor& cand_w2, const Tensor& cand_b2
) {
    int B = left.size(0);
    int L = left.size(1);
    int d = left.size(2);

    auto left_flat = left.reshape({B * L, d});
    auto right_flat = right.reshape({B * L, d});
    auto concat = torch::cat({left_flat, right_flat}, 1);  // (B*L, 2d)

    // Gate
    auto gate_h = torch::elu(torch::addmm(gate_b0, concat, gate_w0.t()));
    auto gate = torch::sigmoid(torch::addmm(gate_b1, gate_h, gate_w1.t()));

    // Candidate
    auto cand = apply_mlp(concat, cand_w0, cand_b0, cand_w1, cand_b1, cand_w2, cand_b2);

    // Residual
    auto residual = (left_flat + right_flat) / 2.0;
    auto result = gate * cand + (1.0 - gate) * residual;

    return result.reshape({B, L, d});
}

/*
 * tree_walk_forward: Execute the pre-computed schedule in C++.
 *
 * Args:
 *   root_emb: (B, N, d) root embeddings
 *   schedule_ops: vector of int — flattened schedule [op_type, arg1, arg2, ...]
 *   no_info_emb: (d,) no-info embedding
 *   calc_left_weights: [w0, b0, w1, b1, w2, b2]
 *   calc_right_weights: [w0, b0, w1, b1, w2, b2]
 *   calc_parent_gate_weights: [w0, b0, w1, b1]
 *   calc_parent_cand_weights: [w0, b0, w1, b1, w2, b2]
 *   parent_second_weights: [w, b]
 *   emb2logits_weights: [w0, b0, w1, b1, w2, b2]
 *   logits2emb_weights: [w0, b0, w1, b1, w2, b2]
 *   u_true: (B, N) or empty
 *   v_true: (B, N) or empty
 *   frozen_u_mask: (N,) bool — true if frozen
 *   frozen_v_mask: (N,) bool — true if frozen
 *
 * Returns: (all_logits, all_targets) as stacked tensors
 */
std::vector<Tensor> tree_walk_forward(
    Tensor root_emb,
    std::vector<int64_t> schedule_flat,
    Tensor no_info_emb,
    std::vector<Tensor> calc_left_w,
    std::vector<Tensor> calc_right_w,
    std::vector<Tensor> parent_gate_w,
    std::vector<Tensor> parent_cand_w,
    std::vector<Tensor> parent_second_w,
    std::vector<Tensor> emb2logits_w,
    std::vector<Tensor> logits2emb_w,
    Tensor u_true,
    Tensor v_true,
    Tensor frozen_u_mask,
    Tensor frozen_v_mask
) {
    int B = root_emb.size(0);
    int N = root_emb.size(1);
    int d = root_emb.size(2);
    int n = 0;
    for (int tmp = N; tmp > 1; tmp >>= 1) n++;

    bool has_teacher = u_true.numel() > 0;
    float LOG_HALF = std::log(0.5f);
    float LOG_QUARTER = std::log(0.25f);

    // Initialize edge data
    std::vector<Tensor> edge_data(2 * N);
    edge_data[1] = root_emb;

    auto no_info_exp = no_info_emb.unsqueeze(0).unsqueeze(0);  // (1, 1, d)
    for (int beta = 2; beta < 2 * N; beta++) {
        int level = 0;
        for (int tmp = beta; tmp > 1; tmp >>= 1) level++;
        int size = N >> level;
        edge_data[beta] = no_info_exp.expand({B, size, d}).clone();
    }

    // Track decisions
    std::vector<Tensor> u_hat(N + 1);
    std::vector<Tensor> v_hat(N + 1);
    std::vector<bool> u_hat_set(N + 1, false);
    std::vector<bool> v_hat_set(N + 1, false);

    std::vector<Tensor> all_logits;
    std::vector<Tensor> all_targets;

    // Parse and execute schedule
    int si = 0;
    int n_ops = schedule_flat.size();

    while (si < n_ops) {
        int op_type = schedule_flat[si++];

        if (op_type == 0) {  // CalcLeft at vertex beta
            int beta = schedule_flat[si++];
            auto parent = edge_data[beta];
            auto right = edge_data[2 * beta + 1];
            int l = right.size(1);
            auto p_first = parent.slice(1, 0, l);
            auto p_second = parent.slice(1, l);

            auto p_first_flat = p_first.reshape({B * l, d});
            auto p_second_flat = p_second.reshape({B * l, d});
            auto right_flat = right.reshape({B * l, d});
            auto inp = torch::cat({p_first_flat, p_second_flat, right_flat}, 1);

            auto out = apply_mlp(inp, calc_left_w[0], calc_left_w[1],
                                 calc_left_w[2], calc_left_w[3],
                                 calc_left_w[4], calc_left_w[5]);
            edge_data[2 * beta] = out.reshape({B, l, d});

        } else if (op_type == 1) {  // CalcRight at vertex beta
            int beta = schedule_flat[si++];
            auto parent = edge_data[beta];
            auto left = edge_data[2 * beta];
            int l = left.size(1);
            auto p_first = parent.slice(1, 0, l);
            auto p_second = parent.slice(1, l);

            auto p_first_flat = p_first.reshape({B * l, d});
            auto p_second_flat = p_second.reshape({B * l, d});
            auto left_flat = left.reshape({B * l, d});
            auto inp = torch::cat({p_first_flat, p_second_flat, left_flat}, 1);

            auto out = apply_mlp(inp, calc_right_w[0], calc_right_w[1],
                                 calc_right_w[2], calc_right_w[3],
                                 calc_right_w[4], calc_right_w[5]);
            edge_data[2 * beta + 1] = out.reshape({B, l, d});

        } else if (op_type == 2) {  // CalcParent at vertex beta
            int beta = schedule_flat[si++];
            auto left = edge_data[2 * beta];
            auto right = edge_data[2 * beta + 1];

            auto pf = apply_calc_parent(left, right,
                parent_gate_w[0], parent_gate_w[1], parent_gate_w[2], parent_gate_w[3],
                parent_cand_w[0], parent_cand_w[1], parent_cand_w[2], parent_cand_w[3],
                parent_cand_w[4], parent_cand_w[5]);

            // parent_second
            int L = right.size(1);
            auto right_flat = right.reshape({B * L, d});
            auto ps = torch::addmm(parent_second_w[1], right_flat, parent_second_w[0].t());
            ps = ps.reshape({B, L, d});

            edge_data[beta] = torch::cat({pf, ps}, 1);

        } else if (op_type == 3 || op_type == 4) {  // Leaf (left=3, right=4)
            int vtx = schedule_flat[si++];
            int leaf = schedule_flat[si++];
            int i_t = schedule_flat[si++];
            int gamma = schedule_flat[si++];
            int is_frozen = schedule_flat[si++];

            auto temp = edge_data[leaf].select(1, 0).clone();  // (B, d)

            // CalcLeft or CalcRight at leaf level
            if (op_type == 3) {  // leaf_left
                auto parent = edge_data[vtx];
                auto right = edge_data[2 * vtx + 1];
                int l = right.size(1);
                auto pf = parent.slice(1, 0, l).reshape({B * l, d});
                auto ps = parent.slice(1, l).reshape({B * l, d});
                auto rf = right.reshape({B * l, d});
                auto inp = torch::cat({pf, ps, rf}, 1);
                auto out = apply_mlp(inp, calc_left_w[0], calc_left_w[1],
                                     calc_left_w[2], calc_left_w[3],
                                     calc_left_w[4], calc_left_w[5]);
                edge_data[2 * vtx] = out.reshape({B, l, d});
            } else {
                auto parent = edge_data[vtx];
                auto left = edge_data[2 * vtx];
                int l = left.size(1);
                auto pf = parent.slice(1, 0, l).reshape({B * l, d});
                auto ps = parent.slice(1, l).reshape({B * l, d});
                auto lf = left.reshape({B * l, d});
                auto inp = torch::cat({pf, ps, lf}, 1);
                auto out = apply_mlp(inp, calc_right_w[0], calc_right_w[1],
                                     calc_right_w[2], calc_right_w[3],
                                     calc_right_w[4], calc_right_w[5]);
                edge_data[2 * vtx + 1] = out.reshape({B, l, d});
            }

            auto top_down = edge_data[leaf].select(1, 0);  // (B, d)
            auto combined = top_down + temp;
            auto logits = apply_mlp(combined, emb2logits_w[0], emb2logits_w[1],
                                    emb2logits_w[2], emb2logits_w[3],
                                    emb2logits_w[4], emb2logits_w[5]);

            Tensor bit;
            if (is_frozen) {
                bit = torch::zeros({B});
            } else {
                all_logits.push_back(logits);
                if (has_teacher) {
                    auto target = (u_true.select(1, i_t - 1) * 2 + v_true.select(1, i_t - 1)).to(torch::kLong);
                    all_targets.push_back(target);
                    bit = (gamma == 0) ? u_true.select(1, i_t - 1) : v_true.select(1, i_t - 1);
                } else {
                    auto p0 = (gamma == 0) ?
                        torch::logsumexp(logits.slice(1, 0, 2), 1) :
                        torch::logsumexp(logits.index({torch::indexing::Slice(), torch::tensor({0, 2})}), 1);
                    auto p1 = (gamma == 0) ?
                        torch::logsumexp(logits.slice(1, 2, 4), 1) :
                        torch::logsumexp(logits.index({torch::indexing::Slice(), torch::tensor({1, 3})}), 1);
                    bit = (p1 > p0).to(torch::kFloat);
                }
            }

            if (gamma == 0) { u_hat[i_t] = bit; u_hat_set[i_t] = true; }
            else { v_hat[i_t] = bit; v_hat_set[i_t] = true; }

            // Set leaf embedding
            auto lp = torch::full({B, 4}, -30.0f);
            bool has_u = u_hat_set[i_t], has_v = v_hat_set[i_t];
            if (has_u && has_v) {
                auto idx = (u_hat[i_t].to(torch::kLong) * 2 + v_hat[i_t].to(torch::kLong)).unsqueeze(1);
                lp.scatter_(1, idx, 0.0f);
            } else if (has_u) {
                lp.scatter_(1, (u_hat[i_t].to(torch::kLong) * 2).unsqueeze(1), LOG_HALF);
                lp.scatter_(1, (u_hat[i_t].to(torch::kLong) * 2 + 1).unsqueeze(1), LOG_HALF);
            } else if (has_v) {
                lp.scatter_(1, v_hat[i_t].to(torch::kLong).unsqueeze(1), LOG_HALF);
                lp.scatter_(1, (v_hat[i_t].to(torch::kLong) + 2).unsqueeze(1), LOG_HALF);
            } else {
                lp.fill_(LOG_QUARTER);
            }

            auto leaf_emb = apply_mlp(lp, logits2emb_w[0], logits2emb_w[1],
                                      logits2emb_w[2], logits2emb_w[3],
                                      logits2emb_w[4], logits2emb_w[5]);
            edge_data[leaf] = leaf_emb.unsqueeze(1);
        }
    }

    if (all_logits.empty()) {
        return {torch::zeros({0, 4}), torch::zeros({0}).to(torch::kLong)};
    }
    return {torch::stack(all_logits), torch::stack(all_targets)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tree_walk_forward", &tree_walk_forward, "Fast tree walk forward pass");
}
