#pragma once
#include <torch/extension.h>
#include <torch/torch.h>

#include "common.h"
#include "subgraph_index.h"

using torch::Tensor;
std::vector<Tensor> neighbor_sample(const std::vector<Index>& ptr,
                                    const std::vector<Index>& idx,
                                    const std::vector<int>& fanouts,
                                    const std::vector<Index>& seed_nodes);