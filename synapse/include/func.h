#ifndef FUNC_H
#define FUNC_H

#include "tensor.h"

namespace synapse {
auto add(const Tensor &tensor_1, const Tensor &tensor_2) -> Tensor;
auto mul(const Tensor &tensor_1, const Tensor &tensor_2) -> Tensor;
auto matmul(const Tensor &tensor_1, const Tensor &tensor_2) -> Tensor;

auto is_close(const Tensor &tensor_1, const Tensor &tensor_2, float tol = 1e-5F)
    -> bool;
} // namespace synapse

#endif // !FUNC
