#include "func.h"
#include "ndarray.h"
#include "tensor.h"
#include <cmath>
#include <cstddef>
#include <format>
#include <stdexcept>
#include <vector>

auto synapse::add(const synapse::Tensor &tensor_1,
                  const synapse::Tensor &tensor_2) -> synapse::Tensor {
  if (tensor_1.size() != tensor_2.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  Tensor tensor_3{std::vector<float>(tensor_1.size()), tensor_1.shape()};
  for (size_t i = 0; i < tensor_1.size(); i++) {
    tensor_3.data()[i] = tensor_1.data()[i] + tensor_2.data()[i];
  }
  return tensor_3;
}

auto synapse::mul(const synapse::Tensor &tensor_1,
                  const synapse::Tensor &tensor_2) -> synapse::Tensor {
  if (tensor_1.size() != tensor_2.size()) {
    throw std::invalid_argument("Tensor sizes do not match");
  }

  Tensor tensor_3{std::vector<float>(tensor_1.size()), tensor_1.shape()};
  for (size_t i = 0; i < tensor_1.size(); i++) {
    tensor_3.data()[i] = tensor_1.data()[i] * tensor_2.data()[i];
  }
  return tensor_3;
}

auto synapse::matmul(const synapse::Tensor &tensor_1,
                     const synapse::Tensor &tensor_2) -> synapse::Tensor {
  if (tensor_1.shape()[tensor_1.shape().size() - 1] != tensor_2.shape()[0]) {
    throw std::invalid_argument(std::format(
        "Matrix multiplication is invalid. Found tensor shapes {} and {}.",
        tensor_1.shape(), tensor_2.shape()));
  }

  synapse::Shape out_shape =
      synapse::shape_broadcast(tensor_1.shape(), tensor_2.shape());

  // TODO(diogo): I need to implement shape broadcast

  return tensor_1;
}

auto synapse::is_close(const synapse::Tensor &tensor_1,
                       const synapse::Tensor &tensor_2, float tol) -> bool {
  if (tensor_1.shape() != tensor_2.shape()) {
    return false;
  }
  for (size_t i = 0; i < tensor_1.size(); ++i) {
    if (std::fabs(tensor_1.data()[i] - tensor_2.data()[i]) > tol) {
      return false;
    }
  }
  return true;
}
