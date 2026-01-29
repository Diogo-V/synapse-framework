#include "tensor.h"
#include "ndarray.h"
#include <string>
#include <utility>
#include <vector>

synapse::Tensor::Tensor(std::vector<float> data, synapse::Shape shape)
    : synapse::NDArray(std::move(data), std::move(shape)) {
  // Move the data into the parent class instead of copying
}

synapse::Tensor::~Tensor() = default; // Does not need to deallocate anything

auto synapse::Tensor::to_string() const -> std::string {
  std::string out;
  out += synapse::NDArray::to_string();
  return out;
}
