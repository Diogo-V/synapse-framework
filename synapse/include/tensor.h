#ifndef SYNAPSE_TENSOR_H
#define SYNAPSE_TENSOR_H

#include "ndarray.h"
#include <string>
#include <vector>

/**
 * @brief Tensor container
 */
namespace synapse {
class Tensor : public NDArray {
public:
  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) = default;
  auto operator=(const Tensor &) -> Tensor & = default;
  auto operator=(Tensor &&) -> Tensor & = default;
  Tensor(std::vector<float> data, synapse::Shape shape);
  ~Tensor();

  [[nodiscard]] auto to_string() const -> std::string;
};
} // namespace synapse

#endif // !SYNAPSE_TENSOR_H
