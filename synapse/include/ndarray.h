#ifndef NDARRAY_H
#define NDARRAY_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace synapse {

/**
 * @brief Represents the dimensions of an N-dimensional array.
 *
 * @details Each element `Shape[i]` corresponds to the extent of the array along
 * axis `i`. The length of the vector defines the rank of the tensor.
 */
using Shape = std::vector<size_t>;

/**
 * @brief Represents the memory offset increments required to traverse
 * dimensions.
 *
 * @details For a given dimension `i`, `Strides[i]` is the number of elements to
 * skip in the underlying linear storage to move to the next logical element
 * along that axis.
 */
using Strides = std::vector<size_t>;

/**
 * @brief Maps multi-dimensional coordinates to a linear memory offset.
 *
 * @param indices The multi-dimensional coordinates (e.g., {y, x}).
 * @param strides The strides of the tensor.
 * @return The 0-indexed position in the flat data vector.
 * @throws std::invalid_argument if indices and strides have mismatched ranks.
 *
 * @note This implements the formula: $\text{pos} = \sum_{i=0}^{n-1}
 * (\text{indices}_i \times \text{strides}_i)$
 */
auto nd_index_to_pos(const Shape &indices, const Strides &strides) -> size_t;

/**
 * @brief Inverse of nd_index_to_pos; reconstructs coordinates from a flat
 * index.
 *
 * @param pos The 0-indexed linear position in memory.
 * @param shape The shape of the tensor.
 * @return A Shape containing the N-dimensional indices for that position.
 *
 * @details Useful for iterating over a flat buffer while needing to know
 * the logical coordinates (e.g., for kernel operations or debugging).
 */
auto pos_to_nd_index(size_t pos, const Shape &shape) -> Shape;

/**
 * @brief Determines the resulting shape when performing element-wise operations
 * on two tensors.
 *
 * @param shape_1 Shape of the first tensor operand.
 * @param shape_2 Shape of the second tensor operand.
 * @return The broadcasted shape.
 * @throws std::invalid_argument if shapes are incompatible.
 *
 * @details Implements NumPy-style broadcasting:
 * 1. Dimensions are compared starting from the trailing (rightmost) side.
 * 2. Two dimensions are compatible if they are equal, or if one of them is 1.
 * 3. The resulting size is the maximum of the two dimensions.
 */
auto shape_broadcast(const Shape &shape_1, const Shape &shape_2) -> Shape;

/**
 * @brief An N-dimensional array (Tensor) container.
 * * NDArray manages a flat block of memory and uses a `Shape` and `Strides`
 * to provide a multi-dimensional view of that data.
 *
 * ### Example
 * ```cpp
 * synapse::NDArray arr({1, 2, 3, 4}, {2, 2});
 * float val = arr(0, 1); // Accesses first row, second column
 * ```
 */
class NDArray {
public:
  NDArray(const NDArray &) = default;
  NDArray(NDArray &&) = default;
  auto operator=(const NDArray &) -> NDArray & = default;
  auto operator=(NDArray &&) -> NDArray & = default;
  NDArray(std::vector<float> data, Shape shape);
  ~NDArray();

  // Accessors
  [[nodiscard]] auto shape() const -> const Shape &;
  [[nodiscard]] auto strides() const -> const Strides &;
  auto data() -> std::vector<float> &;
  [[nodiscard]] auto data() const -> const std::vector<float> &;
  [[nodiscard]] auto ndim() const -> size_t;
  [[nodiscard]] auto size() const -> size_t;

  // Methods
  auto is_contigous() -> bool;
  [[nodiscard]] auto to_string() const -> std::string;

  // Allows accessing elements of the ndarray directly
  template <typename... Indices>
  auto operator()(Indices... indices) const -> const float & {
    return this->data()[_operator_parenthesis(indices...)];
  }
  template <typename... Indices>
  auto operator()(Indices... indices) -> float & {
    return this->data()[_operator_parenthesis(indices...)];
  }

private:
  std::vector<float> _data;
  Shape _shape;
  Strides _strides;
  size_t _ndim;
  size_t _size;

  template <typename... Indices>
  auto _operator_parenthesis(Indices... indices) const -> size_t {
    static_assert(sizeof...(indices) > 0, "At least one index is required.");
    Shape idx_vec{static_cast<size_t>(indices)...};

    // Bounds checking
    if (idx_vec.size() != this->ndim()) {
      throw std::out_of_range(
          "Number of indices does not match the number of dimensions.");
    }
    for (size_t i = 0; i < idx_vec.size(); i++) {
      if (idx_vec[i] >= this->shape()[i]) {
        throw std::out_of_range("Index out of bounds for dimension " +
                                std::to_string(i));
      }
    }

    return synapse::nd_index_to_pos(idx_vec, this->strides());
  }
};
} // namespace synapse

#endif // !NDARRAY_H
