#include "ndarray.h"
#include <algorithm>
#include <cstddef>
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

synapse::NDArray::NDArray(std::vector<float> data, synapse::Shape shape)
    : _data(std::move(data)), _shape(std::move(shape)),
      _ndim(this->_shape.size()), _size(this->_data.size()) {

  // Computes strides where stride_i = shape_i * stride_i-1
  this->_strides = synapse::Strides(this->_ndim, 0);
  this->_strides[this->_ndim - 1] = 1;
  for (size_t i = this->_ndim - 1; i > 0; --i) {
    this->_strides[i - 1] = this->_shape[i] * this->_strides[i];
  }
}

synapse::NDArray::~NDArray() = default; // No action required

auto synapse::NDArray::shape() const -> const synapse::Shape & {
  return this->_shape;
}

auto synapse::NDArray::strides() const -> const synapse::Strides & {
  return this->_strides;
}

auto synapse::NDArray::data() -> std::vector<float> & { return this->_data; }
auto synapse::NDArray::data() const -> const std::vector<float> & {
  return this->_data;
}
auto synapse::NDArray::ndim() const -> size_t { return this->_ndim; }
auto synapse::NDArray::size() const -> size_t { return this->_size; }

auto synapse::NDArray::is_contigous() -> bool {
  if (this->_ndim == 1) {
    return true;
  }

  for (size_t i = this->_ndim - 1; i > 0; --i) {
    if (this->_strides[i - 1] != this->_shape[i] * this->_strides[i]) {
      return false;
    }
  }
  return true;
}

auto synapse::NDArray::to_string() const -> std::string {
  if (this->size() == 0) {
    return "[]";
  }

  std::ostringstream oss;

  // Recursive helper function.
  // 'offset' is the starting index into _data.
  // 'current_dim' indicates the dimension we are printing (0 is outermost).
  // 'indent' is the string of spaces to prepend when starting a new line at
  // this level.
  std::function<void(size_t, size_t, const std::string &)> rec;
  rec = [this, &oss, &rec](size_t offset, size_t current_dim,
                           const std::string &indent) -> void {
    oss << "[";
    // If we are at the last dimension, simply print the numbers.
    if (current_dim == this->ndim() - 1) {
      for (size_t i = 0; i < this->shape()[current_dim]; i++) {
        if (i > 0) {
          oss << ", ";
        }
        oss << std::fixed << std::setprecision(3) << this->data()[offset + i];
      }
    } else {
      // Compute the product of the remaining dimensions.
      size_t subarray_size = 1;
      for (size_t j = current_dim + 1; j < this->ndim(); j++) {
        subarray_size *= this->shape()[j];
      }
      // Loop over the current dimension.
      for (size_t i = 0; i < this->shape()[current_dim]; i++) {
        if (i > 0) {
          oss << ",\n" << indent << " ";
        }
        // Recursively print the subarray.
        rec(offset + (i * subarray_size), current_dim + 1, indent + " ");
      }
    }
    oss << "]";
  };

  rec(0, 0, "");
  return oss.str();
}

// Converts an N dimensional index into a position in the vector of data
auto synapse::nd_index_to_pos(const synapse::Shape &indices,
                              const synapse::Strides &strides) -> size_t {
  if (indices.size() != strides.size()) {
    throw std::invalid_argument(
        std::format("Number of dimensions in indices and strides do not match. "
                    "Found indices size {} and strides size {}.",
                    indices.size(), strides.size()));
  }
  size_t pos{0};
  for (size_t i = 0; i < indices.size(); i++) {
    pos += indices[i] * strides[i];
  }
  return pos;
}

// Converts a position in a vector of data into an N dimensional index
auto synapse::pos_to_nd_index(size_t pos, const synapse::Shape &shape)
    -> synapse::Shape {
  synapse::Shape out(shape.size());
  for (size_t i = shape.size(); i-- > 0;) {
    out[i] = pos % shape[i];
    pos /= shape[i];
  }
  return out;
}

auto synapse::shape_broadcast(const synapse::Shape &shape_1,
                              const synapse::Shape &shape_2) -> synapse::Shape {
  size_t len1 = shape_1.size();
  size_t len2 = shape_2.size();
  size_t out_size = std::max(len1, len2);
  synapse::Shape out(out_size);

  auto it_s1 = shape_1.crbegin();
  auto it_s2 = shape_2.crbegin();
  auto it_out = out.rbegin();

  // Starts from the end and figures the out shape between the two by following
  // the broadcasting rules:
  // 1. Two shapes different than 1 have to be equal
  // 2. If a shapes does not exist, is set to 1
  // 3. 1 can be broadcasted to another higher size
  // 4. 1's can be added to the left side of shapes to allow matching
  for (size_t i = 0; i < out_size; ++i) {
    size_t sz1 = (i < len1) ? *it_s1 : 1;
    size_t sz2 = (i < len2) ? *it_s2 : 1;

    if (sz1 != 1 && sz2 != 1 && sz1 != sz2) {
      throw std::invalid_argument("Tensor shapes cannot be broadcasted.");
    }

    *it_out = std::max(sz1, sz2);

    if (i < len1) {
      ++it_s1;
    }
    if (i < len2) {
      ++it_s2;
    }
    ++it_out;
  }

  return out;
}
