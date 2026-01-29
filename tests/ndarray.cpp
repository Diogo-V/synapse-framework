#include "ndarray.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

class NDArrayTests : public ::testing::Test {};
class ShapeBroadcastTests : public ::testing::Test {};

TEST(NDArrayTest, Build1DArray) {
  std::vector<float> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  synapse::Shape shape{6};
  synapse::Strides strides{1};
  synapse::NDArray arr(data, shape);

  EXPECT_EQ(arr.ndim(), 1);
  EXPECT_EQ(arr.size(), 6);
  EXPECT_EQ(arr.shape(), shape);
  EXPECT_EQ(arr.data(), data);
  EXPECT_EQ(arr.strides(), strides);
}

TEST(NDArrayTest, Build2DArray) {
  std::vector<float> data{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  synapse::Shape shape{2, 3};
  synapse::Strides strides{3, 1};
  synapse::NDArray arr(data, shape);

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr.size(), 6);
  EXPECT_EQ(arr.shape(), shape);
  EXPECT_EQ(arr.data(), data);
  EXPECT_EQ(arr.strides(), strides);
}

TEST(NDArrayTest, Build3DArray) {
  std::vector<float> data{1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                          7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  synapse::Shape shape{2, 2, 3};
  synapse::Strides strides{6, 3, 1};
  synapse::NDArray arr(data, shape);

  EXPECT_EQ(arr.ndim(), 3);
  EXPECT_EQ(arr.size(), 12);
  EXPECT_EQ(arr.shape(), shape);
  EXPECT_EQ(arr.data(), data);
  EXPECT_EQ(arr.strides(), strides);
}

TEST(NDArrayTest, StrideComputation) {
  std::vector<float> data(12, 0);
  synapse::Shape shape = {3, 2, 2};
  synapse::NDArray arr(data, shape);

  EXPECT_EQ(arr.is_contigous(), true);
}

TEST(NDArrayTest, IndexingOperator) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  synapse::Shape shape = {2, 3};
  synapse::NDArray arr(data, shape);

  EXPECT_EQ(arr(0, 0), 1);
  EXPECT_EQ(arr(0, 1), 2);
  EXPECT_EQ(arr(0, 2), 3);
  EXPECT_EQ(arr(1, 0), 4);
  EXPECT_EQ(arr(1, 1), 5);
  EXPECT_EQ(arr(1, 2), 6);
}

TEST(NDArrayTest, ToString) {
  // 1D Tensor Test
  synapse::NDArray arr1d({1.23F, 4.56F, 7.89F}, {3});
  EXPECT_EQ(arr1d.to_string(), "[1.230, 4.560, 7.890]");

  // 2D Tensor Test
  synapse::NDArray arr2d({1.23F, 4.56F, 7.89F, 2.34F, 5.67F, 8.90F}, {2, 3});
  EXPECT_EQ(arr2d.to_string(),
            "[[1.230, 4.560, 7.890],\n [2.340, 5.670, 8.900]]");

  // 3D Tensor Test
  synapse::NDArray arr3d(
      {1.23F, 4.56F, 7.89F, 2.34F, 5.67F, 8.90F, 9.01F, 3.21F, 6.54F},
      {3, 1, 3});
  EXPECT_EQ(arr3d.to_string(), "[[[1.230, 4.560, 7.890]],\n [[2.340, 5.670, "
                               "8.900]],\n [[9.010, 3.210, 6.540]]]");

  // 3D Tensor (More complex shape)
  synapse::NDArray arr3d_complex({1.23F, 4.56F, 7.89F, 2.34F, 5.67F, 8.90F,
                                  9.01F, 3.21F, 6.54F, 7.77F, 8.88F, 9.99F},
                                 {3, 2, 2});
  EXPECT_EQ(arr3d_complex.to_string(), "[[[1.230, 4.560],\n  [7.890, 2.340]],\n"
                                       " [[5.670, 8.900],\n  [9.010, 3.210]],\n"
                                       " [[6.540, 7.770],\n  [8.880, 9.990]]]");
}

TEST(NDArrayTest, OutOfBoundsAccessThrows) {
  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  synapse::Shape shape = {2, 3};
  synapse::NDArray arr(data, shape);

  EXPECT_THROW(arr(2, 0), std::out_of_range);
  EXPECT_THROW(arr(0, 3), std::out_of_range);
}

TEST(ShapeBroadcastTests, NdIndexToPos) {
  // 1D Array
  synapse::Shape indices_1d = {3};
  synapse::Strides strides_1d = {1};
  EXPECT_EQ(synapse::nd_index_to_pos(indices_1d, strides_1d), 3);

  // 2D Array
  synapse::Shape indices_2d = {1, 1};
  synapse::Strides strides_2d = {3, 1};
  EXPECT_EQ(synapse::nd_index_to_pos(indices_2d, strides_2d), 4);

  // 3D Array with standard strides
  synapse::Shape indices_3d = {1, 2, 1};
  synapse::Strides strides_3d = {6, 2, 1};
  EXPECT_EQ(synapse::nd_index_to_pos(indices_3d, strides_3d), 11);

  // 4D Array with complex strides
  synapse::Shape indices_4d = {1, 3, 2, 1};
  synapse::Strides strides_4d = {24, 8, 2, 1};
  EXPECT_EQ(synapse::nd_index_to_pos(indices_4d, strides_4d), 53);

  // Edge case: all indices are 0
  synapse::Shape indices_zeros = {0, 0};
  synapse::Strides strides_zeros = {3, 1};
  EXPECT_EQ(synapse::nd_index_to_pos(indices_zeros, strides_zeros), 0);

  // Edge case: single element in a multi-dimensional array
  synapse::Shape indices_single = {0, 0};
  synapse::Strides strides_single = {1, 1};
  EXPECT_EQ(synapse::nd_index_to_pos(indices_single, strides_single), 0);
}

TEST(ShapeBroadcastTests, PosToNdIndex) {
  // 1D Array
  synapse::Shape shape_1d = {5};
  EXPECT_EQ(synapse::pos_to_nd_index(0, shape_1d), (synapse::Shape{0}));
  EXPECT_EQ(synapse::pos_to_nd_index(2, shape_1d), (synapse::Shape{2}));
  EXPECT_EQ(synapse::pos_to_nd_index(4, shape_1d), (synapse::Shape{4}));

  // 2D Array
  synapse::Shape shape_2d = {3, 4};
  EXPECT_EQ(synapse::pos_to_nd_index(0, shape_2d), (synapse::Shape{0, 0}));
  EXPECT_EQ(synapse::pos_to_nd_index(3, shape_2d), (synapse::Shape{0, 3}));
  EXPECT_EQ(synapse::pos_to_nd_index(4, shape_2d), (synapse::Shape{1, 0}));
  EXPECT_EQ(synapse::pos_to_nd_index(11, shape_2d), (synapse::Shape{2, 3}));

  // 3D Array
  synapse::Shape shape_3d = {2, 3, 2};
  EXPECT_EQ(synapse::pos_to_nd_index(0, shape_3d), (synapse::Shape{0, 0, 0}));
  EXPECT_EQ(synapse::pos_to_nd_index(7, shape_3d), (synapse::Shape{1, 0, 1}));
  EXPECT_EQ(synapse::pos_to_nd_index(11, shape_3d), (synapse::Shape{1, 2, 1}));

  // 4D Array
  synapse::Shape shape_4d = {2, 2, 3, 2};
  EXPECT_EQ(synapse::pos_to_nd_index(0, shape_4d),
            (synapse::Shape{0, 0, 0, 0}));
  EXPECT_EQ(synapse::pos_to_nd_index(1, shape_4d),
            (synapse::Shape{0, 0, 0, 1}));
  EXPECT_EQ(synapse::pos_to_nd_index(5, shape_4d),
            (synapse::Shape{0, 0, 2, 1}));
  EXPECT_EQ(synapse::pos_to_nd_index(11, shape_4d),
            (synapse::Shape{0, 1, 2, 1}));
  EXPECT_EQ(synapse::pos_to_nd_index(17, shape_4d),
            (synapse::Shape{1, 0, 2, 1}));
  EXPECT_EQ(synapse::pos_to_nd_index(23, shape_4d),
            (synapse::Shape{1, 1, 2, 1}));

  // Edge cases
  EXPECT_EQ(synapse::pos_to_nd_index(0, {1}), (synapse::Shape{0}));
  EXPECT_EQ(synapse::pos_to_nd_index(0, {1, 1, 1, 1}),
            (synapse::Shape{0, 0, 0, 0}));
}

TEST(ShapeBroadcastTests, CompatibleShapes) {
  EXPECT_EQ(synapse::shape_broadcast({3, 4}, {1, 4}), (synapse::Shape{3, 4}));
  EXPECT_EQ(synapse::shape_broadcast({1, 4}, {3, 4}), (synapse::Shape{3, 4}));
  EXPECT_EQ(synapse::shape_broadcast({3, 1}, {3, 4}), (synapse::Shape{3, 4}));
  EXPECT_EQ(synapse::shape_broadcast({1, 1}, {3, 4}), (synapse::Shape{3, 4}));
  EXPECT_EQ(synapse::shape_broadcast({5, 1, 4}, {1, 3, 4}),
            (synapse::Shape{5, 3, 4}));
}

TEST(ShapeBroadcastTests, IncompatibleShapes) {
  EXPECT_THROW(synapse::shape_broadcast({3, 4}, {2, 4}), std::invalid_argument);
  EXPECT_THROW(synapse::shape_broadcast({3, 4}, {3, 5}), std::invalid_argument);
  EXPECT_THROW(synapse::shape_broadcast({3, 3, 4}, {2, 3, 1}),
               std::invalid_argument);
}

TEST(ShapeBroadcastTests, ScalarBroadcasting) {
  EXPECT_EQ(synapse::shape_broadcast({}, {3, 4}), (synapse::Shape{3, 4}));
  EXPECT_EQ(synapse::shape_broadcast({3, 4}, {}), (synapse::Shape{3, 4}));
  EXPECT_EQ(synapse::shape_broadcast({}, {}), (synapse::Shape{}));
}

TEST(ShapeBroadcastTests, HigherDimensionalBroadcasting) {
  EXPECT_EQ(synapse::shape_broadcast({8, 1, 6, 1}, {7, 1, 5}),
            (synapse::Shape{8, 7, 6, 5}));
  EXPECT_EQ(synapse::shape_broadcast({1, 2, 1}, {3, 1, 4}),
            (synapse::Shape{3, 2, 4}));
  EXPECT_EQ(synapse::shape_broadcast({3, 1, 2}, {2, 1}),
            (synapse::Shape{3, 2, 2}));
}
