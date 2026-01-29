#include "func.h"
#include "ndarray.h"
#include "tensor.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

// Test fixture for the Calculator class
class FunctionalTests : public ::testing::Test {
protected:
  synapse::Tensor tensor_1{std::vector<float>{0.2F, 0.5F, 46.0F, -5.1F},
                           synapse::Shape{4}};
  synapse::Tensor tensor_2{std::vector<float>{0.2F, 0.5F, 46.0F, -5.1F},
                           synapse::Shape{4}};
};

TEST_F(FunctionalTests, AddTensors) {
  synapse::Tensor result = synapse::add(tensor_1, tensor_2);
  synapse::Tensor expected_tensor{std::vector<float>{0.4F, 1.0F, 92.0F, -10.2F},
                                  synapse::Shape{4}};
  EXPECT_TRUE(synapse::is_close(result, expected_tensor));
}

TEST_F(FunctionalTests, MulTensors) {
  synapse::Tensor result = synapse::mul(tensor_1, tensor_2);
  synapse::Tensor expected_tensor{
      std::vector<float>{0.04F, 0.25F, 2116.0F, 26.01F}, synapse::Shape{4}};
  EXPECT_TRUE(synapse::is_close(result, expected_tensor));
}

TEST_F(FunctionalTests, AddTensorsMismatchShape) {
  synapse::Tensor tensor_3{std::vector<float>{1.0F, 2.0F}, synapse::Shape{2}};
  EXPECT_THROW(synapse::add(tensor_1, tensor_3), std::invalid_argument);
}

TEST_F(FunctionalTests, MulTensorsMismatchShape) {
  synapse::Tensor tensor_3{std::vector<float>{1.0F, 2.0F}, synapse::Shape{2}};
  EXPECT_THROW(synapse::mul(tensor_1, tensor_3), std::invalid_argument);
}
