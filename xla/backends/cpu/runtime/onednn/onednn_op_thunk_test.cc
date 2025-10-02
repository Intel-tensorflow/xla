/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/cpu/runtime/onednn/onednn_op_thunk.h"

#include <vector>

// #include "gtest/gtest.h"
#include "xla/array2d.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(OneDnnOpThunkTest, SimpleOneDnnMatMulThunk) {
  // Set up a thread pool for parallel execution
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Define shapes for lhs (2x3), rhs (3x2), and output (2x2)
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2});
  Shape out_shape = ShapeUtil::MakeShape(F32, {2, 2});

  // Prepare dummy data (row-major):
  // A = [ [1, 2, 3],
  //       [4, 5, 6] ]
  // B = [ [7,  8],
  //       [9, 10],
  //       [11,12] ]
  // C = A * B =
  //     [ [58, 64],
  //       [139,154] ]
  std::vector<float> input_a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> input_b = {7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
  std::vector<float> output(4, 0.f);

  // Create Literals from data
  Literal lhs_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
  Literal rhs_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}}));
  Literal out_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{0.f, 0.f}, {0.f, 0.f}}));

  // Create buffer allocations and slices
  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs_literal, rhs_literal, out_literal);

  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);

  BufferAllocations allocations =
      CreateBufferAllocations(lhs_literal, rhs_literal, out_literal);

  // Set up op_buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {lhs_slice, rhs_slice};
  op_buffers.arguments_shapes = {lhs_shape, rhs_shape};
  op_buffers.results_buffers = {out_slice};
  op_buffers.results_shapes = {out_shape};

  // Create thunk (matmul)
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      OneDnnOpThunk::Create("__onednn$matmul", Thunk::Info(), op_buffers, {}));

  // Set up execute params
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  // Execute the thunk
  auto exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError()) << "OneDnnOpThunk execution failed";

  // Expected output literal
  Literal expected = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{58.f, 64.f}, {139.f, 154.f}}));

  // Load output and verify
  EXPECT_EQ(out_literal, expected);
}

TEST(OneDnnOpThunkTest, SimpleOneDnnLayerNormThunk) {
  // Set up a thread pool for parallel execution
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Shapes: input [2,3], gamma [3], beta [3], output [2,3]
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape gamma_shape = ShapeUtil::MakeShape(F32, {3});
  Shape beta_shape = ShapeUtil::MakeShape(F32, {3});
  Shape out_shape = ShapeUtil::MakeShape(F32, {2, 3});

  // Input:
  // [[1,2,3],
  //  [4,5,6]]
  // gamma = [1,1,1]
  // beta  = [0,0,0]
  // Expected (per-row LN): [[-1.2247449, 0, 1.2247449],
  //                         [-1.2247449, 0, 1.2247449]]
  Literal input_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
  Literal gamma_literal = LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f});
  Literal beta_literal = LiteralUtil::CreateR1<float>({0.f, 0.f, 0.f});
  Literal out_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}}));

  // Buffer allocations
  auto [input_alloc, gamma_alloc, beta_alloc, out_alloc] =
      CreateBufferAllocation(input_literal, gamma_literal, beta_literal,
                             out_literal);

  auto [input_slice, gamma_slice, beta_slice, out_slice] =
      CreateBufferAllocationSlice(input_alloc, gamma_alloc, beta_alloc,
                                  out_alloc);

  BufferAllocations allocations = CreateBufferAllocations(
      input_literal, gamma_literal, beta_literal, out_literal);

  // Op buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {input_slice, gamma_slice, beta_slice};
  op_buffers.arguments_shapes = {input_shape, gamma_shape, beta_shape};
  op_buffers.results_buffers = {out_slice};
  op_buffers.results_shapes = {out_shape};

  // oneDNN LayerNorm config
  OneDnnNormConfig ln_cfg;
  ln_cfg.set_rescale(OneDnnNormConfig::SCALE_AND_SHIFT);
  float epsilon = 1e-5f;
  int32_t epsilon_bits = *reinterpret_cast<int32_t*>(&epsilon);
  ln_cfg.set_epsilon_typecast(epsilon_bits);

  OneDnnOpThunk::OneDnnOpConfig config = ln_cfg;

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$layernorm", Thunk::Info(),
                                        op_buffers, config));

  // Execute
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  tsl::AsyncValueRef<Thunk::ExecuteEvent> exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError()) << "oneDNN LayerNorm thunk failed";

  // Expected output
  const float ref = 1.2247449f;  // sqrt(3/2)
  Literal expected = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{-ref, 0.f, ref}, {-ref, 0.f, ref}}));

  // Compare with tolerance
  const float* exp_view = expected.data<float>().data();
  float* out_view = out_literal.data<float>().data();
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(out_view[i], exp_view[i], 1e-4)
        << "Mismatch at index " << i << ": got " << out_view[i] << " expected "
        << exp_view[i];
  }
}

}  // namespace
}  // namespace xla::cpu
