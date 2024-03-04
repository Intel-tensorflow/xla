/* Copyright 2023 The OpenXLA Authors.

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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <utility>

#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/cpu_info.h"

namespace xla {
namespace cpu {

class ConvolutionTest : public HloTestBase {
  protected:
    const char* conv_rewrite_str_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";

    const char* conv_rewrite_bias_relu_str_ = R"(
    ; CHECK:     custom_call_target="__onednn$convolution",
    ; CHECK:       backend_config={
    ; CHECK-DAG:     "outer_dimension_partitions":[],
    ; CHECK-DAG:       "onednn_conv_config":{
    ; CHECK-DAG:       "fusions":{
    ; CHECK-DAG:         "ops":["BIAS","RELU"]
    ; CHECK-DAG:     }
    ; CHECK-DAG:   }
    ; CHECK:     }
    )";
};

TEST_F(ConvolutionTest, Simple2DTestF32) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.f32, entry_computation_layout={(f32[1,22,22,1]{3,2,1,0}, f32[8,8,1,1]{3,2,1,0})->f32[1,11,11,1]{3,2,1,0}}

  ENTRY convolution.test.f32 {
    arg.0 = f32[1,22,22,1]{3,2,1,0} parameter(0), parameter_replication={false}
    reshape.0 = f32[1,22,22,1]{3,2,1,0} reshape(arg.0)
    arg.1 = f32[8,8,1,1]{3,2,1,0} parameter(1), parameter_replication={false}
    reshape.1 = f32[8,8,1,1]{3,2,1,0} reshape(arg.1)
    convolution.0 = f32[1,11,11,1]{3,2,1,0} convolution(reshape.0, reshape.1), window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    reshape.2 = f32[1,11,11,1]{3,2,1,0} reshape(convolution.0)
    tuple.0 = (f32[1,11,11,1]{3,2,1,0}) tuple(reshape.2)
    ROOT get-tuple-element.0 = f32[1,11,11,1]{3,2,1,0} get-tuple-element(tuple.0), index=0
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, Simple3DTestBF16) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.bf16, entry_computation_layout={(bf16[8,4,5,5,1]{4,3,2,1,0}, bf16[3,3,3,1,32]{4,3,2,1,0})->bf16[8,4,5,5,32]{4,3,2,1,0}}

  ENTRY convolution.test.bf16 {
    p0 = bf16[8,4,5,5,1]{4,3,2,1,0} parameter(0)
    p1 = bf16[3,3,3,1,32]{4,3,2,1,0} parameter(1)
    ROOT conv = bf16[8,4,5,5,32]{4,3,2,1,0} convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
})";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, TestFusedConv2D) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.with.bias.relu.f32.2D, entry_computation_layout={(f32[8,5,5,1]{3,2,1,0}, f32[3,3,1,32]{3,2,1,0}, f32[32]{0})->f32[8,5,5,32]{3,2,1,0}}

  ENTRY TestComputation {
    p0 = f32[8,5,5,1] parameter(0)
    p1 = f32[3,3,1,32] parameter(1)
    conv = f32[8,5,5,32] convolution(p0, p1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
    bias = f32[32] parameter(2)
    broadcasted_bias = f32[8,5,5,32] broadcast(bias), dimensions={3}
    add = f32[8,5,5,32] add(conv, broadcasted_bias)
    zero = f32[] constant(0)
    zeros = f32[8,5,5,32] broadcast(zero), dimensions={}
    ROOT relu = f32[8,5,5,32] maximum(zeros, add)
})";
  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{0.01, 0.01}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_relu_str_);
}

TEST_F(ConvolutionTest, TestFusedConv3D) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.with.bias.relu.f32.3D, entry_computation_layout={(f32[8,4,5,5,1]{4,3,2,1,0}, f32[3,3,3,1,32]{4,3,2,1,0}, f32[32]{0})->f32[8,4,5,5,32]{4,3,2,1,0}}

  ENTRY TestComputation {
    p0 = f32[8,4,5,5,1] parameter(0)
    p1 = f32[3,3,3,1,32] parameter(1)
    conv = f32[8,4,5,5,32] convolution(p0, p1), window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = f32[32] parameter(2)
    broadcasted_bias = f32[8,4,5,5,32] broadcast(bias), dimensions={4}
    add = f32[8,4,5,5,32] add(conv, broadcasted_bias)
    zero = f32[] constant(0)
    zeros = f32[8,4,5,5,32] broadcast(zero), dimensions={}
    ROOT relu = f32[8,4,5,5,32] maximum(zeros, add)
})";
  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{0.01, 0.01}));
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_relu_str_);
}


}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
