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

TEST_F(ConvolutionTest, DequantizeConv2D) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.f32, entry_computation_layout={(s8[1,3,224,224]{3,2,1,0}, s8[64,3,7,7]{3,2,1,0})->f32[1,112,112,64]{3,2,1,0}}

  ENTRY convolution.test.f32 {
    Arg_inp = s8[1,3,224,224]{3,2,1,0} parameter(0)
    convert.194 = s32[1,3,224,224]{3,2,1,0} convert(Arg_inp)
    constant.65 = s32[] constant(-4)
    broadcast.1 = s32[1,3,224,224]{3,2,1,0} broadcast(constant.65), dimensions={}
    add = s32[1,3,224,224]{3,2,1,0} add(convert.194, broadcast.1)
    convert.196 = f32[1,3,224,224]{3,2,1,0} convert(add)
    constant.48 = f32[] constant(0.5)
    broadcast.186 = f32[1,3,224,224]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.197 = f32[1,3,224,224]{3,2,1,0} multiply(convert.196, broadcast.186)
    transpose = f32[1,224,224,3]{3,2,1,0} transpose(multiply.197), dimensions={0,2,3,1}
    Arg_9.10 = s8[64,3,7,7]{3,2,1,0} parameter(1)
    convert.205 = s32[64,3,7,7]{3,2,1,0} convert(Arg_9.10)
    constant.66 = s32[] constant(0)
    broadcast.3 = s32[64,3,7,7]{3,2,1,0} broadcast(constant.66), dimensions={}
    add.1 = s32[64,3,7,7]{3,2,1,0} add(convert.205, broadcast.3)
    convert.207 = f32[64,3,7,7]{3,2,1,0} convert(add.1)
    broadcast.163 = f32[64,3,7,7]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.208 = f32[64,3,7,7]{3,2,1,0} multiply(convert.207, broadcast.163)
    transpose.1 = f32[7,7,3,64]{3,2,1,0} transpose(multiply.208), dimensions={2,3,1,0}
    ROOT convolution = f32[1,112,112,64]{3,2,1,0} convolution(transpose, transpose.1), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  // TODO(intel-tf): Check that the fusion has the expected quantized type.
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_str_);
}

TEST_F(ConvolutionTest, DequantizeConv2DBiasRelu) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.f32, entry_computation_layout={(s8[1,3,224,224]{3,2,1,0}, s8[64,3,7,7]{3,2,1,0}, f32[64]{0})->f32[1,112,112,64]{3,2,1,0}}

  ENTRY convolution.test.f32 {
    Arg_inp = s8[1,3,224,224]{3,2,1,0} parameter(0)
    convert.194 = s32[1,3,224,224]{3,2,1,0} convert(Arg_inp)
    constant.65 = s32[] constant(-4)
    broadcast.1 = s32[1,3,224,224]{3,2,1,0} broadcast(constant.65), dimensions={}
    add = s32[1,3,224,224]{3,2,1,0} add(convert.194, broadcast.1)
    convert.196 = f32[1,3,224,224]{3,2,1,0} convert(add)
    constant.48 = f32[] constant(0.5)
    broadcast.186 = f32[1,3,224,224]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.197 = f32[1,3,224,224]{3,2,1,0} multiply(convert.196, broadcast.186)
    transpose = f32[1,224,224,3]{3,2,1,0} transpose(multiply.197), dimensions={0,2,3,1}
    Arg_9.10 = s8[64,3,7,7]{3,2,1,0} parameter(1)
    convert.205 = s32[64,3,7,7]{3,2,1,0} convert(Arg_9.10)
    constant.66 = s32[] constant(0)
    broadcast.3 = s32[64,3,7,7]{3,2,1,0} broadcast(constant.66), dimensions={}
    add.1 = s32[64,3,7,7]{3,2,1,0} add(convert.205, broadcast.3)
    convert.207 = f32[64,3,7,7]{3,2,1,0} convert(add.1)
    broadcast.163 = f32[64,3,7,7]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.208 = f32[64,3,7,7]{3,2,1,0} multiply(convert.207, broadcast.163)
    transpose.1 = f32[7,7,3,64]{3,2,1,0} transpose(multiply.208), dimensions={2,3,1,0}
    convolution = f32[1,112,112,64]{3,2,1,0} convolution(transpose, transpose.1), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
    Arg_8.9 = f32[64]{0} parameter(2)
    broadcast.171 = f32[1,112,112,64]{2,1,3,0} broadcast(Arg_8.9), dimensions={3}
    add.57 = f32[1,112,112,64]{3,2,1,0} add(convolution, broadcast.171)
    constant.171 = f32[] constant(0)
    broadcast.205 = f32[1,112,112,64]{2,1,3,0} broadcast(constant.171), dimensions={}
    ROOT maximum.0 = f32[1,112,112,64]{3,2,1,0} maximum(add.57, broadcast.205)
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  // TODO(intel-tf): Check that the fusion has the expected quantized type.
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_relu_str_);
}

TEST_F(ConvolutionTest, DequantizeConv2DBiasReluRequantize) {
  const char* convolution_module_str = R"(
  HloModule convolution.test.f32, entry_computation_layout={(s8[1,64,56,56]{3,2,1,0}, s8[64,64,3,3]{3,2,1,0}, f32[64]{0}, s8[64,64,3,3]{3,2,1,0})->f32[1,56,56,64]{3,2,1,0}}

  ENTRY convolution.test.f32 {
    Arg_inp = s8[1,64,56,56]{3,2,1,0} parameter(0)
    convert.194 = s32[1,64,56,56]{3,2,1,0} convert(Arg_inp)
    constant.65 = s32[] constant(-4)
    broadcast.1 = s32[1,64,56,56]{3,2,1,0} broadcast(constant.65), dimensions={}
    add = s32[1,64,56,56]{3,2,1,0} add(convert.194, broadcast.1)
    convert.196 = f32[1,64,56,56]{3,2,1,0} convert(add)
    constant.48 = f32[] constant(0.5)
    broadcast.186 = f32[1,64,56,56]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.197 = f32[1,64,56,56]{3,2,1,0} multiply(convert.196, broadcast.186)
    transpose = f32[1,56,56,64]{3,2,1,0} transpose(multiply.197), dimensions={0,2,3,1}
    Arg_9.10 = s8[64,64,3,3]{3,2,1,0} parameter(1)
    convert.205 = s32[64,64,3,3]{3,2,1,0} convert(Arg_9.10)
    constant.66 = s32[] constant(0)
    broadcast.3 = s32[64,64,3,3]{3,2,1,0} broadcast(constant.66), dimensions={}
    add.1 = s32[64,64,3,3]{3,2,1,0} add(convert.205, broadcast.3)
    convert.207 = f32[64,64,3,3]{3,2,1,0} convert(add.1)
    broadcast.163 = f32[64,64,3,3]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.208 = f32[64,64,3,3]{3,2,1,0} multiply(convert.207, broadcast.163)
    transpose.1 = f32[3,3,64,64]{3,2,1,0} transpose(multiply.208), dimensions={2,3,1,0}
    convolution = f32[1,56,56,64]{3,2,1,0} convolution(transpose, transpose.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
    Arg_8.9 = f32[64]{0} parameter(2)
    broadcast.171 = f32[1,56,56,64]{2,1,3,0} broadcast(Arg_8.9), dimensions={3}
    add.57 = f32[1,56,56,64]{3,2,1,0} add(convolution, broadcast.171)
    constant.171 = f32[] constant(0)
    broadcast.205 = f32[1,56,56,64]{2,1,3,0} broadcast(constant.171), dimensions={}
    maximum.0 = f32[1,56,56,64]{3,2,1,0} maximum(add.57, broadcast.205)
    constant = f32[] constant(2)
    broadcast.266 = f32[1,56,56,64]{3,2,1,0} broadcast(constant), dimensions={}
    multiply.76 = f32[1,56,56,64]{3,2,1,0} multiply(maximum.0, broadcast.266)
    constant.46 = f32[] constant(4)
    broadcast.336 = f32[1,56,56,64]{3,2,1,0} broadcast(constant.46), dimensions={}
    add.93 = f32[1,56,56,64]{3,2,1,0} add(multiply.76, broadcast.336)
    constant.184 = f32[] constant(127)
    broadcast.413 = f32[1,56,56,64]{3,2,1,0} broadcast(constant.184), dimensions={}
    constant.183 = f32[] constant(-128)
    broadcast.328 = f32[1,56,56,64]{3,2,1,0} broadcast(constant.183), dimensions={}
    clamp.15 = f32[1,56,56,64]{3,2,1,0} clamp(broadcast.328, add.93, broadcast.413)
    round-nearest-even.15 = f32[1,56,56,64]{3,2,1,0} round-nearest-even(clamp.15)
    convert.17 = s8[1,56,56,64]{3,2,1,0} convert(round-nearest-even.15)
    Arg_12.13 = s8[64,64,3,3]{3,2,1,0} parameter(3)
    convert.38 = s32[1,56,56,64]{3,2,1,0} convert(convert.17)
    broadcast.468 = s32[1,56,56,64]{2,1,3,0} broadcast(constant.65), dimensions={}
    add.116 = s32[1,56,56,64]{3,2,1,0} add(convert.38, broadcast.468)
    convert.61 = f32[1,56,56,64]{3,2,1,0} convert(add.116)
    broadcast.515 = f32[1,56,56,64]{2,1,3,0} broadcast(constant.48), dimensions={}
    multiply.92 = f32[1,56,56,64]{3,2,1,0} multiply(convert.61, broadcast.515)
    convert.274 = s32[64,64,3,3]{3,2,1,0} convert(Arg_12.13)
    constant.67 = s32[] constant(0)
    broadcast.9 = s32[64,64,3,3]{3,2,1,0} broadcast(constant.67), dimensions={}
    add.6 = s32[64,64,3,3]{3,2,1,0} add(convert.274, broadcast.9)
    convert.276 = f32[64,64,3,3]{3,2,1,0} convert(add.6)
    broadcast.145 = f32[64,64,3,3]{3,2,1,0} broadcast(constant.48), dimensions={}
    multiply.277 = f32[64,64,3,3]{3,2,1,0} multiply(convert.276, broadcast.145)
    transpose.7 = f32[3,3,64,64]{3,2,1,0} transpose(multiply.277), dimensions={2,3,1,0}
    ROOT convolution.2 = f32[1,56,56,64]{3,2,1,0} convolution(multiply.92, transpose.7), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  })";

  EXPECT_TRUE(RunAndCompare(convolution_module_str, ErrorSpec{1e-4, 1e-4}));
  // TODO(intel-tf): Check that the fusion has the expected quantized type.
  MatchOptimizedHlo(convolution_module_str, conv_rewrite_bias_relu_str_);
}



}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
