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

#include "xla/service/cpu/onednn_convolution_rewriter.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;

template <typename Pattern>
auto ElementwiseSafeIntermediate(HloInstruction** instr, Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Broadcast(instr, pattern.WithOneUser()),
                                  m::Slice(instr, pattern.WithOneUser()),
                                  m::Bitcast(instr, pattern.WithOneUser()),
                                  m::Reshape(instr, pattern.WithOneUser()),
                                  pattern);
}

inline auto OneDnnConvolutionInstr(HloInstruction** instr) {
  return m::CustomCall(instr, {"__onednn$convolution"});
}

inline bool CompatibleElementType(const HloInstruction* instr) {
  PrimitiveType element_type = instr->shape().element_type();
  return element_type == BF16 || element_type == F32;
}

inline auto BcastConstScalar(HloInstruction** instr, double value) {
  return m::Broadcast(instr, m::ConstantScalar(value));
}

inline auto BcastConstScalar(double value) {
  return BcastConstScalar(nullptr, value);
}

inline bool IsRowMajor(const Shape& shape) {
  return LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

template <typename Pattern>
inline auto SupportedConvert(Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(pattern).WithPredicate(supported_convert);
}

template <typename Pattern>
inline auto SupportedConvert(HloInstruction** convert, Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return CompatibleElementType(instr) &&
           CompatibleElementType(instr->operand(0));
  };
  return m::Convert(convert, pattern).WithPredicate(supported_convert);
}

template <typename Pattern>
inline auto BitcastWithReshapeSemantics(HloInstruction** bitcast,
                                        Pattern pattern) {
  auto is_reshape = [](const HloInstruction* instr) -> bool {
    if (!instr) return false;
    auto input_shape = instr->operand(0)->shape();
    auto output_shape = instr->shape();
    bool is_same_type = ShapeUtil::SameElementType(input_shape, output_shape);
    bool has_equal_num_elems = ShapeUtil::ElementsIn(input_shape) ==
                               ShapeUtil::ElementsIn(output_shape);
    bool has_rowmajor_layout =
        IsRowMajor(input_shape) && IsRowMajor(output_shape);
    return is_same_type && has_equal_num_elems && has_rowmajor_layout;
  };
  return m::Bitcast(bitcast, pattern).WithPredicate(is_reshape);
}

template <typename Pattern>
inline auto OptionalConvertAndBitcast(HloInstruction** optional_convert,
                                      HloInstruction** optional_bitcast,
                                      Pattern pattern) {
  auto common =
      m::AnyOf<HloInstruction>(
          SupportedConvert(optional_convert, std::move(pattern).WithOneUser())
              .WithOperand(0, m::Op().WithElementType(PrimitiveType::BF16))
              .WithElementType(PrimitiveType::F32),
          std::move(pattern).WithOneUser())
          .WithOneUser();
  return m::AnyOf<HloInstruction>(
      BitcastWithReshapeSemantics(optional_bitcast, common), common);
}

inline bool IsOperandFusible(HloInstruction* operand, HloInstruction* conv) {
  auto operand_dims = operand->shape().dimensions();
  auto conv_dims = conv->shape().dimensions();
  if (operand_dims.size() > conv_dims.size()) return false;
  int operand_idx = operand_dims.size() - 1;
  int conv_idx = conv_dims.size() - 1;
  for (; operand_idx >= 0; --operand_idx, --conv_idx) {
    if (operand_dims[operand_idx] != 1 &&
        operand_dims[operand_idx] != conv_dims[conv_idx])
      return false;
  }
  return true;
}

}  // namespace

bool OneDnnConvolutionRewriter::ShouldRewrite(const HloInstruction* conv) {
  if (conv->HasControlDependencies()) return false;
  if (!IsSupportedType(conv->shape().element_type())) return false;
  if (conv->batch_group_count() != 1) return false;

  if (conv->operand(1)->opcode() == HloOpcode::kReverse) return false;

  const Shape& inp_shape = conv->operand(0)->shape();
  const Shape& ker_shape = conv->operand(1)->shape();
  const Shape& out_shape = conv->shape();
  if (ShapeUtil::IsZeroElementArray(inp_shape) ||
      ShapeUtil::IsZeroElementArray(ker_shape) ||
      ShapeUtil::IsZeroElementArray(out_shape)) {
    return false;
  }

  auto dims = conv->window().dimensions().size();
  if (dims >= 4 || dims <= 0) return false;

  if (inp_shape.rank() != ker_shape.rank() ||
      inp_shape.rank() != out_shape.rank()) {
    return false;
  }

  return true;
}

class OneDnnConvolutionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleConvolution(HloInstruction* conv) override {
    auto pattern = match::Op(&conv).WithOpcode(HloOpcode::kConvolution);
    if (!Match(conv, pattern)) return OkStatus();
    if (!OneDnnConvolutionRewriter::ShouldRewrite(conv)) return OkStatus();

    const Shape& conv_shape = conv->shape();
    auto dims = conv->window().dimensions().size();
    const ConvolutionDimensionNumbers& conv_ddata =
        conv->convolution_dimension_numbers();

    BackendConfig backend_config;
    OneDnnConvolutionConfig* conv_config =
        backend_config.mutable_onednn_conv_config();

    conv_config->set_dims(conv_shape.rank());
    conv_config->set_feature_groups(conv->feature_group_count());
    conv_config->mutable_inp()->set_ibdim(conv_ddata.input_batch_dimension());
    conv_config->mutable_ker()->set_ibdim(
        conv_ddata.kernel_input_feature_dimension());
    conv_config->mutable_out()->set_ibdim(conv_ddata.output_batch_dimension());
    conv_config->mutable_inp()->set_ofdim(conv_ddata.input_feature_dimension());
    conv_config->mutable_ker()->set_ofdim(
        conv_ddata.kernel_output_feature_dimension());
    conv_config->mutable_out()->set_ofdim(
        conv_ddata.output_feature_dimension());

    const Shape& output_shape = conv->shape();

    for (auto rit = conv->window().dimensions().begin();
         rit != conv->window().dimensions().end(); rit++) {
      if ((*rit).padding_low() < 0 || (*rit).padding_high() < 0 ||
          (*rit).stride() < 0) {
        return OkStatus();
      }
      conv_config->mutable_window()->add_pad_l((*rit).padding_low() + 1);
      conv_config->mutable_window()->add_pad_r((*rit).padding_high() + 1);
      conv_config->mutable_window()->add_strides((*rit).stride() + 1);
      conv_config->mutable_window()->add_rhs_dil((*rit).window_dilation() + 1);
      if ((*rit).base_dilation() != 1 || (*rit).window_reversal()) {
        return OkStatus();
      }
    }

    for (int i = 0; i < dims; i++) {
      conv_config->mutable_inp()->add_sdims(
          conv_ddata.input_spatial_dimensions()[i] + 1);
      conv_config->mutable_ker()->add_sdims(
          conv_ddata.kernel_spatial_dimensions()[i] + 1);
      conv_config->mutable_out()->add_sdims(
          conv_ddata.output_spatial_dimensions()[i] + 1);
    }

    HloInstruction* custom_call =
        conv->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {conv->mutable_operand(0), conv->mutable_operand(1)},
            "__onednn$convolution"));

    TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(conv, custom_call));
    return OkStatus();
  }

  Status HandleAdd(HloInstruction* instr) override {
    HloInstruction *addend_intermediate, *conv;
    HloInstruction* optional_conv_bitcast = nullptr;
    HloInstruction* optional_conv_convert = nullptr;

    auto pattern = m::AddAnyOrder(
        &instr,
        OptionalConvertAndBitcast(&optional_conv_convert, &optional_conv_bitcast,
                                  OneDnnConvolutionInstr(&conv))
            .WithOneUser(),
        m::Op(&addend_intermediate).WithOneUser());

    if (Match(instr, pattern)) {
      if (!IsSupportedType(conv->shape().element_type())) return OkStatus();
      if (!conv->backend_config<BackendConfig>()
               ->mutable_onednn_conv_config()
               ->mutable_fusions()->ops()
               .empty() &&
          conv->backend_config<BackendConfig>()
                  ->mutable_onednn_conv_config()
                  ->mutable_fusions()->ops(0) == OneDnnFusionConfig::BIAS) {
        return OkStatus();
      }
      std::vector<HloInstruction*> new_operands;
      for (auto operand : conv->operands()) {
        new_operands.push_back(operand);
      }

      HloInstruction* addend = nullptr;
      HloInstruction* optional_addend_broadcast = nullptr;
      auto addend_pattern = m::AnyOf<HloInstruction>(
          m::Broadcast(&optional_addend_broadcast,
                       m::Convert(&addend, m::Op())),
          m::Convert(m::Broadcast(&optional_addend_broadcast, m::Op(&addend))),
          m::Convert(&addend, m::Op()),
          m::Broadcast(&optional_addend_broadcast, m::Op(&addend)),
          m::Op(&addend));
      if (!Match(addend_intermediate, addend_pattern)) return OkStatus();

      if (CompatibleElementType(addend) && IsOperandFusible(addend, conv)) {
        new_operands.push_back(addend);
      } else {
        return OkStatus();
      }

      auto conv_call = Cast<HloCustomCallInstruction>(instr->AddInstruction(
          conv->CloneWithNewOperands(conv->shape(), new_operands)));

      auto backend_config = conv_call->backend_config<BackendConfig>();
      backend_config->mutable_onednn_conv_config()->mutable_fusions()->add_ops(
          addend->shape().rank() != 1 ? OneDnnFusionConfig::BINARY_ADD
                                      : OneDnnFusionConfig::BIAS);
      if (optional_addend_broadcast) {
        backend_config->mutable_onednn_conv_config()->set_bias_broadcast(
            true);
      }
      TF_RETURN_IF_ERROR(conv_call->set_backend_config(*backend_config));

      HloInstruction* new_instr;
      if (optional_conv_bitcast != nullptr &&
          optional_conv_bitcast->opcode() == HloOpcode::kBitcast) {
        if (conv_call->shape().element_type() == PrimitiveType::BF16) {
          auto bitcast_call =
              conv_call->AddInstruction(HloInstruction::CreateBitcast(
                  ShapeUtil::ChangeElementType(instr->shape(),
                                               PrimitiveType::BF16),
                  conv_call));
          new_instr =
              bitcast_call->AddInstruction(HloInstruction::CreateConvert(
                  ShapeUtil::ChangeElementType(bitcast_call->shape(),
                                               PrimitiveType::F32),
                  bitcast_call));
        } else {
          new_instr = conv_call->AddInstruction(
              HloInstruction::CreateBitcast(instr->shape(), conv_call));
        }
      } else {
        if (conv_call->shape().element_type() == PrimitiveType::BF16) {
          new_instr = conv_call->AddInstruction(HloInstruction::CreateConvert(
              ShapeUtil::ChangeElementType(conv_call->shape(),
                                           PrimitiveType::F32),
              conv_call));
        } else {
          new_instr = conv_call;
        }
      }
      TF_RETURN_IF_ERROR(ReplaceInstruction(instr, new_instr));
    }

    return OkStatus();
  }

  Status HandleMaximum(HloInstruction* instr) override {
    HloInstruction* conv_call;
    HloInstruction* intermediate_instr = nullptr;
    if (Match(instr, m::MaximumAnyOrder(ElementwiseSafeIntermediate(
                                            &intermediate_instr,
                                            OneDnnConvolutionInstr(&conv_call))
                                            .WithOneUser(),
                                        BcastConstScalar(0)))) {
      return FuseActivation(OneDnnFusionConfig::RELU, instr, conv_call,
                            intermediate_instr);
    }
    return OkStatus();
  }

  Status FuseActivation(OneDnnFusionConfig_FusionKind kind,
                        HloInstruction* activation, HloInstruction* conv,
                        HloInstruction* intermediate_instr = nullptr) {
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        conv->backend_config<BackendConfig>());
    auto* conv_config = backend_config.mutable_onednn_conv_config();
    conv_config->mutable_fusions()->add_ops(kind);

    std::unique_ptr<HloInstruction> output = conv->Clone();
    TF_RETURN_IF_ERROR(output->set_backend_config(backend_config));

    if (intermediate_instr) {
      output = intermediate_instr->CloneWithNewOperands(
          intermediate_instr->shape(),
          {conv->parent()->AddInstruction(std::move(output))});
    }

    return ReplaceWithNewInstruction(activation, std::move(output));
  }
};

StatusOr<bool> OneDnnConvolutionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnConvolutionRewriterVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
