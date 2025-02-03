/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/ffi/attribute_map.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/ffi/call_frame.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla::ffi {

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertBoolAttr(
    std::string_view name, mlir::BoolAttr boolean) {
  return static_cast<bool>(boolean.getValue());
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertStringAttr(
    std::string_view name, mlir::StringAttr str) {
  return str.getValue().str();
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertIntegerAttr(
    std::string_view name, mlir::IntegerAttr integer) {
  if (integer.getType().isUnsignedInteger()) {
    switch (integer.getType().getIntOrFloatBitWidth()) {
      case 8:
        return static_cast<uint8_t>(integer.getUInt());
      case 16:
        return static_cast<uint16_t>(integer.getUInt());
      case 32:
        return static_cast<uint32_t>(integer.getUInt());
      case 64:
        return static_cast<uint64_t>(integer.getUInt());
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported integer attribute bit width for attribute: ", name));
    }
  } else {
    switch (integer.getType().getIntOrFloatBitWidth()) {
      case 8:
        return static_cast<int8_t>(integer.getInt());
      case 16:
        return static_cast<int16_t>(integer.getInt());
      case 32:
        return static_cast<int32_t>(integer.getInt());
      case 64:
        return static_cast<int64_t>(integer.getInt());
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported integer attribute bit width for attribute: ", name));
    }
  }
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertFloatAttr(
    std::string_view name, mlir::FloatAttr fp) {
  switch (fp.getType().getIntOrFloatBitWidth()) {
    case 32:
      return static_cast<float>(fp.getValue().convertToFloat());
    case 64:
      return static_cast<double>(fp.getValue().convertToDouble());
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported float attribute bit width for attribute: ", name));
  }
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertArrayAttr(
    std::string_view name, mlir::DenseArrayAttr arr) {
  if (auto dense = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseI16ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseI32ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseF32ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else if (auto dense = mlir::dyn_cast<mlir::DenseF64ArrayAttr>(arr)) {
    return dense.asArrayRef().vec();
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported array element type for attribute: ", name));
  }
}

template <typename T>
static std::vector<T> CopyDenseElementsToVec(
    mlir::DenseIntOrFPElementsAttr arr) {
  auto it = arr.getValues<T>();
  return std::vector<T>(it.begin(), it.end());
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertDenseElementsAttr(
    std::string_view name, mlir::DenseIntOrFPElementsAttr arr) {
  auto type = arr.getElementType();
  if (type.isInteger()) {
    if (type.isUnsignedInteger()) {
      switch (type.getIntOrFloatBitWidth()) {
        case 8:
          return CopyDenseElementsToVec<uint8_t>(arr);
        case 16:
          return CopyDenseElementsToVec<uint16_t>(arr);
        case 32:
          return CopyDenseElementsToVec<uint32_t>(arr);
        case 64:
          return CopyDenseElementsToVec<uint64_t>(arr);
      }
    } else {
      switch (type.getIntOrFloatBitWidth()) {
        case 8:
          return CopyDenseElementsToVec<int8_t>(arr);
        case 16:
          return CopyDenseElementsToVec<int16_t>(arr);
        case 32:
          return CopyDenseElementsToVec<int32_t>(arr);
        case 64:
          return CopyDenseElementsToVec<int64_t>(arr);
      }
    }
  } else if (type.isIntOrFloat()) {
    switch (type.getIntOrFloatBitWidth()) {
      case 32:
        return CopyDenseElementsToVec<float>(arr);
      case 64:
        return CopyDenseElementsToVec<double>(arr);
    }
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported array element type for attribute: ", name));
}

static absl::StatusOr<CallFrameBuilder::Attribute> ConvertDictionaryAttr(
    std::string_view name, mlir::DictionaryAttr dict) {
  TF_ASSIGN_OR_RETURN(auto attrs, BuildAttributesMap(dict));
  return CallFrameBuilder::Dictionary{
      std::make_shared<CallFrameBuilder::AttributesMap>(std::move(attrs))};
}

#ifndef TENSORFLOW_USE_SYCL
absl::StatusOr<CallFrameBuilder::AttributesMap> BuildAttributesMap(
    mlir::DictionaryAttr dict) {
  CallFrameBuilder::AttributesMap attributes;
  for (auto& kv : dict) {
    std::string_view name = kv.getName().strref();
    mlir::Attribute value = kv.getValue();

    // Wraps attribute conversion function into callable object.
    auto convert_with = [&](auto converter_fn) {
      return [&, fn = converter_fn](auto attr) -> absl::Status {
        TF_ASSIGN_OR_RETURN(attributes[name], fn(name, attr));
        return absl::OkStatus();
      };
    };

    TF_RETURN_IF_ERROR(
        llvm::TypeSwitch<mlir::Attribute, absl::Status>(value)
            .Case<mlir::BoolAttr>(convert_with(ConvertBoolAttr))
            .Case<mlir::IntegerAttr>(convert_with(ConvertIntegerAttr))
            .Case<mlir::FloatAttr>(convert_with(ConvertFloatAttr))
            .Case<mlir::DenseArrayAttr>(convert_with(ConvertArrayAttr))
            .Case<mlir::DenseIntOrFPElementsAttr>(
                convert_with(ConvertDenseElementsAttr))
            .Case<mlir::StringAttr>(convert_with(ConvertStringAttr))
            .Case<mlir::DictionaryAttr>(convert_with(ConvertDictionaryAttr))
            .Default([&](mlir::Attribute) {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported attribute type for attribute: ", name));
            }));
  }

  return attributes;
}
#else
absl::StatusOr<CustomCallThunk::AttributesMap> BuildAttributesMap(
  const HloCustomCallInstruction* instr){
  CustomCallThunk::AttributesMap attrs;
  attrs["backend_config_str"] = instr->raw_backend_config_string();
  if (IsCustomCallToDnnConvolution(*instr)) {
    TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
    const Window& window = instr->window();
    const ConvolutionDimensionNumbers& dnums = instr->convolution_dimension_numbers();
    const int num_dimensions = window.dimensions_size();
    const Shape& operand0_shape = instr->operand(0)->shape();
    const Shape& operand1_shape = instr->operand(1)->shape();
    const Shape& result_shape = instr->shape().tuple_shapes(0);

    attrs["window_ShortDebugString"] = window.ShortDebugString();
    attrs["window_num_dimensions"] = window.dimensions_size();
    for (int i = 0; i < window.dimensions_size(); ++i) {
      attrs["window_padding_low_" + std::to_string(i)] =
        window.dimensions(i).padding_low();
      attrs["window_padding_high_" + std::to_string(i)] =
        window.dimensions(i).padding_high();
      attrs["window_stride_" + std::to_string(i)] = window.dimensions(i).stride();
      attrs["window_dilation_" + std::to_string(i)] =
        window.dimensions(i).window_dilation();
    }

    attrs["dnums_ShortDebugString"] = dnums.ShortDebugString();
    attrs["input_feature_dimension"] = dnums.input_feature_dimension();
    attrs["input_batch_dimension"] = dnums.input_batch_dimension();
    attrs["output_feature_dimension"] = dnums.output_feature_dimension();
    attrs["kernel_input_feature_dimension"] =
      dnums.kernel_input_feature_dimension();
    attrs["kernel_output_feature_dimension"] =
      dnums.kernel_output_feature_dimension();
    for (int i = 0; i < num_dimensions; ++i) {
      attrs["input_spatial_dimensions_" + std::to_string(i)] =
        dnums.input_spatial_dimensions(i);
      attrs["output_spatial_dimensions_" + std::to_string(i)] =
        dnums.output_spatial_dimensions(i);
      attrs["kernel_spatial_dimensions_" + std::to_string(i)] =
        dnums.kernel_spatial_dimensions(i);
      attrs["kernel_spatial_dimensions_" + std::to_string(i)] =
        dnums.kernel_spatial_dimensions(i);
    }
    stream_executor::dnn::DataLayout input_dl;
    stream_executor::dnn::FilterLayout filter_dl;
    stream_executor::dnn::DataLayout output_dl;
    if(kind == CudnnConvKind::kForward || kind == CudnnConvKind::kForwardActivation){
      TF_ASSIGN_OR_RETURN(std::tie(input_dl, filter_dl, output_dl),
                          XlaConvShapesToStreamExecutorLayouts(
                            dnums, operand0_shape, operand1_shape, result_shape));
    }else if(kind == CudnnConvKind::kBackwardInput){
      TF_ASSIGN_OR_RETURN(std::tie(input_dl, filter_dl, output_dl),
                          XlaConvShapesToStreamExecutorLayouts(
                            dnums, result_shape, operand1_shape, operand0_shape));
    }else if(kind == CudnnConvKind::kBackwardFilter){
      TF_ASSIGN_OR_RETURN(std::tie(input_dl, filter_dl, output_dl),
                          XlaConvShapesToStreamExecutorLayouts(
                            dnums, operand0_shape, result_shape, operand1_shape));
    }else{
      return Internal("Unkown convolution kind");
    }
    attrs["input_dl"] = static_cast<int32_t>(input_dl);
    attrs["filter_dl"] = static_cast<int32_t>(filter_dl);
    attrs["output_dl"] = static_cast<int32_t>(output_dl);
  }else if (IsLegacyCublasMatmul(*instr) || IsCublasLtMatmul(*instr)) {
    const Shape& lhs_shape = instr->operand(0)->shape();
    const Shape& rhs_shape = instr->operand(1)->shape();
    const Shape& output_shape =
        instr->shape().IsTuple() ? instr->shape().tuple_shapes(0) : instr->shape();
    for (int i = 0; i < lhs_shape.layout().minor_to_major().size(); ++i) {
        attrs["lhs_minor_to_major_" + std::to_string(i)] =
            lhs_shape.layout().minor_to_major()[i];
    }
    for (int i = 0; i < rhs_shape.layout().minor_to_major().size(); ++i) {
        attrs["rhs_minor_to_major_" + std::to_string(i)] =
            rhs_shape.layout().minor_to_major()[i];
    }
    for (int i = 0; i < output_shape.layout().minor_to_major().size(); ++i) {
        attrs["output_minor_to_major_" + std::to_string(i)] =
            output_shape.layout().minor_to_major()[i];
    }
  }else return absl::InternalError("Unknown CustomCall To SYCL FFI Call");
  return attrs;
}
#endif // TENSORFLOW_USE_SYCL

}  // namespace xla::ffi
