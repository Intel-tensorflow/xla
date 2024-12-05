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

#include "xla/backends/gpu/collectives/gpu_collectives.h"

#include <cstddef>

#include "absl/status/statusor.h"
#include "xla/core/collectives/collectives.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

GpuCollectives::Device::Device(se::StreamExecutor* stream_executor)
    : stream_executor_(stream_executor) {}

se::StreamExecutor* GpuCollectives::Device::stream_executor() const {
  return stream_executor_;
}

absl::StatusOr<GpuCollectives::Device*> GpuCollectives::TryCast(
    Collectives::Device* device) {
  if (auto* gpu_device = tsl::down_cast<Device*>(device)) {
    return gpu_device;
  }
  return InvalidArgument("Collectvies device is not a GPU device");
}

absl::StatusOr<const GpuCollectives::Config*> GpuCollectives::TryCast(
    const Collectives::Config* config) {
  if (auto* gpu_config = tsl::down_cast<const Config*>(config)) {
    return gpu_config;
  }
  return InvalidArgument("Collectvies config is not a GPU config");
}

se::DeviceMemoryBase GpuCollectives::Slice(se::DeviceMemoryBase buff,
                                           PrimitiveType dtype, size_t offset,
                                           size_t count) {
  size_t multiplier = ShapeUtil::ByteSizeOfPrimitiveType(dtype);
  return buff.GetByteSlice(offset * multiplier, count * multiplier);
}

}  // namespace xla::gpu