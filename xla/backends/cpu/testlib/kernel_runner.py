# Copyright 2024 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU specific kernel runner implementations."""

from xla.backends.cpu.testlib import kernel_runner_extention

# go/keep-sorted start
ElementalKernelEmitter = kernel_runner_extention.ElementalKernelEmitter
KernelRunner = kernel_runner_extention.KernelRunner
LlvmIrKernelEmitter = kernel_runner_extention.LlvmIrKernelEmitter
LlvmIrKernelSpec = kernel_runner_extention.LlvmIrKernelSpec
# go/keep-sorted end