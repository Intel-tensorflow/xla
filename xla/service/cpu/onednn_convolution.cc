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

#include "xla/service/cpu/onednn_convolution.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <utility>
#include <vector>

#define EIGEN_USE_THREADS

#include "absl/base/dynamic_annotations.h"
#include "dnnl.hpp"
#include "tsl/platform/logging.h"
#include "tsl/util/onednn_threadpool.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/executable_run_options.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/runtime_lightweight_check.h"

namespace xla {
namespace cpu {
namespace {
using dnnl::algorithm;
using dnnl::convolution_forward;
using dnnl::memory;
using dnnl::prop_kind;
using dnnl::stream;
}  // namespace

dnnl::memory ReorderMemory(const dnnl::engine& engine,
                           const dnnl::memory::desc& dest_md,
                           dnnl::memory& src_mem,
                           const dnnl::stream& onednn_stream) {
  auto dest_mem = memory(dest_md, engine);
  dnnl::reorder(src_mem, dest_mem).execute(onednn_stream, src_mem, dest_mem);
  return dest_mem;
}

dnnl::memory::format_tag GetFormatTag(const int dims) {
  return (dims == 3)   ? dnnl::memory::format_tag::nwc
         : (dims == 4) ? dnnl::memory::format_tag::nhwc
         : (dims == 5) ? dnnl::memory::format_tag::ndhwc
                       : dnnl::memory::format_tag::any;
}

ABSL_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_OneDnnConvolution(
    void* result, void** args) {
  // args[0]: ptr to nargs
  // args[1]: ptr to ExecutableRunOptions
  // args[2]: ptr to OneDnnConvolutionConfig
  // args[3...]: ptrs to operands
  int arg_indx = 0;
  const int64_t num_args = *(static_cast<int64_t*>(args[arg_indx++]));

  const xla::ExecutableRunOptions* run_options =
      static_cast<const xla::ExecutableRunOptions*>(args[arg_indx++]);
  XLA_LIGHTWEIGHT_CHECK(run_options != nullptr);
  XLA_LIGHTWEIGHT_CHECK(run_options->intra_op_thread_pool() != nullptr);
  tsl::OneDnnThreadPool thread_pool(
      run_options->intra_op_thread_pool()->getPool(), false);
  dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
#ifndef ENABLE_ONEDNN_OPENMP
  auto onednn_stream =
      stream(dnnl::threadpool_interop::make_stream(cpu_engine, &thread_pool));
#else
  auto onednn_stream = stream(cpu_engine);
#endif  // ENABLE_ONEDNN_OPENMP

  std::string config_str(static_cast<const char*>(args[arg_indx++]));
  OneDnnConvolutionConfig conv_config;
  conv_config.ParseFromString(config_str);

  // Generate permutations to create memory descriptors
  std::vector<int64_t> inp_perm_axes(conv_config.dims());
  std::vector<int64_t> ker_perm_axes(conv_config.dims());
  std::vector<int64_t> out_perm_axes(conv_config.dims());

  int index_i = 0;
  int index_o = 0;
  int index_k = 0;

  inp_perm_axes[conv_config.inp().ibdim()] = index_i++;
  out_perm_axes[conv_config.out().ibdim()] = index_o++;
  ker_perm_axes[conv_config.ker().ofdim()] = index_k++;

  inp_perm_axes[conv_config.inp().ofdim()] = index_i++;
  out_perm_axes[conv_config.out().ofdim()] = index_o++;
  ker_perm_axes[conv_config.ker().ibdim()] = index_k++;

  std::vector<int64_t> inp_dim_axes(conv_config.inp().sdims().begin(),
                                    conv_config.inp().sdims().end());
  std::vector<int64_t> ker_dim_axes(conv_config.ker().sdims().begin(),
                                    conv_config.ker().sdims().end());
  std::vector<int64_t> out_dim_axes(conv_config.out().sdims().begin(),
                                    conv_config.out().sdims().end());

  std::for_each(inp_dim_axes.begin(), inp_dim_axes.end(),
                [&inp_perm_axes, &index_i](int64_t& n) {
                  n -= 1;
                  inp_perm_axes[n] = index_i++;
                });
  std::for_each(ker_dim_axes.begin(), ker_dim_axes.end(),
                [&ker_perm_axes, &index_k](int64_t& n) {
                  n -= 1;
                  ker_perm_axes[n] = index_k++;
                });
  std::for_each(out_dim_axes.begin(), out_dim_axes.end(),
                [&out_perm_axes, &index_o](int64_t& n) {
                  n -= 1;
                  out_perm_axes[n] = index_o++;
                });

  std::vector<int64_t> strds(conv_config.window().strides().begin(),
                             conv_config.window().strides().end());
  std::vector<int64_t> pad_l(conv_config.window().pad_l().begin(),
                             conv_config.window().pad_l().end());
  std::vector<int64_t> pad_r(conv_config.window().pad_r().begin(),
                             conv_config.window().pad_r().end());
  std::vector<int64_t> rhs_dil(conv_config.window().rhs_dil().begin(),
                               conv_config.window().rhs_dil().end());

  std::for_each(strds.begin(), strds.end(), [](int64_t& n) { n -= 1; });
  std::for_each(pad_l.begin(), pad_l.end(), [](int64_t& n) { n -= 1; });
  std::for_each(pad_r.begin(), pad_r.end(), [](int64_t& n) { n -= 1; });
  std::for_each(rhs_dil.begin(), rhs_dil.end(), [](int64_t& n) { n -= 2; });

  auto groups = conv_config.feature_groups();

  MemrefInfo inp_minfo(args[arg_indx++]);
  MemrefInfo ker_minfo(args[arg_indx++]);
  MemrefInfo res_minfo(result);

  // Permute memory descriptors
  auto inp_md = inp_minfo.GetOneDnnMemDesc();
  auto ker_md = ker_minfo.GetOneDnnMemDesc();
  auto res_md = res_minfo.GetOneDnnMemDesc();
  auto bias_md = memory::desc();

  std::vector<int> inp_axes(inp_perm_axes.begin(), inp_perm_axes.end());
  std::vector<int> ker_axes(ker_perm_axes.begin(), ker_perm_axes.end());
  std::vector<int> out_axes(out_perm_axes.begin(), out_perm_axes.end());

  auto new_inp_md = inp_md.permute_axes(inp_axes);
  auto new_ker_md = ker_md.permute_axes(ker_axes);
  auto new_res_md = res_md.permute_axes(out_axes);

  if (groups > 1) {
    auto corr_dims = new_ker_md.get_dims();
    corr_dims.insert(corr_dims.begin(), 1, groups);
    corr_dims[1] = corr_dims[1] / groups;
    new_ker_md = new_ker_md.reshape(corr_dims);
  }

  auto any_ker_md =
      memory::desc(new_ker_md.get_dims(), new_ker_md.get_data_type(),
                   dnnl::memory::format_tag::any);
  auto any_inp_md =
      memory::desc(new_inp_md.get_dims(), new_inp_md.get_data_type(),
                   GetFormatTag(new_inp_md.get_ndims()));
  auto any_res_md =
      memory::desc(new_res_md.get_dims(), new_res_md.get_data_type(),
                   GetFormatTag(new_res_md.get_ndims()));

  dnnl::post_ops post_ops;
  auto bias_mem = memory(nullptr);
  std::vector<std::pair<int, dnnl::memory>> postop_args;
  for (auto& fused_op : conv_config.fusions().ops()) {
    switch (fused_op) {
      case OneDnnFusionConfig::RELU:
        post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0.f, 0.f);
        break;
      case OneDnnFusionConfig::BIAS: {
        MemrefInfo bias_minfo(args[arg_indx++]);
        bias_md = bias_minfo.GetOneDnnMemDesc();
        bias_mem = memory(bias_md, cpu_engine, bias_minfo.Data());
      } break;
      default:
        LOG(FATAL) << __FILE__ << ":" << __LINE__
                   << " Attempt to call OneDNN MatMul runtime library with "
                      "unsupported post op."
                   << std::endl;
    }
  }
  
  XLA_LIGHTWEIGHT_CHECK(num_args == arg_indx);

  dnnl::primitive_attr attrs;
  if (post_ops.len() > 0) {
      attrs.set_post_ops(post_ops);
  }

  memory::dims strides_dims = strds;
  memory::dims padding_dims_l = pad_l;
  memory::dims padding_dims_r = pad_r;
  memory::dims dilations = rhs_dil;

  auto inp_mem = memory(new_inp_md, cpu_engine, inp_minfo.Data());
  auto ker_mem = memory(new_ker_md, cpu_engine, ker_minfo.Data());
  auto res_mem = memory(new_res_md, cpu_engine, res_minfo.Data());

  auto conv_pd = convolution_forward::primitive_desc(
      cpu_engine, prop_kind::forward_inference, algorithm::convolution_direct,
      any_inp_md, any_ker_md, bias_md, any_res_md, strides_dims, dilations,
      padding_dims_l, padding_dims_r, attrs);

  auto new_inp_mem = (conv_pd.src_desc() == inp_mem.get_desc())
                         ? inp_mem
                         : ReorderMemory(cpu_engine, conv_pd.src_desc(),
                                         inp_mem, onednn_stream);
  auto new_ker_mem = (conv_pd.weights_desc() == ker_mem.get_desc())
                         ? ker_mem
                         : ReorderMemory(cpu_engine, conv_pd.weights_desc(),
                                         ker_mem, onednn_stream);
  auto new_res_mem = (conv_pd.dst_desc() == res_mem.get_desc())
                         ? res_mem
                         : memory(conv_pd.dst_desc(), cpu_engine);

  auto conv_prim = convolution_forward(conv_pd);

  std::unordered_map<int, memory> conv_args{{DNNL_ARG_SRC, new_inp_mem},
                                            {DNNL_ARG_WEIGHTS, new_ker_mem},
                                            {DNNL_ARG_BIAS, bias_mem},
                                            {DNNL_ARG_DST, new_res_mem}};

  conv_prim.execute(onednn_stream, conv_args);

  if (conv_pd.dst_desc() == res_mem.get_desc()) {
    res_mem = new_res_mem;
  } else {
    dnnl::reorder(new_res_mem, res_mem)
        .execute(onednn_stream, new_res_mem, res_mem);
  }
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
