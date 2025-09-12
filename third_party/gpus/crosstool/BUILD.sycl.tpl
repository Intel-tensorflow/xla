# This file is expanded from a template sycl_configure.bzl
# Update sycl_configure.bzl#verify_build_defines when adding new variables.

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# --- oneAPI inputs staged for the toolchain on RBE ---
# These assume you defined these targets inside the @sycl_hermetic http_archive
# via build_file/build_file_content (oneapi_bin/oneapi_include/oneapi_lib/mkl_*).
filegroup(
    name = "oneapi_bin",
    srcs = ["@sycl_hermetic//:oneapi_bin"],
)
filegroup(
    name = "oneapi_include",
    srcs = ["@sycl_hermetic//:oneapi_include"],
)
filegroup(
    name = "oneapi_lib",
    srcs = ["@sycl_hermetic//:oneapi_lib"],
)
filegroup(
    name = "mkl_include",
    srcs = ["@sycl_hermetic//:mkl_include"],
)
filegroup(
    name = "mkl_lib",
    srcs = ["@sycl_hermetic//:mkl_lib"],
)

# If Level Zero is needed remotely, uncomment these and make sure the external repos define them.
# filegroup(name = "l0_include", srcs = ["@level_zero_redist//:l0_include"])
# filegroup(name = "l0_lib",     srcs = ["@ze_loader_redist//:l0_lib"])

# Bundle everything the toolchain should carry to the remote sandbox:
filegroup(
    name = "sycl_tool_files",
    srcs = [
        ":crosstool_wrapper_driver_sycl",  # the wrapper itself
        ":oneapi_bin",
        ":oneapi_include",
        ":oneapi_lib",
        ":mkl_include",
        ":mkl_lib",
        ":l0_include",
        ":l0_lib",
    ],
)

toolchain(
    name = "toolchain-linux-x86_64",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    toolchain = ":cc-compiler-local",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-local",
        "k8": ":cc-compiler-local",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files      = ":sycl_tool_files",
    compiler_files = ":oneapi_bin",                 
    linker_files   = ":crosstool_wrapper_driver_sycl",          
    ar_files       = ":crosstool_wrapper_driver_sycl",
    as_files       = ":crosstool_wrapper_driver_sycl",
    dwp_files      = ":empty",
    objcopy_files  = ":empty",
    strip_files    = ":empty",
    # To support linker flags that need to go to the start of command line
    # we need the toolchain to support parameter files. Parameter files are
    # last on the command line and contain all shared libraries to link, so all
    # regular options will be left of them.
    supports_param_files = 1,
    toolchain_identifier = "local_linux",
    toolchain_config = ":cc-compiler-local-config",
)

cc_toolchain_config(
    name = "cc-compiler-local-config",
    cpu = "local",
    builtin_include_directories = [%{cxx_builtin_include_directories}],
    host_compiler_path = "%{host_compiler_path}",
    host_compiler_prefix = "%{host_compiler_prefix}",
    host_unfiltered_compile_flags = [%{unfiltered_compile_flags}],
    linker_bin_path = "%{linker_bin_path}",
    compiler = "unknown",
    ar_path = "%{ar_path}",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "crosstool_wrapper_driver_sycl",
    srcs = ["clang/bin/crosstool_wrapper_driver_sycl"]
)
