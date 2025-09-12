package(default_visibility = ["//visibility:public"])

# oneAPI compiler/runtime binaries (icpx/clang, tools)
filegroup(
    name = "oneapi_bin",
    srcs = glob(["oneapi/*/bin/**"], allow_empty = False),
)

# All relevant include trees (version-agnostic)
filegroup(
    name = "oneapi_include",
    srcs = glob([
        "oneapi/*/include/**",
        "oneapi/*/linux/include/**",
        "oneapi/*/lib/clang/**/include/**",
        "oneapi/*/opt/compiler/include/**",
    ], allow_empty = False),
)

# oneAPI libs (compiler/clang runtimes, etc.)
filegroup(
    name = "oneapi_lib",
    srcs = glob(["oneapi/*/lib/**"], allow_empty = False),
)

# MKL headers and libs (version-agnostic)
filegroup(
    name = "mkl_include",
    srcs = glob(["oneapi/mkl/*/include/**"], allow_empty = False),
)

filegroup(
    name = "mkl_lib",
    srcs = glob(["oneapi/mkl/*/lib/**"], allow_empty = False),
)

