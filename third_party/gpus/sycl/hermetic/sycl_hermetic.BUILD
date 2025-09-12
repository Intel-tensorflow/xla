package(default_visibility = ["//visibility:public"])

# oneAPI compiler/runtime binaries (icpx/clang, tools)
filegroup(
    name = "oneapi_bin",
    srcs = glob(
        ["oneapi/*/bin/**"],
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# Includes (compiler headers, clang resource headers, etc.)
filegroup(
    name = "oneapi_include",
    srcs = glob(
        [
            "oneapi/*/include/**",
            "oneapi/*/linux/include/**",
            "oneapi/*/lib/clang/**/include/**",
            "oneapi/*/opt/compiler/include/**",
        ],
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# Libraries (compiler runtimes, etc.)
filegroup(
    name = "oneapi_lib",
    srcs = glob(
        ["oneapi/*/lib/**"],
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# MKL headers/libs — explicitly skip the 'latest' symlink to avoid loops
filegroup(
    name = "mkl_include",
    srcs = glob(
        ["oneapi/mkl/*/include/**"],
        exclude = ["oneapi/mkl/latest/**"],
        allow_empty = False,
    ),
)

filegroup(
    name = "mkl_lib",
    srcs = glob(
        ["oneapi/mkl/*/lib/**"],
        exclude = ["oneapi/mkl/latest/**"],
        allow_empty = False,
    ),
)
