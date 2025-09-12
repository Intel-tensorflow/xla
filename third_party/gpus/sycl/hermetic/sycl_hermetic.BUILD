package(default_visibility = ["//visibility:public"])

# oneAPI compiler/runtime binaries
filegroup(
    name = "oneapi_bin",
    srcs = glob(
        ["oneapi/*/bin/*"],   # no ** to avoid following any weird symlinks
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# Includes (compiler headers, clang resource headers, etc.)
filegroup(
    name = "oneapi_include",
    srcs = glob(
        [
            "oneapi/*/include/**",               # headers trees are usually safe
            "oneapi/*/linux/include/**",
            "oneapi/*/lib/clang/*/include/**",   # constrain depth (no ** after version)
            "oneapi/*/opt/compiler/include/**",
        ],
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# oneAPI generic libs (limit recursion)
filegroup(
    name = "oneapi_lib",
    srcs = glob(
        [
            "oneapi/*/lib/*",            # top-level only
            "oneapi/*/lib/intel64/*",    # common subdir; single level
        ],
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# MKL headers (headers trees are fine)
filegroup(
    name = "mkl_include",
    srcs = glob(
        ["oneapi/mkl/*/include/**"],
        exclude = ["oneapi/mkl/latest/**"],
        allow_empty = False,
    ),
)

# MKL libs — **DO NOT** use **; only take first level in intel64
filegroup(
    name = "mkl_lib",
    srcs = glob(
        [
            "oneapi/mkl/*/lib/*",           # top-level (some platforms)
            "oneapi/mkl/*/lib/intel64/*",   # primary location
        ],
        exclude = ["oneapi/mkl/latest/**"],
        allow_empty = False,
    ),
)
