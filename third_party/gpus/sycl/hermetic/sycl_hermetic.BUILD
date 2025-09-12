package(default_visibility = ["//visibility:public"])

# --- Binaries ---
# Avoid deep recursion; bring top-level bin files and the 'compiler' subdir
filegroup(
    name = "oneapi_bin",
    srcs = glob(
        [
            "oneapi/*/bin/*",
            "oneapi/*/bin/compiler/*",
        ],
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# --- Includes ---
# Includes are usually safe to recurse; still skip 'latest'
filegroup(
    name = "oneapi_include",
    srcs = glob(
        [
            "oneapi/*/include/**",
            "oneapi/*/linux/include/**",
            "oneapi/*/lib/clang/*/include/**",  # constrain depth (no ** after version)
            "oneapi/*/opt/compiler/include/**",
        ],
        exclude = ["oneapi/*/latest/**"],
        allow_empty = False,
    ),
)

# --- Generic oneAPI libs (non-MKL) ---
# Do NOT recurse; pick first-level files only to avoid symlink cycles.
filegroup(
    name = "oneapi_lib",
    srcs = glob(
        [
            "oneapi/*/lib/*",            # top-level files
            "oneapi/*/lib/intel64/*",    # common subdir; one level only
        ],
        exclude = [
            "oneapi/*/latest/**",
            "oneapi/*/lib/*/**",         # prevent deeper directories
        ],
        allow_empty = True,  # some components may not have libs
    ),
)

# --- MKL headers ---
filegroup(
    name = "mkl_include",
    srcs = glob(
        ["oneapi/mkl/*/include/**"],
        exclude = ["oneapi/mkl/latest/**"],
        allow_empty = False,
    ),
)

# --- MKL libs ---
# CRITICAL: choose *one* location and avoid recursion.
# Prefer 'lib/' first-level files; explicitly skip 'intel64/**' to break cycles.
filegroup(
    name = "mkl_lib",
    srcs = glob(
        [
            "oneapi/mkl/*/lib/*.a",
            "oneapi/mkl/*/lib/*.so",
            "oneapi/mkl/*/lib/*.so.*",
            "oneapi/mkl/*/lib/*.dylib",
            "oneapi/mkl/*/lib/*.dll",
            "oneapi/mkl/*/lib/*.lib",
        ],
        exclude = [
            "oneapi/mkl/latest/**",
            "oneapi/mkl/*/lib/intel64/**",  # avoid the cycle entirely
            "oneapi/mkl/*/lib/*/**",        # no deeper dirs
        ],
        allow_empty = False,
    ),
)
