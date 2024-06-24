# windows_tests.bzl
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@llvm-project//llvm:lit_test.bzl", "lit_test", "package_path")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")


def create_windows_tests():
    return [
        lit_test(
            name = "%s.test" % src,
            srcs = [src],
            data = [
                "lit.cfg.py",
                "lit.site.cfg.py",
                "//xla/mlir_hlo:mlir-hlo-opt",
                "//xla/translate:xla-translate",
                "@llvm-project//llvm:FileCheck",
                "@llvm-project//llvm:not"
            ],
            # copybara:uncomment driver = "@llvm-project//mlir:run_lit",
            tags = [
                "nomsan",  # The execution engine doesn't work with msan, see b/248097619.
            ],
            deps = ["@pypi_lit//:pkg"],
        )
        for src in native.glob(["**/*.hlo"])
    ]
