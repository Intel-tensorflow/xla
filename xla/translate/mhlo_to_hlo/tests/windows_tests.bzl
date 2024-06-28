# windows_tests.bzl

load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@llvm-project//llvm:lit_test.bzl", "lit_test", "package_path")

#Create and execute the tests on Windows platform

def windows_tests(a):
   if a!=[]:
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
                "nomsan",
            ],
            deps = ["@pypi_lit//:pkg"], # The execution engine doesn't work with msan, see b/248097619.
        )
        for src in native.glob(["**/*.mlir"])
    ] 