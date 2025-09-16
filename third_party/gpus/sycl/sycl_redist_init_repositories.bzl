# third_party/gpus/sycl_redist_init_repositories.bzl
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def _maybe(name):
    return native.existing_rule(name) == None

def _http_from_item(item):
    http_archive(
        name = item["name"],
        urls = item["urls"],
        sha256 = item["sha256"],
        build_file_content = item.get("build_file_content", 'package(default_visibility=["//visibility:public"])'),
        strip_prefix = item.get("strip_prefix"),
    )

def sycl_redist_init_repositories(
        sycl_redistributions,
        ze_redistributions):
    for item in sycl_redistributions:
        if _maybe(item["name"]):
            _http_from_item(item)
    for item in ze_redistributions:
        if _maybe(item["name"]):
            _http_from_item(item)
