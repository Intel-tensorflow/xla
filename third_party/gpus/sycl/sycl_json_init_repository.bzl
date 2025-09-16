# third_party/gpus/sycl_json_init_repository.bzl
def _sycl_json_repo_impl(repo_ctx):
    repo_ctx.file("WORKSPACE", "")
    repo_ctx.file("BUILD.bazel", "package(default_visibility=[\"//visibility:public\"])")
    # Keep this simple; you can later switch to reading a JSON manifest if needed.
    repo_ctx.file("distributions.bzl", """
SYCL_REDISTRIBUTIONS = [
    {
        "name": "sycl_hermetic",
        "urls": ["https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/intel-oneapi-base-toolkit-2025.1.3.7.tar.gz"],
        "sha256": "2213104bd122336551aa144512e7ab99e4a84220e77980b5f346edc14ebd458a",
        "strip_prefix": None,
        "build_file_content": "package(default_visibility=[\\"//visibility:public\\"])",
    },
]

ZE_REDISTRIBUTIONS = [
    {
        "name": "level_zero_redist",
        "urls": ["https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/level-zero-1.21.10.tar.gz"],
        "sha256": "e0ff1c6cb9b551019579a2dd35c3a611240c1b60918c75345faf9514142b9c34",
        "strip_prefix": "level-zero-1.21.10",  # adjust if your tar has no top dir
        "build_file_content": "package(default_visibility=[\\"//visibility:public\\"])",
    },
    {
        "name": "ze_loader_redist",
        "urls": ["https://tensorflow-file-hosting.s3.us-east-1.amazonaws.com/ze_loader_libs.tar.gz"],
        "sha256": "71cbfd8ac59e1231f013e827ea8efe6cf5da36fad771da2e75e202423bd6b82e",
        "strip_prefix": None,
        "build_file_content": "package(default_visibility=[\\"//visibility:public\\"])",
    },
]
""")

sycl_json_init_repository = repository_rule(
    implementation = _sycl_json_repo_impl,
    attrs = {},  # keep room for future knobs (e.g., alt manifests)
    local = True,
)
