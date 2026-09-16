"""Intel XPU Triton archive metadata."""

XPU_TRITON_COMMIT = "bb4208fe6306e602885a77c7522804f8670fb26f"
XPU_TRITON_SHA256 = "9fb9f08d10bb9d36d898992b7333de6d592b0712db20d0a7646c84b24ea579cf"

def use_xpu_triton(repository_ctx):
    return repository_ctx.getenv("ENABLE_INTEL_XPU_TRITON", "").strip() == "1"
