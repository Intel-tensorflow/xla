
workspace=$1
cd $workspace/jax
export SYCL_TOOLKIT_PATH=$workspace/oneapi/compiler/2024.0/
export TF_L0_PATH=$workspace/level_zero/usr
bazel clean --expunge
python build/build.py --enable_sycl --bazel_options=--override_repository=xla=$workspace/xla
