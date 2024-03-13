
workspace=$1
SCRIPT_PATH="`dirname \"$0\"`"
bash $SCRIPT_PATH/install_oneapi.sh $workspace
bash $SCRIPT_PATH/install_l0.sh $workspace
cd $workspace
git clone -b yang/runtime https://github.com/Intel-tensorflow/xla xla
cd $workspace/xla
git apply $SCRIPT_PATH/xla.patch
cd $workspace
git clone https://github.com/google/jax jax
git clone https://github.com/intel/intel-extension-for-openxla ixla
cd $workspace/jax
git checkout ceb198582b62b9e6f6bdf20ab74839b0cf1db16e
git apply $workspace/ixla/test/jax.patch
bash $SCRIPT_PATH/build.sh $workspace
