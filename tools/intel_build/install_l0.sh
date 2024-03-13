
workspace=$1
cd $workspace
mkdir -p level_zero
cd level_zero
wget -c https://repositories.intel.com/gpu/ubuntu/pool/unified/l/level-zero-loader/level-zero-dev_1.14.0-744~22.04_amd64.deb
wget -c https://repositories.intel.com/gpu/ubuntu/pool/unified/l/level-zero-loader/level-zero_1.14.0-744~22.04_amd64.deb
dpkg-deb -x level-zero-dev_1.14.0-744~22.04_amd64.deb $workspace/level_zero
dpkg-deb -x level-zero_1.14.0-744~22.04_amd64.deb $workspace/level_zero
