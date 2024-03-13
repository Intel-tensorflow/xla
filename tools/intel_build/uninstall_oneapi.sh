
workspace=$1
echo "Uninstall Intel OneAPI in $workspace/oneapi"
cd $workspace
if ! [ -f $workspace/l_BaseKit_p_2024.0.1.46.sh ]; then
  echo "Download oneAPI package for uninstallation."
  wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/163da6e4-56eb-4948-aba3-debcec61c064/l_BaseKit_p_2024.0.1.46.sh
fi

bash l_BaseKit_p_2024.0.1.46.sh -a -s --eula accept --action remove --install-dir $workspace/oneapi --log-dir $workspace/oneapi/log --download-cache $workspace/oneapi/cache --components=intel.oneapi.lin.dpcpp-cpp-compiler:intel.oneapi.lin.mkl.devel
