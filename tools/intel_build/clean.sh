
workspace=$1
SCRIPT_PATH="`dirname \"$0\"`"
bash $SCRIPT_PATH/uninstall_oneapi.sh $workspace
rm -rf $workspace
