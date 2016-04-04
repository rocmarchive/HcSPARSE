# This script is invoked to uninstall the hcblas library and test sources
# Preliminary version

# CURRENT_WORK_DIRECTORY
current_work_dir=$PWD

# Remove system wide installed lib and headers
sudo xargs rm < $current_work_dir/build/install_manifest.txt

# Remove build
sudo rm -rf $current_work_dir/build

# remove ld_library_path
sed -i '/hcsparse/ d' ~/.bashrc
include_path="$hcsparse_install/include/hcsparse"

#remove include path
CPLUS_INCLUDE_PATH=$(REMOVE_PART="$include_path" sh -c 'echo ":$CPLUS_INCLUDE_PATH:" | sed "s@:$REMOVE_PART:@:@g;s@^:\(.*\):\$@\1@"')

exec bash
