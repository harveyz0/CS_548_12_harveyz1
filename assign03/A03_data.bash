#!/bin/bash

# Stole from https://stackoverflow.com/questions/5947742/how-to-change-the-output-color-of-echo-in-linux
RED='\033[01;31m'
NC='\033[01;0m' # No Color

script_home=$(dirname "$0")

pushd "$script_home" &> /dev/null


if [ "$1" = "-a" ]
then
	exit 0
fi


tree_setup() {
	local data_dir="../train_images/skull2dog"
	local error_code=0

	traina_zip="trainA.zip"
	mkdir -p $data_dir
	if ! command -v unzip > /dev/null
	then
		echo -e "${RED}ERROR : Can not find the unzip command. Please install it.${NC}"
		error_code=2
	fi

	if [ -f $traina_zip ]
	then
		unzip -o $traina_zip -d $data_dir
	else
		echo -e "${RED}ERROR : Can not find the ${traina_zip} file. Please download it from my OneDrive${NC}"
		error_code=2
	fi

	return $error_code
}
tree_setup
err=$?


popd &> /dev/null

exit "$err"
