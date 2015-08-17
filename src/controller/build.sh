#! /bin/bash


if [ -z "$BL_INSTALL_PREFIX" ]; then
    echo "ERROR: BL_INSTALL_PREFIX needs to be set!"
    exit 1
fi

NAME=${PWD##*/}

echo  -e "\033[32;1m"
echo "********** building ${NAME} **********"
echo -e "\033[0m"

rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${BL_INSTALL_PREFIX} -DCMAKE_BUILD_TYPE=DEBUG
make install
cd ..

echo  -e "\033[32;1m"
echo "********** done building ${NAME} **********"
echo -e "\033[0m"
