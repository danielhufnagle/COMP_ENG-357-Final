# Installing OpenROAD

This is a guide to install OpenROAD on WSL2 (Ubuntu).
### Install Dependencies
First, run the following commands
```
sudo apt update && sudo apt upgrade -y
```
```
sudo apt install -y build-essential cmake git tcl-dev tk-dev libffi-dev libboost-all-dev libeigen3-dev flex bison libreadline-dev libncurses-dev libgsl-dev libx11-dev libxaw7-dev libglu1-mesa-dev freeglut3-dev libcurl4-openssl-dev libspdlog-dev qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools swig python3 python3-dev python3-pip
```
### Clone the OpenROAD Repository
```
git clone https://github.com/The-OpenROAD-Project/OpenROAD.git && cd OpenROAD
```
```
git submodule update --init --recursive
```
With this, the repo is cloned
### Double check WSL Resource Allocation
In WINDOWS, make `.wslconfig` in your home folder (`C:\Users\your-user-name`), and in `.wslconfig`, make sure the following is present
```
[wsl2]
memory=8GB
processors=4
swap=16GB
swapFile=C:\\Users\\your-user-name\\wsl-swap.vhdx
```
The additional `swap` and `swapFile` is necessary if your machine has less than 16GB or RAM (building OpenROAD is very resource intensive). Then run 
```
wsl --shutdown
```
### Build OpenROAD
Now, go back to your OpenROAD directory in Ubuntu and run the following:
```
./etc/DependencyInstaller.sh
```
```
mkdir -p build && cd build
```
```
cmake ..
```
```
make -j$(nproc)
```
With this, OpenROAD should be built. To run it:
```
cd src
```
```
./openroad
```
or to open with the gui 
```
./openroad -gui
```
