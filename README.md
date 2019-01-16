ccminer-dyn Dynamic Nvidia GPU miner
========================

This fork is only meant to be used to mine Dynamic (DYN) with Argon2d algorithm.

releases: https://github.com/duality-solutions/Dynamic-GPU-Miner-Nvidia/releases

git tree: https://github.com/duality-solutions/Dynamic-GPU-Miner-Nvidia

Donation addresses
-------------------

Please consider supporting this project by donating to these addresses (EhssanD):

	BTC  : 15h2QmsRwwwEdNNC6HbYHJU9mpbLrjUdDK

  	DYN  : DKPnTs1s71DtesAvvLMchtsj4gRFxphW55

Based on Christian Buchner's &amp; Christian H.'s CUDA project and tpruvot@github.


Building on windows
-------------------

Required: msvc2015 and cuda 10.x
Dependencies for windows are included in compat directory, using a different version of msvc will most likely require to recompile those libraries.

In order to build ccminer, choose "Release" and "x64" (this version won't work with win32)
Then click "generate"

Building on Linux (tested on Ubuntu 16.04)
------------------------------------------

A developpement environnement is required together with curl, jansson and openssl


	* sudo apt-get update && sudo apt-get -y dist-upgrade
	* sudo apt-get -y install gcc g++ build-essential automake linux-headers-$(uname -r) git gawk libcurl4-openssl-dev libjansson-dev xorg libc++-dev libgmp-dev python-dev

	* Installing CUDA 10.0 and compatible drivers from nvidia website and not from ubuntu package is usually easier
	
	* Compiling ccminner:

	./autogen.sh
	./configure
	./make


About source code dependencies for windows
------------------------------------------

This project requires some libraries to be built :

- OpenSSL (prebuilt for win)

- Curl (prebuilt for win)

- pthreads (prebuilt for win)

The tree now contains recent prebuilt openssl and curl .lib for both x86 and x64 platforms (windows).

To rebuild them, you need to clone this repository and its submodules :
    git clone https://github.com/peters/curl-for-windows.git compat/curl-for-windows


Sample command line
----------------------------------------

ccminer -a argon2d -o stratum+tcp://server:port -u walletaddress -p c=DYN

You can also use --intensity/-i (1-40) to increase gpu memory utilization.







