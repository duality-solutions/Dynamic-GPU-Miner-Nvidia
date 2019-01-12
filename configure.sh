# To change the cuda arch, edit Makefile.am and run ./build.sh

extracflags="-march=native -D_REENTRANT -falign-functions=16 -falign-jumps=16 -falign-labels=16"

CUDA_CFLAGS="-lineno -Xcompiler -Wall  -D_FORCE_INLINES" \
	CFLAGS="" ./configure CXXFLAGS="$extracflags" --with-cuda=/usr/local/cuda-10.0 --with-nvml=libnvidia-ml.so

