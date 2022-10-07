CC = gcc
CFLAGS = -I$(INCDIR) -mavx -mavx2 -mfma -msse -msse2 -msse3 -O3 -Wall

export CC
export CFLAGS

all:
	@$(MAKE) -C src all

.PHONY: clean run
clean:
	@$(MAKE) -C src clean

run:
	@$(MAKE) -C src runall
