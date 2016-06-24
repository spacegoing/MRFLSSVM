# Defualt: Python 3.4
# You need to modify the following path as well as the PYTHONHOME path in cfile
#python-config=/usr/bin/python-config
python-config=/Users/spacegoing/anaconda/bin/python3-config
ldflags:=$(shell $(python-config) --ldflags)
cflags:=$(shell $(python-config) --cflags)
gcc=gcc-5

pyxfile=test
cfile=testcc

default: cysvm.o

cysvm.o: $(pyxfile).o $(cfile).o
	$(gcc) $(ldflags) $? -o $@ -lm

$(cfile).o: $(cfile).c
	$(gcc) -c $(cflags) $< -o $@

$(pyxfile).o: $(pyxfile).c
	$(gcc) -c $(cflags) $< -o $@

run:
	./cysvm.o

$(pyxfile).c: $(pyxfile).pyx setup.py
	python setup.py build_ext --inplace
	cython $<

clean:
	rm -rf build cysvm*
	rm -f *.o *.so $(pyxfile).c $(pyxfile).h
