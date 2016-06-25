# Defualt: Python 3.4
# You need to modify the following path as well as the PYTHONHOME path in cfile
#python-config=/usr/bin/python-config
python-config=/Users/spacegoing/anaconda/bin/python3-config
ldflags:=$(shell $(python-config) --ldflags)
cflags:=$(shell $(python-config) --cflags)
gcc=gcc-5

pyxfile=CCyHelper
wrapperFile=CythonWrapper
mainFile=ttt

default: cysvm.o

cysvm.o: $(pyxfile).o $(mainFile).o $(wrapperFile).o
	$(gcc) $(ldflags) $? -o $@ -lm

$(wrapperFile).o: $(wrapperFile).c
	$(gcc) -c $(cflags) $< -o $@

$(pyxfile).o: $(pyxfile).c
	$(gcc) -c $(cflags) $< -o $@

$(mainFile).o: $(mainFile).c
	$(gcc) -c $(cflags) $< -o $@

$(pyxfile).c: $(pyxfile).pyx #setup.py
#python setup.py build_ext --inplace
	cython $<

run:
	./cysvm.o

clean:
	rm -rf build cysvm*
	rm -f *.o *.so $(pyxfile).c $(pyxfile).h $(pyxfile)_api.h
