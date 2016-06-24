# Defualt: Python 3.4
# You need to modify the following path as well as the PYTHONHOME path in cfile
#python-config=/usr/bin/python-config
python-config=/Users/spacegoing/anaconda/bin/python3-config
ldflags:=$(shell $(python-config) --ldflags)
cflags:=$(shell $(python-config) --cflags)

pyxfile=test
cfile=testcc.c

default: cysvm.o

cysvm.o: $(pyxfile).o testcc.o
	gcc $(ldflags) $? $(cfile) -o $@

testcc.o: $(cfile).c
	gcc $(cflags) $< -o $@

test.o: $(pyxfile).c
	gcc $(cflags) $< -o $@

$(pyxfile).c: $(pyxfile).pyx setup.py
	python setup.py build_ext --inplace
	cython $<

run:
	./cysvm.o

clean:
	rm -rf $(pyxfile).c $(pyxfile).h $(pyxfile)c.so build cysvm*
