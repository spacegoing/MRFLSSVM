# Defualt: Python 2.7
#python-config=/usr/bin/python-config
python-config=/Users/spacegoing/anaconda/bin/python3-config
ldflags:=$(shell $(python-config) --ldflags)
cflags:=$(shell $(python-config) --cflags)

pyxfile=test
cfile=testcc.c

default: cysvm.out

$(pyxfile).c: $(pyxfile).pyx setup.py
	python setup.py build_ext --inplace
	cython $<

cysvm.out: $(pyxfile).c
	gcc $(cflags) $(ldflags) $< $(cfile) -o $@

run:
	./cysvm.out

clean:
	rm -rf $(pyxfile).c $(pyxfile).h $(pyxfile)c.so build cysvm*
