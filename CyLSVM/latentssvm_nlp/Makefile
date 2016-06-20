# Makefile for Latent Structural SVM

CC=gcc-5
CXX=g++-5
#CFLAGS= -g -Wall
CFLAGS= -O3 -fomit-frame-pointer -ffast-math
#CFLAGS = -O3 -pg
LD=g++-5
#LDFLAGS= -g
LDFLAGS= -O3
#LDFLAGS = -O3 -pg
LIBS= -lm

all: svm_coref_learn svm_coref_classify

s_v:
	$(LD) -v

clean: 
	rm -f *.o
	rm -f svm_coref_learn svm_coref_classify

svm_light_hideo_noexe: 
	cd svm_light; make svm_learn_hideo_noexe

svm_coref_learn: svm_light_hideo_noexe svm_struct_latent_cccp.o svm_struct_latent_api.o np_helper.o DisjointSets.o
	$(LD) $(LDFLAGS) svm_struct_latent_cccp.o svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct_latent_api.o np_helper.o DisjointSets.o -o svm_coref_learn -lm

svm_coref_classify: svm_light_hideo_noexe svm_struct_latent_classify.o svm_struct_latent_api.o np_helper.o DisjointSets.o
	$(LD) $(LDFLAGS) svm_struct_latent_classify.o svm_light/svm_common.o svm_struct_latent_api.o np_helper.o DisjointSets.o -o svm_coref_classify $(LIBS)

svm_struct_latent_cccp.o: svm_struct_latent_cccp.c
	$(CC) -c $(CFLAGS) svm_struct_latent_cccp.c -o svm_struct_latent_cccp.o

svm_struct_latent_api.o: svm_struct_latent_api.c svm_struct_latent_api_types.h np_helper.h
	$(CC) -c $(CFLAGS) svm_struct_latent_api.c -o svm_struct_latent_api.o

svm_struct_latent_classify.o: svm_struct_latent_classify.c
	$(CC) -c $(CFLAGS) svm_struct_latent_classify.c -o svm_struct_latent_classify.o

np_helper.o: np_helper.cc
	$(CXX) -c $(CFLAGS) np_helper.cc -o np_helper.o

DisjointSets.o: DisjointSets.cpp
	$(CXX) -c $(CFLAGS) DisjointSets.cpp -o DisjointSets.o
