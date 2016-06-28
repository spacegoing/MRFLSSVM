default: graph.o maxflow.o

graph.o: graph.cpp
	gcc-5 -c $< -o $@

maxflow.o: maxflow.cpp
	gcc-5 -c $< -o $@

clean:
	rm -f *.o
