# ifndef __GRAPH_SIMP_H__
# define __GRAPH_SIMP_H__

#include "maxflow-v3.03.src/graph.h"

class GraphSimp{
public:
    GraphSimp(int node_num_max, int edge_num_max);
    int add_node(int num = 1);
    void add_edge(int i,int j,double cap, double rev_cap);
    void add_tweights(int i,double cap_source,double cap_sink);
    double maxflow();
    int what_segment(int i);


private:
    Graph<double, double, double> * g;
};
# endif
