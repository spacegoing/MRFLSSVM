#include "GraphSimp.h"
#include <stdio.h>
GraphSimp::GraphSimp(int node_num_max, int edge_num_max){
    g = new Graph<double,double,double>(node_num_max, edge_num_max);
}

int GraphSimp::add_node(int num){
    return g->add_node(num);
}
void GraphSimp::add_edge(int i,int j,double cap, double rev_cap){
    g->add_edge(i,j,cap,rev_cap);
}
void GraphSimp::add_tweights(int i,double cap_source,double cap_sink){
    g->add_tweights(i,cap_source,cap_sink);
}

double GraphSimp::maxflow(){
    return g->maxflow();
}
int GraphSimp::what_segment(int i){
    return (int)(g->what_segment(i) == Graph<double,double,double>::SOURCE) ? 1.0 : 0.0;
}
