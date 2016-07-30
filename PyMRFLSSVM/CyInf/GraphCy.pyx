from cython.operator cimport dereference as deref, preincrement as inc
cdef extern from "/Users/spacegoing/macCodeLab-MBP2015/Python/cyWLLEP/CyInf/GraphSimp.h":
    cdef cppclass GraphSimp:
        GraphSimp(int node_num_max, int edge_num_max)
        int add_node(int num)
        void add_edge(int i,int j,double cap, double rev_cap)
        void add_tweights(int i,double cap_source,double cap_sink)
        double maxflow()
        int what_segment(int i)

cdef class GraphCy:

    cdef GraphSimp * _thisptr

    def __cinit__(self, int node_num_max, int edge_num_max):
        self._thisptr = new GraphSimp(node_num_max, edge_num_max)
        if self._thisptr == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    def add_node(self, int num):
        return  self._thisptr.add_node(num)

    def add_edge(self, int i,int j,double cap, double rev_cap):
        self._thisptr.add_edge(i,j,cap,rev_cap)

    def add_tweights(self, int i,double cap_source,double cap_sink):
        self._thisptr.add_tweights(i, cap_source, cap_sink)

    def maxflow(self):
        return self._thisptr.maxflow()

    def what_segment(self, int i):
        return self._thisptr.what_segment(i)