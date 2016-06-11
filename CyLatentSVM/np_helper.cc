// noun phrase coreference code

#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <bitset>
#include <iostream>
#include "DisjointSets.h"
#include "np_helper.h"
#include "assert.h"

extern "C" {
//#include "svm_struct_latent_api_types.h"
#include "./svm_light/svm_common.h"
}

#define DEBUG 0

using namespace std;

/* Kruskal's algorithm */
vector<pair<int, int> > kruskal(DisjointSets &s, vector<pair<int, int> >::const_iterator beg,
                                vector<pair<int, int> >::const_iterator end) {

  vector<pair<int, int> > mst;

  for (vector<pair<int, int> >::const_iterator iter = beg; iter != end; ++iter) {
    if (s.FindSet(iter->first) != s.FindSet(iter->second)) {
      s.Union(s.FindSet(iter->first), s.FindSet(iter->second));
      mst.push_back(*iter);
    }
  }

  return (mst);

}


void classify_struct_example_helper(PATTERN x, LABEL *y, LATENT_VAR *h, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  vector<pair<double, pair<int, int> > > weighted_edges;
  vector<pair<int, int> > sorted_edges;
  vector<pair<int, int> > mst;

  for (int i = 0; i < x.num_nps; i++) {
    for (int j = 0; j < i; j++) {
      double edge_cost = sprod_ns(sm->w, x.pair_features[i][j]);
      if (edge_cost > 0) {
        weighted_edges.push_back(make_pair(-edge_cost, make_pair(i, j)));
      }
    }
  }
  sort(weighted_edges.begin(), weighted_edges.end());

  for (vector<pair<double, pair<int, int> > >::const_iterator iter = weighted_edges.begin();
       iter != weighted_edges.end(); ++iter) {
    sorted_edges.push_back(iter->second);
  }

  DisjointSets s(x.num_nps);
  mst = kruskal(s, sorted_edges.begin(), sorted_edges.end());

  /* fill in information for latent variable h */
  LL_NODE *ptr = (LL_NODE *) malloc(sizeof(LL_NODE));
  ptr->next = NULL;
  h->head = ptr;
  for (vector<pair<int, int> >::const_iterator iter = mst.begin(); iter != mst.end(); ++iter) {
    ptr->next = (LL_NODE *) malloc(sizeof(LL_NODE));
    ptr = ptr->next;
    ptr->u = iter->first;
    ptr->v = iter->second;
    assert(ptr->u > ptr->v);
    ptr->next = NULL;
  }
  LL_NODE *temp = h->head;
  h->head = (h->head)->next;
  free(temp);

  y->num_nps = x.num_nps;
  y->num_clusters = s.NumSets();
  y->cluster_id = (int *) malloc(sizeof(int) * x.num_nps);
  for (int i = 0; i < x.num_nps; i++) {
    y->cluster_id[i] = s.FindSet(i);
  }

}


void find_most_violated_constraint_marginrescaling_helper(PATTERN x, LABEL y, LABEL *ybar, LATENT_VAR *hbar,
                                                          STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  vector<pair<double, pair<int, int> > > weighted_edges;
  vector<pair<int, int> > sorted_edges;
  vector<pair<int, int> > mst;
#if (DEBUG == 1)
  map<pair<int,int>,double> edge_cost_dict;
#endif

  LL_NODE *ptr, *temp;

  for (int i = 0; i < x.num_nps; i++) {
    for (int j = 0; j < i; j++) {
      double edge_cost = sprod_ns(sm->w, x.pair_features[i][j]);
      if (y.cluster_id[i] != y.cluster_id[j]) {
        //edge_cost += 1.0;
        edge_cost += sparm->cost_factor;
      } else {
        edge_cost -= 1.0;
      }
      if (edge_cost > 0) {
        /* note the minus sign in edge cost, so as to turn the argmax into a minimum spanning tree problem (also for sorting) */
        weighted_edges.push_back(make_pair(-edge_cost, make_pair(i, j)));
#if (DEBUG == 1)
        edge_cost_dict[make_pair(i,j)] = edge_cost;
#endif
      }
    }
  }
  sort(weighted_edges.begin(), weighted_edges.end());

  for (vector<pair<double, pair<int, int> > >::const_iterator iter = weighted_edges.begin();
       iter != weighted_edges.end(); ++iter) {
    sorted_edges.push_back(iter->second);
  }

  DisjointSets s(x.num_nps);
  mst = kruskal(s, sorted_edges.begin(), sorted_edges.end());

#if (DEBUG == 1)
  double mst_cost = 0.0;
#endif
  ptr = (LL_NODE *) malloc(sizeof(LL_NODE));
  ptr->next = NULL;
  hbar->head = ptr;
  for (vector<pair<int, int> >::const_iterator iter = mst.begin(); iter != mst.end(); ++iter) {
    ptr->next = (LL_NODE *) malloc(sizeof(LL_NODE));
    ptr = ptr->next;
    ptr->u = iter->first;
    ptr->v = iter->second;
    assert(ptr->u > ptr->v);
    ptr->next = NULL;
#if (DEBUG == 1)
    mst_cost += edge_cost_dict[*iter];
#endif
  }
  temp = hbar->head;
  hbar->head = (hbar->head)->next;
  free(temp);
#if (DEBUG == 1)
  printf("mst_cost in find_most_violated_constraint: %.8g\n", mst_cost); fflush(stdout);
  printf("y.num_nps: %d, y.num_clusters: %d\n", y.num_nps, y.num_clusters); fflush(stdout);
#endif

  ybar->cluster_id = (int *) malloc(sizeof(int) * y.num_nps);
  for (int i = 0; i < y.num_nps; i++) {
    ybar->cluster_id[i] = s.FindSet(i);
  }

}


LATENT_VAR infer_latent_variables_helper(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) {
  vector<pair<double, pair<int, int> > > weighted_edges;
  vector<pair<int, int> > sorted_edges;
  vector<pair<int, int> > mst;
  LATENT_VAR h;
  LL_NODE *ptr, *temp;

  for (int i = 0; i < x.num_nps; i++) {
    for (int j = 0; j < i; j++) {
      /* only add edges within the same cluster defined by y */
      if (y.cluster_id[i] == y.cluster_id[j]) {
        double edge_cost = sprod_ns(sm->w, x.pair_features[i][j]);
        /* add edge even if cost is negative, since spanning tree has to span the clustering defined by y */
        weighted_edges.push_back(make_pair(-edge_cost, make_pair(i, j)));
      }
    }
  }

  sort(weighted_edges.begin(), weighted_edges.end());

  for (vector<pair<double, pair<int, int> > >::const_iterator iter = weighted_edges.begin();
       iter != weighted_edges.end(); ++iter) {
    sorted_edges.push_back((*iter).second);
  }

  DisjointSets s(x.num_nps);
  mst = kruskal(s, sorted_edges.begin(), sorted_edges.end());

  ptr = (LL_NODE *) malloc(sizeof(LL_NODE));
  ptr->next = NULL;
  h.head = ptr;
  for (vector<pair<int, int> >::const_iterator iter = mst.begin(); iter != mst.end(); ++iter) {
    ptr->next = (LL_NODE *) malloc(sizeof(LL_NODE));
    ptr = ptr->next;
    ptr->u = iter->first;
    ptr->v = iter->second;
    assert(ptr->u > ptr->v);

    ptr->next = NULL;
  }
  temp = h.head;
  h.head = (h.head)->next;
  free(temp);

  return (h);

}

/* modified from Tom's function */
SAMPLE read_struct_examples_helper(char *filename, STRUCT_LEARN_PARM *sparm) {

  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    printf("Cannot open input file %s!\n", filename);
    exit(1);
  }

  map<int, map<int, map<int, vector<WORD> > > > pair_features;
  map<int, int> num_nps;
  map<int, map<int, vector<int> > > coreferent_nps;
  set<int> list_doc_id;

  sparm->max_feature_key = 0;
  while (true) {
    FNUM key;
    FVAL value;
    int positive, doc_id, np_index1, np_index2;
    vector<WORD> np_pair;

    if (fscanf(fp, "%d", &positive) != 1) {
      break;
    }
    while (fscanf(fp, "%ld:%f", &key, &value) == 2) {
      WORD w;
      w.wnum = key;
      w.weight = value;
      np_pair.push_back(w);
      if (key > sparm->max_feature_key) sparm->max_feature_key = key;
    }
    fscanf(fp, "#%d%d%d", &doc_id, &np_index1, &np_index2);

    /* assume file is formatted in such a way that np_index2>np_index1 */
    assert(np_index2 > np_index1);

    if (np_index2 > num_nps[doc_id]) {
      num_nps[doc_id] = np_index2;
    }
    np_index1--;
    np_index2--;

    pair_features[doc_id][np_index2][np_index1] = np_pair;

    if (positive == 1) {
      coreferent_nps[doc_id][np_index1].push_back(np_index2);
      coreferent_nps[doc_id][np_index2].push_back(np_index1);
    }

    list_doc_id.insert(doc_id);

  }


  fclose(fp);

  long n = list_doc_id.size();

  SAMPLE sample;
  sample.n = n;
  sample.examples = (EXAMPLE *) malloc(sizeof(EXAMPLE) * sample.n);

  int i = 0;
  for (set<int>::const_iterator set_it = list_doc_id.begin(); set_it != list_doc_id.end(); ++set_it) {
    sample.examples[i].x.num_nps = num_nps[*set_it];
    sample.examples[i].x.pair_features = (SVECTOR ***) malloc(sizeof(SVECTOR **) * num_nps[*set_it]);
    for (int j = 0; j < num_nps[*set_it]; j++) {
      if (j > 0) { // to avoid allocating 0 memory
        sample.examples[i].x.pair_features[j] = (SVECTOR **) malloc(sizeof(SVECTOR *) * j);
      }
      for (int k = 0; k < j; k++) {
        WORD w;
        w.wnum = 0;
        w.weight = 0;
        pair_features[*set_it][j][k].push_back(w);

        WORD *words;
        words = new WORD[pair_features[*set_it][j][k].size()];
        copy(pair_features[*set_it][j][k].begin(), pair_features[*set_it][j][k].end(), words);

        sample.examples[i].x.pair_features[j][k] = create_svector(words, "", 1);

        delete[] words;
      }
    }

    sample.examples[i].y.num_nps = num_nps[*set_it];
    sample.examples[i].y.cluster_id = (int *) malloc(sizeof(int) * num_nps[*set_it]);
    set<int> cluster_rep;
    for (int j = 0; j < num_nps[*set_it]; j++) {
      coreferent_nps[*set_it][j].push_back(j);
      sort(coreferent_nps[*set_it][j].begin(), coreferent_nps[*set_it][j].end());
      /* take smallest id for noun phrase as the cluster id */
      sample.examples[i].y.cluster_id[j] = coreferent_nps[*set_it][j][0];
      cluster_rep.insert(coreferent_nps[*set_it][j][0]);
    }
    sample.examples[i].y.num_clusters = cluster_rep.size();

    vector<pair<int, int> > edges;
    int u, v;
    for (set<int>::const_iterator iter_i = cluster_rep.begin(); iter_i != cluster_rep.end(); ++iter_i) {
      vector<int>::const_iterator iter_j = coreferent_nps[*set_it][*iter_i].begin();
      u = *iter_j;
      ++iter_j;
      while (iter_j != coreferent_nps[*set_it][*iter_i].end()) {
        v = *iter_j;
        /* make sure id of first node > id of second node */
        edges.push_back(make_pair(v, u));
        u = v;
        ++iter_j;
      }
    }

    LL_NODE *ptr = (LL_NODE *) malloc(sizeof(LL_NODE));
    ptr->next = NULL;
    sample.examples[i].h.head = ptr;
    for (vector<pair<int, int> >::const_iterator iter = edges.begin(); iter != edges.end(); ++iter) {
      ptr->next = (LL_NODE *) malloc(sizeof(LL_NODE));
      ptr = ptr->next;
      /* make sure u>v */
      ptr->u = iter->first;
      ptr->v = iter->second;
      assert(ptr->u > ptr->v);
      ptr->next = NULL;
    }
    LL_NODE *temp = sample.examples[i].h.head;
    sample.examples[i].h.head = (sample.examples[i].h.head)->next;
    free(temp);

    i++;

  }

  return (sample);

}



/* helper functions copied from Tom's code */

/* --------------- EVALUATION FUNCTIONS ----------------- */

inline unsigned int cardinalityP(const TOM_LABEL &y, const TOM_LABEL &ybar,
                                 unsigned int &numClusters) {
  static bitset<1024> enc;
  unsigned int p = 0, other;
  numClusters = 0;
  for (unsigned int i = 0; i < (unsigned) y.number_of_nps; i++) {
    if (i != y.first_member[i]) continue; // Not the "start" of a cluster.
    ++numClusters;
    enc.reset(); // Reset the bitset to all zero bits.
    // Iterate through the cluster.  Handle first as special case.
    ++p;
    enc.set(ybar.first_member[i]); // Mark off corresponding cluster.
    // Handle rest of the NPs in this cluster.
    for (unsigned int n = y.next_member[i]; n != i; n = y.next_member[n]) {
      other = ybar.first_member[n]; // Get the first NP of other cluster.
      if (enc.test(other)) continue; // Already seen him?  Move on.
      enc.set(other); // Mark that this cluster is now seen. 
      ++p; // Increase number of clusters within this one cluster.
    }
  }
  return p;
}

double helper_vilain(TOM_LABEL y, TOM_LABEL ybar, STRUCT_LEARN_PARM *param) {
  unsigned int numClusters, p, pp;
  /* Note, Vilain's formula has division by 0 errors in the event that
     the "target" clustering is composed of singleton clusters.  I
     correspondingly treat the "singleton" case as a special case, and
     consider that 1.0 since in that event we don't need to add any
     edges to get the target clustering. */
  p = cardinalityP(y, ybar, numClusters);
  double recall = (unsigned) y.number_of_nps == numClusters ? 1.0 :
                  ((double) (y.number_of_nps - p)) /
                  ((double) (y.number_of_nps - numClusters));
  pp = cardinalityP(ybar, y, numClusters);
  double prec = (unsigned) y.number_of_nps == numClusters ? 1.0 :
                ((double) (y.number_of_nps - pp)) /
                ((double) (y.number_of_nps - numClusters));
  cout << "RECALL IS " << recall << " PRECISION IS " << prec << endl;
  if (recall == 0.0 && prec == 0.0) return 0.0;
  double f1 = 2.0 * (recall * prec) / (recall + prec);
  return f1;
}

double helper_joachims(TOM_LABEL y, TOM_LABEL ybar, STRUCT_LEARN_PARM *param) {
  /** This counts the number of links that two clusters have in
      common. */
  int i, j, count = 0, denom = (y.number_of_nps * (y.number_of_nps - 1)) / 2;
  double loss;
  if (ybar.relax) {
    double dcount = 0.0;
    // The relaxation is present in ybar.
    for (i = 1; i < y.number_of_nps; ++i)
      for (j = 0; j < i; ++j)
        if (y.first_member[i] == y.first_member[j])
          dcount += ybar.relax[i][j];
        else
          dcount += (1.0 - ybar.relax[i][j]);
    loss = dcount / ((double) denom);
  } else {
    // The relaxation is not present in ybar.
    for (i = 1; i < y.number_of_nps; ++i) {
      for (j = 0; j < i; ++j) {
        if (y.first_member[i] == y.first_member[j]) {
          if (ybar.first_member[i] != ybar.first_member[j])
            count++;
        } else {
          if (ybar.first_member[i] == ybar.first_member[j])
            count++;
        }
      }
    }
    loss = ((double) count) / ((double) denom);
  }
  return 1.0 - loss;
}

inline unsigned int subpurity(const TOM_LABEL &y, const TOM_LABEL &ybar) {
  unsigned int counts[1024];
  unsigned int other, max, p = 0;
  for (unsigned int i = 0; i < (unsigned) y.number_of_nps; i++) {
    if (i != y.first_member[i]) continue; // Not the "start" of a cluster.
    memset(counts, 0, 1024 * sizeof(unsigned int));
    // Iterate through the cluster.  Handle first as special case.
    max = 1;
    counts[ybar.first_member[i]]++;
    // Handle rest of the NPs in this cluster.
    for (unsigned int n = y.next_member[i]; n != i; n = y.next_member[n]) {
      other = ybar.first_member[n]; // Get the first NP of other cluster.
      counts[other]++;
      if (max < counts[other])
        max = counts[other];
    }
    p += max;
  }
  return p;
}

double helper_purity(TOM_LABEL y, TOM_LABEL ybar, STRUCT_LEARN_PARM *param) {
  unsigned int p, pp;
  /* Note, Vilain's formula has division by 0 errors in the event that
     the "target" clustering is composed of singleton clusters.  I
     correspondingly treat the "singleton" case as a special case, and
     consider that 1.0 since in that event we don't need to add any
     edges to get the target clustering. */
  p = subpurity(y, ybar);
  double recall = ((double) p) / ((double) y.number_of_nps);
  pp = subpurity(ybar, y);
  double prec = ((double) pp) / ((double) y.number_of_nps);
  //cout << "RECALL IS " << recall << " PRECISION IS " << prec << endl;
  double f1 = 2.0 * (recall * prec) / (recall + prec);
  return f1;
}


TOM_LABEL label2tomlabel(LABEL y) {
  TOM_LABEL ty;

  ty.number_of_nps = y.num_nps;
  ty.number_of_clusters = y.num_clusters;

  map<int, vector<int> > cluster_members;

  for (int i = 0; i < y.num_nps; i++) {
    cluster_members[y.cluster_id[i]].push_back(i);
  }

  ty.next_member = (unsigned short *) malloc(sizeof(unsigned short) * y.num_nps);
  ty.first_member = (unsigned short *) malloc(sizeof(unsigned short) * y.num_nps);
  ty.number = (unsigned short *) malloc(sizeof(unsigned short) * y.num_nps);

  for (int i = 0; i < y.num_nps; i++) {
    ty.first_member[i] = y.cluster_id[i];
    ty.number[i] = cluster_members[y.cluster_id[i]].size();
  }

  for (map<int, vector<int> >::const_iterator map_it = cluster_members.begin();
       map_it != cluster_members.end(); ++map_it) {
    vector<int>::const_iterator iter = (map_it->second).begin();
    int last_element = *iter;
    int first_element = *iter;
    ++iter; // assume no empty cluster
    while (iter != (map_it->second).end()) {
      ty.next_member[last_element] = *iter;
      last_element = *iter;
      ++iter;
    }
    ty.next_member[last_element] = first_element;
  }
  ty.relax = NULL;

  return (ty);

}
