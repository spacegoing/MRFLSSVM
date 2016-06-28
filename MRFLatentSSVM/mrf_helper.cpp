#include "mrf_helper.h"
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <bitset>
#include <iostream>

using namespace std;

/* modified from Tom's function */
SAMPLE read_struct_examples_helper_nlp(char *filename, STRUCT_LEARN_PARM *sparm) {

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


SAMPLE read_struct_examples_helper(STRUCT_LEARN_PARM *sparm) {

}