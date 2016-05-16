Code for Noun Phrase Coreference, implemented using the Latent Structural SVM (v0.12, Feb 8 2010)
--------------------------------------------
(implemented by Chun-Nam Yu based on the SVM-light code by Thorsten Joachims. The code also employs a disjoint set implementation by Emil Stefanov at http://www.emilstefanov.net/Projects/DisjointSets.aspx)


INSTALLATION
------------
1. Type 'make' in the directory latentnpcoref and make sure that the compilation is successful.


TRAINING
--------
1. To train a model with training set 'example.train' and output the model to 'example.model', type the command:
	./svm_coref_learn -c REGULARIZATION_CONSTANT --k COST_FACTOR muc6-october.train example.model

2. The following command line options are available during training:
   -c : specify the regularization constant
   --k: specify the relative cost of adding an incorrect tree edge compared to failing to include a correct tree edge (DEFAULT=1.0)

3. Input file format: 
   - the input file format is similar to the SVM^light input file format for classification. Each line represent an edge in the complete graph in the noun phrase coreference problem. Label '1' indicates that the edge connects two coreferent noun phrases, while label '-1' indicates that the edge connects two non-coreferent noun phrases. The label is followed by a set of edge features in sparse vector format. 
   - the last three number after the comment symbol '#' represents the 'document id', 'noun phrase id for first noun phrase', and 'noun phrase id for second noun phrase' respectively. It is assumed that the noun phrase id for the first noun phrase has to be smaller than the noun phrase id for the second noun phrase. 
   - For example, the line: 
'-1 3:1 5:1 6:-10.0 9:1 10:-10.0 12:1 14:1 15:1 19:1 21:1 25:1 27:1 31:1 34:1 36:1 39:1 40:1 42:1 44:1 46:1 48:1 51:1 52:1 56:1 60:1 69:1 73:1 75:1 77:1 79:1 82:1 83:1 88:1 90:1 92:1 94:1 96:1 97:1 100:1 101:1 103:1 107:1 109:1 110:1 112:1 114:1 116:1 118:1 120:1 122:1 124:20.0 125:20.0 127:1 129:1 131:1 132:1 # 4 52 53'
	represents an edge connecting two non-coreferent noun phrases (with id 52 and 53) in document 4, with features in sparse vector format '3:1 ... 132:1'. 

TESTING
-------
1. To predict on a test set 'example.test' with a learned model 'example.model', type the command:
	./svm_coref_classify muc6-october.test example.model


CONTACT
-------
If you had any suggestions to the program or have bugs to report, you can email Chun-Nam Yu at cnyu@cs.cornell.edu.  


REFERENCES
----------
[1] C.-N. Yu and T. Joachims: Learning Structural SVMs with Latent Variables, ICML 2009 
[2] I. Tsochantaridis, T. Hofmann, T. Joachims, and Y. Altun: Support Vector Learning for Interdependent and Structured Output Spaces, ICML 2004
