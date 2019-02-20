#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <iterator>
#include <set>
#include "Cluster.h"
#include "ClusterMember.h"
using namespace std;

const int k = 75;

double cosine_similarity(vector<double> &center, vector<double> &v) {
    double dotProduct = inner_product(begin(center), end(center), begin(v), 0.0);
    double normCenter = sqrt(inner_product(begin(center), end(center), begin(center), 0.0 ));
    double normV = sqrt(inner_product(begin(v), end(v), begin(v), 0.0 ));
    double cosineSim = dotProduct/(normCenter * normV);
    if(cosineSim < -1 || cosineSim > 1)
        throw invalid_argument("cosine sim out of range");
    return cosineSim;
}

const set<CmPtr, CmComp> read_file(char* fileName, int dims) {
    set<CmPtr, CmComp> clusterMembers;
    string word;
    double vec[dims];
    
    FILE *fp;
    fp = fopen(fileName, "r");
    int i = 0;
    while(true) {
        int r = fscanf(fp, "%s %d\n", &word, vec);
        if( r == EOF ) 
           break;
        vector<double> v;
        v.assign(vec, vec + dims);
        CmPtr cmptr = make_shared<ClusterMember>(word, v, i++);
        clusterMembers.insert(cmptr);
    }
    fclose(fp);
    /*
    ifstream infile("/Users/liamadams/Documents/school/cs512/assignment1/YELP_auto_phrase.emb");
    while(getline(infile, line)) {
        stringstream ss(line);
        string item;
        int splitNum = 0;
        string currentWord;
        ClusterMember cm;
        while(getline(ss, item, ' ')) {
            if (splitNum == 0) {
                vector<double> v;
                cm(item, v);
                splitNum++;
                currentWord = item;
            }
            double val = stod(item);
            set<ClusterMember>::iterator iter = clusterMembers.find
            wordVectors[currentWord].push_back(val);
        }
    }
    */
    return clusterMembers;
}

const set<ClusterPtr, ClusterComp> init_clusters(set<CmPtr, CmComp> members) {
    set<ClusterPtr, ClusterComp> centers;
    vector<int> generatedInts;
    int size = members.size();
    int interval = size/k;
    set<CmPtr, CmComp>::iterator iter = members.begin();
    for(int i = 0; i*interval < size; i++){
        advance(iter, interval);
        ClusterPtr cptr = make_shared<Cluster>(*iter, i);
        centers.insert(cptr);
    }
    return centers;
}

set<ClusterPtr, ClusterComp> create_clusters(set<CmPtr, CmComp> members) {
    set<ClusterPtr, ClusterComp> centers = init_clusters(members);
    set<CmPtr, CmComp>::iterator it;
    while(true) {
        bool clusterChange = false;
        for(it = members.begin(); it != members.end(); it++){
            vector<double> currentVec = (*it)->getVector();
            double maxSim = (*it)->getClusterSim(); // cosine similarity is [-1, 1]
            ClusterPtr simCluster;
            for(ClusterPtr center: centers) {
                vector<double> vecCenter = center->getCenter();
                double sim = cosine_similarity(vecCenter, currentVec);
                if(sim > maxSim) {
                    maxSim = sim;
                    simCluster = center;
                }
            }
            // assign to new cluster
            if(maxSim != (*it)->getClusterSim()) {
                clusterChange = true;
                ClusterPtr oldCluster = (*it)->getCluster();
                oldCluster->removeMember(*it);
                (*it)->setCluster(simCluster);
                simCluster->insert(*it);
            }
        }
        if(!clusterChange)
            break;
    }
    return centers;
}

int main(int argc, char** argv) {    
    set<CmPtr, CmComp> members = read_file(argv[1], 100);
    set<ClusterPtr, ClusterComp> clusters = create_clusters(members);
    for(ClusterPtr c: clusters)
        cout << c << "\n\n";
    return 0;
}