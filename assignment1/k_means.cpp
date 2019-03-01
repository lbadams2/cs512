#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iterator>
#include <set>
#include "Cluster.h"
#include "ClusterMember.h"
using namespace std;

const int k = 75;

const set<CmPtr, CmComp> read_file(string fileName, int dims) {
    set<CmPtr, CmComp> clusterMembers;
    ifstream infile(fileName);
    string line;
    int i = 0;
    while(getline(infile, line)) {
        stringstream ss(line);
        string item;
        int splitNum = 0;
        string currentWord;
        vector<double> v;
        while(getline(ss, item, ' ')) {
            if (splitNum == 0) {
                splitNum++;
                currentWord = item;
                continue;
            }
            double val = stod(item);
            v.push_back(val);
        }
        CmPtr cmptr = make_shared<ClusterMember>(currentWord, v, i++);
        clusterMembers.insert(cmptr);
    }
    return clusterMembers;
}

const set<ClusterPtr, ClusterComp> init_clusters(set<CmPtr, CmComp> members) {
    set<ClusterPtr, ClusterComp> centers;
    vector<int> generatedInts;
    int size = members.size();
    int interval = size/k;
    set<CmPtr, CmComp>::iterator iter = members.begin();
    ClusterPtr cptr = make_shared<Cluster>(*iter, 0);
    centers.insert(cptr);
    for(int i = 1; i*interval < size; i++){
        advance(iter, interval);
        cptr = make_shared<Cluster>(*iter, i*interval);
        centers.insert(cptr);
    }
    return centers;
}

set<ClusterPtr, ClusterComp> create_clusters(set<CmPtr, CmComp> members) {
    set<ClusterPtr, ClusterComp> centers = init_clusters(members);
    set<CmPtr, CmComp>::iterator it;
    int maxIterations = 250;
    int i = 0;
    while(true && i < maxIterations) {
        i++;
        //cout << i << "\n";
        bool clusterChange = false;
        for(it = members.begin(); it != members.end(); it++){
            vector<double> currentVec = (*it)->getVector();
            double maxSim = (*it)->getClusterSim(); // cosine similarity is [-1, 1]
            ClusterPtr simCluster;
            for(ClusterPtr center: centers) {
                double sim = center->cosineSimilarity(currentVec);
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
    clock_t begin = clock();
    set<CmPtr, CmComp> members = read_file("/Users/liamadams/Documents/school/cs512/assignment1/YELP_auto_phrase.emb", 100);
    cout << "Done reading file\n";
    set<ClusterPtr, ClusterComp> clusters = create_clusters(members);
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "Runtime: " << elapsed_secs << "\n\n";
    for(ClusterPtr c: clusters)
        cout << *c << "\n\n";
    return 0;
}
