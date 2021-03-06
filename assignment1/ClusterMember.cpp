#include "ClusterMember.h"
#include "Cluster.h"
using namespace std;

ClusterMember::ClusterMember(string word, vector<double> vec, int id) : word(word), vec(move(vec)), memId(id),
clusterSim(-2), cluster(make_shared<Cluster>(Cluster()))
{
}

const vector<double>& ClusterMember::getVector() const {
    return vec;
}

const string ClusterMember::getWord() const {
    return word;
}

const double ClusterMember::getClusterSim() const {
    return clusterSim;
}

void ClusterMember::setClusterSim(double sim) {
    this->clusterSim = sim;
}

void ClusterMember::setCluster(ClusterPtr theCluster) {
    cluster = theCluster;
}

int ClusterMember::getId() const {
    return memId;
}

ClusterPtr ClusterMember::getCluster() const {
    return cluster;
}

bool ClusterMember::operator< (const ClusterMember &right) const {
    return memId < right.getId();
}

bool CmComp::operator() (const CmPtr& l, const CmPtr& r) const {
    return *l < *r;
}

ostream& operator<<(ostream& os, const ClusterMember& obj) {
    os << obj.getWord() << "\n";
    return os;
}
