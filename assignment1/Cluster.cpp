#include <cmath>
#include <numeric>
#include "Cluster.h"
#include "ClusterMember.h"
using namespace std;

Cluster::Cluster(): clusterId(-1){}

Cluster::Cluster(CmPtr cm, const int id) : clusterId(id), center(100), sum(100) {
    center = cm->getVector();
    insert(cm);
}

const double Cluster::cosineSimilarity(const vector<double> &v) {
    double dotProduct = inner_product(begin(center), end(center), begin(v), 0.0);
    double normCenter = sqrt(inner_product(begin(center), end(center), begin(center), 0.0 ));
    double normV = sqrt(inner_product(begin(v), end(v), begin(v), 0.0 ));
    double cosineSim = dotProduct/(normCenter * normV);
    if(cosineSim < -1.01 || cosineSim > 1.01)
        throw invalid_argument("cosine sim out of range");
    return cosineSim;
}

void Cluster::insert(CmPtr member) {
    members.insert(member);
    this->update_center(member->getVector());
    double cosineSim = this->cosineSimilarity(member->getVector());
    member->setClusterSim(cosineSim);
}

vector<double> Cluster::getCenter() const {
    return this->center;
}

void Cluster::removeMember(CmPtr member) {
    set<CmPtr, CmComp>::iterator iter = members.find(member);
    if(iter != members.end())
        members.erase(iter);
}

const int Cluster::getId() const {
    return clusterId;
}

const set<CmPtr, CmComp> Cluster::getMembers() const {
    return members;
}

bool Cluster::operator< (const Cluster &right) const {
    return clusterId < right.getId();
}

void Cluster::update_center(const vector<double> &v) {
    vector<double> result;
    transform(sum.begin(), sum.end(), v.begin(), back_inserter(result), plus<double>());
    sum = result;
    vector<double> mean(100);
    int size = members.size();
    transform(sum.begin(), sum.end(), mean.begin(), bind(divides<double>(), placeholders::_1, size));
    center = mean;
}

bool ClusterComp::operator() (const ClusterPtr& l, const ClusterPtr& r) const {
    return *l < *r;
}

ostream& operator<<(ostream& os, const Cluster& obj) {
    for(const CmPtr& cm: obj.getMembers())
        os << *cm;
    return os;
}
