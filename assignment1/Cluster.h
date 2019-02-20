#ifndef CLUSTER_H
#define CLUSTER_H

#include <set>
#include <vector>
#include <iostream>

class ClusterMember;
typedef std::shared_ptr<ClusterMember> CmPtr;

struct CmComp{
    bool operator() (const CmPtr& l, const CmPtr& r) const;
};

class Cluster {
    public:        
        Cluster();
        Cluster(CmPtr cm, const int id);
        void insert(CmPtr member);
        std::vector<double> getCenter() const;
        void removeMember(CmPtr member);
        const int getId() const;
        const std::set<CmPtr, CmComp> getMembers() const;
        bool operator< (const Cluster &right) const;

    private:
        const int clusterId;
        std::vector<double> center;
        std::vector<double> sum;
        std::set<CmPtr, CmComp> members;
        void update_center(const std::vector<double> &v);
};

typedef std::shared_ptr<Cluster> ClusterPtr;
struct ClusterComp{
    bool operator() (const ClusterPtr& l, const ClusterPtr& r) const;
};

std::ostream& operator<<(std::ostream& os, const Cluster& obj);

#endif /* CLUSTER_H */