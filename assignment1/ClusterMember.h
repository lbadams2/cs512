#ifndef CLUSTERMEMBER_H
#define CLUSTERMEMBER_H

#include <vector>
#include <string>
#include <iostream>

class Cluster;
typedef std::shared_ptr<Cluster> ClusterPtr;

class ClusterMember{
    public:
        ClusterMember(std::string word, std::vector<double> &vec, int id);
        const std::vector<double>& getVector() const;
        const std::string getWord() const;
        double getClusterSim() const;
        void setCluster(ClusterPtr theCluster);
        int getId() const;
        ClusterPtr getCluster() const;
        bool operator< (const ClusterMember &right) const;

    private:
        const int memId;
        double clusterSim;
        const std::string word;
        const std::vector<double>& vec;
        ClusterPtr cluster;
};

typedef std::shared_ptr<ClusterMember> CmPtr;

std::ostream& operator<<(std::ostream& os, const ClusterMember& obj);

#endif /* CLUSTERMEMBER_H */