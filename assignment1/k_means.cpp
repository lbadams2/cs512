#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <iterator>
#include <set>
using namespace std;

const int k = 75;

class ClusterMember {
    public:
        ClusterMember(string word, vector<double> &vec) : word(word), vec(vec), clusterSim(-2)
        {
        }

        const vector<double>& getVector() const {
            return vec;
        }

        bool operator< (const ClusterMember &right) const {
            return word < right.word;
        }

        const string getWord() const {
            return word;
        }

        double getClusterSim() const {
            return clusterSim;
        }

        void setCluster(Cluster &theCluster) {
            cluster = theCluster;
        }

    private:
        double clusterSim;
        const string word;
        const vector<double>& vec;
        shared_ptr<Cluster> cluster;
};

class Cluster {
    public:
        Cluster(){}

        Cluster(ClusterMember &cm, int id) {
            clusterId = id;
            center = cm.getVector();
            sum = center;
            insert(cm);
        }

        void insert(ClusterMember &member) {
            members.insert(member);
            this->update_center(member.getVector());
        }

        vector<double> getCenter() const {
            return this->center;
        }

        void removeMember(ClusterMember &member) {
            set<ClusterMember>::iterator iter = members.find(member);
            if(iter != members.end())
                members.erase(iter);
        }

        int getId() const {
            return clusterId;
        }

        bool operator< (const Cluster &right) const {
            return clusterId < right.getId();
        }

    private:
        int clusterId;
        vector<double> center;
        vector<double> sum;
        set<ClusterMember> members;

        void update_center(vector<double> &v) {
            vector<double> result;
            transform(sum.begin(), sum.end(), v.begin(), back_inserter(result), plus<double>());
            sum = result;
            vector<double> mean;
            int size = members.size();
            transform(sum.begin(), sum.end(), mean.begin(), bind(divides<double>(), placeholders::_1, size));
            center = mean;
        }
};

double cosine_similarity(vector<double> &center, vector<double> &v) {
    double dotProduct = inner_product(begin(center), end(center), begin(v), 0.0);
    double normCenter = sqrt(inner_product(begin(center), end(center), begin(center), 0.0 ));
    double normV = sqrt(inner_product(begin(v), end(v), begin(v), 0.0 ));
    double cosineSim = dotProduct/(normCenter * normV);
    if(cosineSim < -1 || cosineSim > 1)
        throw invalid_argument("cosine sim out of range");
    return cosineSim;
}

const set<ClusterMember> read_file(char* fileName, int dims) {
    set<ClusterMember> clusterMembers;
    string word;
    double vec[dims];
    
    FILE *fp;
    fp = fopen(fileName, "r");
    while(true) {
        int r = fscanf(fp, "%s %d\n", &word, vec);
        if( r == EOF ) 
           break;
        vector<double> v;
        v.assign(vec, vec + dims);
        ClusterMember cm(word, v);
        clusterMembers.insert(cm);
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

const set<Cluster> init_clusters(set<ClusterMember> members) {
    set<Cluster> centers;
    vector<int> generatedInts;
    int size = members.size();
    int interval = size/k;
    set<ClusterMember>::iterator iter = members.begin();
    for(int i = 0; i*interval < size; i++){
        advance(iter, interval);
        ClusterMember test = *iter;
        Cluster c(test, i);
        centers.insert(c);
    }
    /*
    auto item = wordVecs.begin();
    int i = 0;
    while(i < k-1) {
        int randNum = random_0_to_n(wordVecs.size());
        if(find(generatedInts.begin(), generatedInts.end(), randNum) == generatedInts.end()) {
            generatedInts.push_back(randNum);
            i++;
            advance(item, randNum);
        }
    }
    */
    return centers;
}

void create_clusters(set<ClusterMember> members) {
    set<Cluster> centers = init_clusters(members);
    set<ClusterMember>::iterator it;
    bool clusterChange = false;
    for(it = members.begin(); it != members.end(); it++){
        vector<double> currentVec = it->getVector();
        double maxSim = it->getClusterSim(); // cosine similarity is [-1, 1]
        Cluster simCluster;
        for(Cluster &center: centers) {
            vector<double> vecCenter = center.getCenter();
            double sim = cosine_similarity(vecCenter, currentVec);
            if(sim > maxSim) {
                maxSim = sim;
                simCluster = center;
            }
        }
        // assign to new cluster
        if(maxSim != it->getClusterSim()) {
            clusterChange = true;

        }
    }
}

int main(int argc, char** argv) {    
    set<ClusterMember> members = read_file(argv[1], 100);
    create_clusters(members);
    return 0;
}