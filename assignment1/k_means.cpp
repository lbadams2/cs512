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
using namespace std;

const int k = 75;

struct Cluster {
    vector<double> center;
    vector<pair<string, vector<double> > > vectors;
    vector<double> sum;

    Cluster(string word, vector<double> &v) {
        pair<string, vector<double> > p(word, v);
        this->insert(p);
    }

    void insert(pair<string, vector<double> > v) {
        vectors.push_back(v);
        this->update_center(v.second);
    }

    void update_center(vector<double> v) {
        if(vectors.size() == 1) {
            center = v;
            sum = center;
        }
        else {
            vector<double> result;
            transform(sum.begin(), sum.end(), v.begin(), back_inserter(result), plus<double>());
            sum = result;
            vector<double> mean;
            int size = vectors.size();
            transform(sum.begin(), sum.end(), mean.begin(), bind(divides<double>(), placeholders::_1, size));
            center = mean;
        }
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

const map<string, vector<double> > read_file() {
    map<string, vector<double> > wordVectors;
    ifstream infile("/Users/liamadams/Documents/school/cs512/assignment1/YELP_auto_phrase.emb");
    string line;
    while(getline(infile, line)) {
        stringstream ss(line);
        string item;
        int splitNum = 0;
        string currentWord;
        while(getline(ss, item, ' ')) {
            if (splitNum == 0) {
                if(wordVectors.count(item))
                    throw invalid_argument("duplicate word " + item);
                vector<double> v;
                wordVectors[item] = v;
                splitNum++;
                currentWord = item;
            }
            double val = stod(item);
            wordVectors[currentWord].push_back(val);
        }
    }
    return wordVectors;
}

const vector<Cluster> init_clusters(map<string, vector<double> > &wordVecs) {
    vector<Cluster> centers;
    vector<int> generatedInts;
    int size = wordVecs.size();
    int interval = size/k;
    map<string, vector<double> >::iterator iter = wordVecs.begin();
    for(int i = 0; i*interval < size; i++){
        advance(iter, i*interval);
        Cluster c(iter->first, iter->second);
        centers.push_back(c);
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

void create_clusters(map<string, vector<double> > wordVecs) {
    vector<Cluster> centers = init_clusters(wordVecs);
    map<string, vector<double> >::iterator it;
    for(it = wordVecs.begin(); it != wordVecs.end(); it++){
        vector<double> currentVec = it->second;
        double maxSim = -2; // cosine similarity is [-1, 1]
        vector<double> simCenter;
        for(vector<double> center: centers) {
            double sim = cosine_similarity(center, currentVec);
            if(sim > maxSim) {
                maxSim = sim;
                simCenter = center;
            }
        }
    }
}

int main(int argc, char** argv) {
    auto wordVecs = read_file();
    create_clusters(wordVecs);
    return 0;
}