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

double cosine_similarity(vector<double> center, vector<double> v) {
    double dotProduct = inner_product(begin(center), end(center), begin(v), 0.0);
    double normCenter = sqrt(inner_product(begin(center), end(center), begin(center), 0.0 ));
    double normV = sqrt(inner_product(begin(v), end(v), begin(v), 0.0 ));
    double cosineSim = dotProduct/(normCenter * normV);
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

const vector<vector<double> > get_centers(map<string, vector<double> > wordVecs) {
    vector<vector<double> > centers;
    vector<int> generatedInts;
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
    return centers;
}

int main(int argc, char** argv) {
    read_file();

    return 0;
}