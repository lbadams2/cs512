k_means: cluster.o clustermember.o k_means.o
	clang++ -std=c++17 -g -o k_means.out Cluster.o ClusterMember.o k_means.o

k_means.o: Cluster.h ClusterMember.h k_means.cpp
	clang++ -std=c++17 -g -c k_means.cpp

cluster.o: Cluster.h Cluster.cpp ClusterMember.h
	clang++ -std=c++17 -g -c Cluster.cpp

clustermember.o: ClusterMember.h ClusterMember.cpp Cluster.h
	clang++ -std=c++17 -g -c ClusterMember.cpp

clean:
	rm *.o *.out