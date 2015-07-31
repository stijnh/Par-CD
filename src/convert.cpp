#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

#define MAGIC_WORD ("PCD")

void read_edges(char *filename, int &num_nodes, vector<pair<int, int> > &edges) {
    ifstream input(filename);
    char line[1024];
    num_nodes = 0;

    // read input
    for (int i = 1; input.good(); i++) {
        int idx = 0;
        input.getline(line, sizeof(line));

        while (isspace(line[idx])) {
            idx++;
        }

        if (line[idx] == '\0' || line[idx] == '#') {
            continue;
        }

        if (!isdigit(line[idx])) {
            cerr << "line " << i << " is corrupt" << endl;
            exit(-1);
        }

        int a = 0;

        while (isdigit(line[idx])) {
            a = 10 * a + (line[idx] - '0');
            idx++;
        }

        while (isspace(line[idx])) {
            idx++;
        }

        if (!isdigit(line[idx])) {
            cerr << "line " << i << " is corrupt" << endl;
            exit(-1);
        }

        int b = 0;

        while (isdigit(line[idx])) {
            b = 10 * b + (line[idx] - '0');
            idx++;
        }

        edges.push_back(make_pair(a, b));
        num_nodes = max(max(a, b) + 1, num_nodes);

        if (edges.size() % 1000000 == 0) {
            cout << (edges.size() / 2) << " lines read so far" << endl;
        }
    }

    if (!input.eof() && input.fail()) {
        cerr << "an error occured while reading input file" << endl;
        exit(-1);
    }

    input.close();
}

void convert_to_csr(int num_nodes, vector<pair<int, int> > edges, int *&indices, int *&adj) {
    int *deg = new int[num_nodes];
    indices = new int[num_nodes + 1];
    adj = new int[2 * edges.size()];

    fill(deg, deg + num_nodes, 0);

    for (int i = 0; i < int(edges.size()); i++) {
        deg[edges[i].first]++;
        deg[edges[i].second]++;
    }

    indices[0] = 0;
    for (int i = 0; i < num_nodes; i++) {
        indices[i + 1] = indices[i] + deg[i];
        deg[i] = 0;
    }

    for (int i = 0; i < int(edges.size()); i++) {
        int a = edges[i].first;
        int b = edges[i].second;

        adj[indices[a] + deg[a]] = b;
        adj[indices[b] + deg[b]] = a;

        deg[a]++;
        deg[b]++;
    }

    delete[] deg;
}


void remove_duplicates_and_loops(int &num_nodes, int *indices, int *adj) {
    int index = 0;

    for (int i = 0; i < num_nodes; i++) {
        int start = indices[i];
        int end = indices[i + 1];

        indices[i] = index;

        sort(adj + start, adj + end);
        int prev_v = -1;

        for (int j = start; j < end; j++) {
            int v = adj[j];

            if (v != i && v != prev_v) {
                adj[index++] = v;
            }

            prev_v = v;
        }
    }

    indices[num_nodes] = index;
}


void sort_on_degree(int num_nodes, int *&indices, int *&adj, int *&rmap) {
    pair<int, int> *degree_ids = new pair<int, int>[num_nodes];
    int *map = new int[num_nodes];
    rmap = new int[num_nodes];

    int new_index = 0;
    int num_endpoints = indices[num_nodes];
    int *new_indices = new int[num_nodes + 1];
    int *new_adj = new int[num_endpoints];

    for (int i = 0; i < num_nodes; i++) {
        degree_ids[i] = make_pair(indices[i + 1] - indices[i], i);
    }

    sort(degree_ids, degree_ids + num_nodes, greater<pair<int, int> >());

    for (int i = 0; i < num_nodes; i++) {
        int v = degree_ids[i].second;
        map[v] = i;
        rmap[i] = v;
    }

    for (int i = 0; i < num_nodes; i++) {
        int v = rmap[i];

        new_indices[i] = new_index;

        for (int j = indices[v]; j < indices[v + 1]; j++) {
            new_adj[new_index++] = map[adj[j]];
        }
    }

    new_indices[num_nodes] = new_index;

    for (int i = 0; i < num_nodes; i++) {
        sort(new_adj + new_indices[i], new_adj + new_indices[i + 1]);
    }

    delete[] indices;
    delete[] adj;

    indices = new_indices;
    adj = new_adj;

    delete[] degree_ids;
    delete[] map;
}

void remove_isolated_vertices(int &num_nodes, int *&indices, int *&adj) {

    // since vertices are sorted by degree, we simply look for the first
    // vertex having degree zero.
    for (int i = 0; i < num_nodes; i++) {
        int deg = indices[i + 1] - indices[i];

        if (deg == 0) {
            num_nodes = i;
            break;
        }
    }
}

void count_triangles(int &num_nodes, int *indices, int *adj, int *&tri, int *&tri_deg) {
    tri = new int[num_nodes];
    tri_deg = new int[num_nodes];

    for (int p = 0; p < num_nodes; p++) {
        int t = 0;
        int td = 0;

        for (int j = indices[p]; j < indices[p + 1]; j++) {
            int q = adj[j];

            int a = indices[p];
            int a_end = indices[p + 1];

            int b = indices[q];
            int b_end = indices[q + 1];

            bool found = false;

            while (a < a_end && b < b_end) {
                int d = adj[a] - adj[b];

                if (d == 0) {
                    found = true;
                    t++;
                }

                if (d <= 0) {
                    a++;
                }

                if (d >= 0) {
                    b++;
                }
            }

            if (found) {
                td++;
            }
        }

        tri[p] = t;
        tri_deg[p] = td;
    }
}

void write_edges(char *filename, int num_nodes, int *indices, int *adj, int *rmap) {
    int num_edges = indices[num_nodes] / 2;

    ofstream output(filename, ios::binary);
    output.write((char*) MAGIC_WORD, sizeof(char) * strlen(MAGIC_WORD));

    output.write((char*) &num_nodes, sizeof(int));
    output.write((char*) &num_edges, sizeof(int));

    output.write((char*) indices, sizeof(int) * (num_nodes + 1));
    output.write((char*) adj, sizeof(int) * (2 * num_edges));
    output.write((char*) rmap, sizeof(int) * num_nodes);

    cout << num_nodes << " " << num_edges << endl;

    for (int i = 0; i < num_nodes; i++) {
        cout << i << "] " << indices[i] << " " << indices[i + 1] << " | ";

        for (int j = indices[i]; j < indices[i + 1]; j++) {
            cout << adj[j] << " ";
        }

        cout << endl;
    }

    //output.write((char*) triangles, sizeof(int) * num_nodes);
    //output.write((char*) triangle_degs, sizeof(int) * num_nodes);

    if (output.fail()) {
        cerr << "an error occured while writing output file" << endl;
        exit(-1);
    }

    output.close();
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " <input file> <output file>" << endl;
        return -1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    vector<pair<int, int> > edges;
    int num_nodes;
    int *rmap;
    int *indices, *adj;

    // read input file
    cout << "reading graph from file " << input_file << endl;
    read_edges(input_file, num_nodes, edges);
    cout << "found " << num_nodes << " nodes and " << edges.size()
        << " edges" << endl;

    // convert to CSR
    cout << "converting to CSR format" << endl;
    convert_to_csr(num_nodes, edges, indices, adj);
    cout << "done " << endl;

    // remove duplicates and loops
    cout << "removing duplicate edges and loops" << endl;
    int old_num_edges = indices[num_nodes] / 2;
    remove_duplicates_and_loops(num_nodes, indices, adj);
    int new_num_edges = indices[num_nodes] / 2;
    cout << "removed " << (old_num_edges - new_num_edges) << " edges" << endl;

    // sort on degree
    cout << "sort nodes according to degree" << endl;
    sort_on_degree(num_nodes, indices, adj, rmap);
    cout << "done " << endl;

    // remove isolated vertices
    cout << "remove isolated vertices" << endl;
    int old_num_nodes = num_nodes;
    remove_isolated_vertices(num_nodes, indices, adj);
    cout << "removed " << (old_num_nodes - num_nodes) << " vertices" << endl;

    // count triangles
    //int *tri, *tri_deg;
    //cout << "count number of triangles" << endl;
    //count_triangles(num_nodes, edges, tri, tri_deg);

    // write to file
    cout << "final graph has " << num_nodes << " nodes and "
        << (edges.size() / 2) << " edges" << endl;
    cout << "aaaa" << indices[num_nodes] << endl;
    cout << "writing final graph to " << "output_file" << endl;
    cout << "aaaa" << indices[num_nodes] << endl;
    write_edges(output_file, num_nodes, indices, adj, /*tri, tri_deg,*/ rmap);

    return 0;
}

