#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
using namespace std;

vector<vector<int>> data;

// Argument:
//  + argv[1]   =   file_input (char*, e.g. "matrix.txt")
int main(int argc, char **argv){
    // Collect program arguments
    if (argc < 2)
        exit(0);
    const string file_input(argv[1]);
    int USERS = 943;
    int ITEMS = 1682;

    // Init mat space
    data.resize(USERS);
    for(int i=0; i<data.size();i++)
        data[i] = vector<int>(ITEMS, 0);

    // Read data
    ifstream data_file_train(file_input, ios::in);
    if (data_file_train.is_open()) {
        int usr_id, item_id, rating, timestep;
        while(data_file_train >> usr_id >> item_id >> rating >> timestep){
            data[usr_id-1][item_id-1] = rating;
        }
    }
    data_file_train.close();

    // Declare output file name
    std::size_t found = file_input.find_last_of("/\\");
    string _path_ = file_input.substr(0,found);
    string _file_ = file_input.substr(found+1);
    string file_output = _path_ + "/sparse_" + _file_;

    // Write data
    ofstream export_file(file_output, ios::out);
    export_file << USERS << " " << ITEMS << endl;
    if (export_file.is_open()) {
        for (auto row : data) {
            for (auto v : row) 
                export_file << v << "  ";
            export_file << endl;
        }
    }
    export_file.close();

    return 0;
}


