//
// @file    : main.cpp
// @purpose : A demo of NOMAD with UPCXX
// @author  : Hung Ngoc Phan
// @project : NOMAD algorithm for matrix completion with UPCXX
// @licensed: N/A
// @created : 03/07/2020
// @modified: 09/07/2020
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <cmath>
#include <ctime>
#include "worker.h"
#include <upcxx/upcxx.hpp>
#define bug(x) cout << #x << " = " << x << endl
using namespace std;

void                    read_data(const string file_input, int &NROW, int &NCOL, 
                                  vector<vector<double>> &arr_data, vector<int> &row_count);
void                    write_data(const string file_output, vector<vector<double>> &arr_data);
void                    assert_matrix_size(vector<vector<double>> &mat, int nRows, int nCols);
vector<vector<int>>     split_array_index(const vector<int> &arr, int num_segment);


// Argument:
//  + argv[0]   =   file_input
//  + argv[1]   =   NUM_EPOCHS
int main() {
    // Predefined params for sparse matrix input
    const string file_input = "matrix3.txt";
    int NROW, NCOL;
    vector<vector<double>> mat_data;
    vector<int> num_element_row;

    // Read input matrix as data
    read_data(file_input, NROW, NCOL, mat_data, num_element_row);
    assert_matrix_size(mat_data, NROW, NCOL);

    // Define matrix completion kernel: K = max(1, dim/6)
    int K_embeddings = max(1, (int) ((0.5 * (NROW + NCOL)) / 3.0));


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MAIN PROCESS  --  Starts from here
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Premilinary definition
    upcxx::init();
    int num_proc = upcxx::rank_n();
    int proc_id = upcxx::rank_me();

    // Split rows of W into 'num_worker' parts using a simple naive approach
    vector<vector<int>> split_row_index = split_array_index(num_element_row, num_proc);

    // Store usr_idx and corresponding rows in W, A in each process.
    vector<vector<double>> segments_A;
    for (auto user_index : split_row_index[proc_id])
        segments_A.push_back(mat_data[user_index]);
    assert_matrix_size(segments_A, split_row_index[proc_id].size(), NCOL);

    // Initialize worker object as upcxx::dist_object
    double alpha_rate = 0.012;
    double beta_rate = 0.01;
    double lambda_rate = 0.05;
    upcxx::dist_object<Worker> worker(Worker(proc_id, NROW,
                                             NCOL, K_embeddings,
                                             alpha_rate, beta_rate, lambda_rate,
                                             split_row_index[proc_id], segments_A));

    // Initialize item queue of each worker randomly
    std::default_random_engine generator(time(NULL));
    std::uniform_int_distribution<int> distribution(0, num_proc - 1);
    for (int i = 0; i < NCOL; i++) {
        int receiver_id = distribution(generator);
        if (upcxx::rank_me() == receiver_id){
            worker->add_item_idx_to_queue(i);
        }
        upcxx::barrier();
    }

    // Print to test the distributing procedure
    // for (int i = 0; i < num_proc; i++) {
    //     if (upcxx::rank_me() == i) {
    //         worker->print_debug_matrix(true, true, true);
    //         // worker->print_debug_queue();
    //     }
    //     upcxx::barrier();
    // }

    //////////////////////////
    // Model update
    //////////////////////////
    long long int NUM_EPOCHS = 700;
    for (long long int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        if(proc_id == 0) printf("-----| Epoch #%lld\n",epoch);
        worker->update(epoch + 1);
    }
    upcxx::barrier();

    // Print to test the distributing procedure
    // for (int i = 0; i < num_proc; i++) {
    //     if (upcxx::rank_me() == i) {
    //         printf("\nAfter update %lld epoch\n", NUM_EPOCHS);
    //         worker->print_debug_matrix(true, true, true);
    //         // worker->print_debug_queue();
    //     }
    //     upcxx::barrier();
    // }

    upcxx::finalize();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // MAIN PROCESS  --  Ends at here
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    return 0;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Subsidiary functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// @brief: Read the input data into pass matrix
//
void read_data(const string file_input, int &NROW, int &NCOL, 
               vector<vector<double>> &arr_data, vector<int> &row_count) {
    // Init file stream
    ifstream data_file(file_input, ios::in);

    // Read data
    if (data_file.is_open()) {
        data_file >> NROW >> NCOL;
        arr_data.resize(NROW);
        row_count = vector<int>(NROW, 0);

        for (int i = 0; i < NROW; i++) {
            for (int j = 0; j < NCOL; j++) {
                double v;
                data_file >> v;
                arr_data[i].push_back(v);
                if (v != 0)
                    row_count[i]++;
            }
        }
    }

    data_file.close();
}

//
// @brief: Write output matrix into export file
//
void write_data(const string file_output, vector<vector<double>> &arr_data){
    // Init file stream
    ofstream export_file(file_output, ios::out);

    // Read data
    if (export_file.is_open()) {
        for(auto row : arr_data){
            for(auto v : row){
                cout << fixed << setprecision(5) << v << "  ";
            }
            export_file << endl;
        }
    }

    export_file.close();
}

//
// @brief: Read the input data into pass matrix
//
void assert_matrix_size(vector<vector<double>> &mat, int nRows, int nCols) {
    assert(mat.size() == nRows);
    for (auto row : mat)
        assert(row.size() == nCols);
}

//
// @brief: Split the input array into K least-size-different parts
//
vector<vector<int>> split_array_index(const vector<int> &arr, int num_segment) {
    assert(num_segment <= (int)arr.size());

    vector<pair<int, int>> input_arr;
    for (int i = 0; i < arr.size(); i++) {
        input_arr.push_back(make_pair(arr[i], i));
    }

    sort(input_arr.begin(), input_arr.end(), greater<pair<int, int>>());

    vector<vector<int>> ans(num_segment);
    vector<int> sum(num_segment, 0);

    for (auto v : input_arr) {
        int min_val = *min_element(sum.begin(), sum.end());
        vector<int>::iterator it = find(sum.begin(), sum.end(), min_val);
        int min_pos = it - sum.begin();

        sum[min_pos] += v.first;
        ans[min_pos].push_back(v.second);
    }

    return ans;
}
