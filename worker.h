//
// @file    : worker.h
// @purpose : A definition class for Worker classes
// @author  : Hung Ngoc Phan
// @project : NOMAD algorithm for matrix completion with UPCXX 
// @licensed: N/A
// @created : 03/07/2020
// @modified: 09/07/2020
// 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef WORKER_H_
#define WORKER_H_
#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <queue>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <upcxx/upcxx.hpp>
using namespace std;

class Worker {

public: 
    // C.35: A base class destructor should be either public and virtual, or protected and nonvirtual
    // C.21: If you define or =delete any default operation, define or =delete them all
    ///////////////////////////////////////////////////////
    // Default operations
    ///////////////////////////////////////////////////////
    Worker()                                = default;

    Worker(int proc_id, int num_users,                      // User-defined constructor
           int num_items,int num_embeddings,
           double _alpha_, double _beta_, double _lambda_,
           vector<int>user_index, vector<vector<double>>A);

    Worker(const Worker& old)               = default;
    Worker& operator=(const Worker& old)    = default;
    Worker(Worker&& old)                    = default;
    Worker& operator=(Worker&& old)         = default;
    virtual ~Worker() noexcept              = default;

    ///////////////////////////////////////////////////////
    // SGD-NOMAD Model functions
    ///////////////////////////////////////////////////////
    void                    initialize_W_uniform_random();
    void                    initialize_H_uniform_random();
    void                    add_item_idx_to_queue(int item_idx);
    void                    update(int epoch_idx);
    vector<vector<double>>  compute_approximate_A();

    ///////////////////////////////////////////////////////
    // Debugging functions
    ///////////////////////////////////////////////////////
    void                    print_debug_matrix(bool print_A, bool print_W, bool print_H);
    void                    print_debug_queue();

private:
    ///////////////////////////////////////////////////////
    // Private SGD update functions
    ///////////////////////////////////////////////////////
    double                  compute_learning_rate(int time);
    void                    update_value_W_and_H(int item_index);
    int                     get_priority_process_index();
    upcxx::future<>         transfer_item(int worker_id, int item_index);

    ///////////////////////////////////////////////////////
    // Linear algebra functions
    ///////////////////////////////////////////////////////
    vector<double>          vec_scalar_add(vector<double> vec, double scalar);
    vector<double>          vec_scalar_multiply(vector<double> vec, double scalar);
    double                  vec_vec_multiply(vector<double> vec1, vector<double> vec2);
    vector<double>          vec_vec_add(vector<double> vec1, vector<double> vec2);
    vector<double>          vec_vec_subtract(vector<double> vec1, vector<double> vec2);
    double                  vec_norm_2(vector<double> vec);

    ///////////////////////////////////////////////////////
    // Member
    ///////////////////////////////////////////////////////
    int                                             proc_id         { -1 };
    int                                             num_users       { -1 };
    int                                             num_items       { -1 };
    int                                             num_embeddings  { -1 };
    double                                          _alpha_         { 0.0 };
    double                                          _beta_          { 0.0 };
    double                                          _lambda_        { 0.0 };
    unsigned                                        random_seed     { 0 };

    std::default_random_engine                      random_engine;
    std::uniform_int_distribution<int>              randomer;

    int*                                            update_step;
    upcxx::dist_object<vector<int>>                 user_index;
    vector<vector<double>>                          A;
    upcxx::dist_object<upcxx::global_ptr<double>>   W;
    upcxx::dist_object<upcxx::global_ptr<double>>   H;          // default pointed by proc-0
    upcxx::dist_object<queue<int>>                  item_queue;

};

#endif // WORKER_H_
