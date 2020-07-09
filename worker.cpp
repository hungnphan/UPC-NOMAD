//
// @file    : worker.cpp
// @purpose : A implementation class for Worker classes
// @author  : Hung Ngoc Phan
// @project : NOMAD algorithm for matrix completion with UPCXX
// @licensed: N/A
// @created : 03/07/2020
// @modified: 07/07/2020
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "worker.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Default operations
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
Worker::Worker(int proc_id, int num_users,
			   int num_items, int num_embeddings,
			   double _alpha_, double _beta_, double _lambda_,
			   vector<int> user_index, vector<vector<double>> A)
	: proc_id		{ proc_id },
	  num_users		{ num_users },
	  num_items		{ num_items },
	  num_embeddings{ num_embeddings },
	  _alpha_		{ _alpha_ },
	  _beta_		{ _beta_ },
	  _lambda_		{ _lambda_ },
	  update_step	{ new int[(int) (user_index.size() * num_items)] },
	  user_index	{ user_index },
	  A				{ A },
	  W				( upcxx::new_array<double>(user_index.size() * num_embeddings) ),
	  H				( upcxx::new_array<double>(num_items * num_embeddings) ),
	  item_queue	( queue<int>() ) {

	assert(proc_id != -1);
	assert(num_users > 0);
	assert(num_items > 0);
	assert(0 < num_embeddings && num_embeddings < min(num_users, num_items));

	// Inititialize random generator
	this->random_seed = std::chrono::system_clock::now().time_since_epoch().count() + 1234567890 * this->proc_id;
	this->random_engine = std::default_random_engine(this->random_seed);
	this->randomer = std::uniform_int_distribution<int>(0, upcxx::rank_n() - 1);

	// Initialize kernel W in global share memory
	memset(this->update_step, 0, (int) (user_index.size() * num_items));
	this->initialize_W_uniform_random();

	// Initialize kernel H in global share memory by proc-0
	if (this->proc_id == 0)	{
		this->initialize_H_uniform_random();
		printf(">>>\tA worker with id=%d performed init matrix H !\n", this->proc_id);
	}

	printf(">\tA worker with id=%d is created with: num_embed=%d, rand_state=%u! \n",
		this->proc_id, this->num_embeddings, this->random_seed);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Accessors and mutators
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// @brief: Initialize matrix W using random uniform distribution on real values
// 
void Worker::initialize_W_uniform_random() {
	double *w_ptr = this->W->local();
	std::uniform_real_distribution<double> distribution(0.0, 1.0 / sqrt(1.0 * this->num_embeddings));

	for (int i = 0; i < (int) this->user_index.size(); i++) {
		for (int j = 0; j < this->num_embeddings; j++) {
			int flatten_idx = i * (this->num_embeddings) + j;
			w_ptr[flatten_idx] = distribution(this->random_engine);
		}
	}
	return;
}

//
// @brief: Initialize matrix H using random uniform distribution on real values
//
void Worker::initialize_H_uniform_random() {
	double *h_ptr = this->H->local();
	std::uniform_real_distribution<double> distribution(0.0, 1.0 / sqrt(1.0 * this->num_embeddings));

	for (int i = 0; i < (int) this->num_items; i++) {
		for (int j = 0; j < this->num_embeddings; j++) {
			int flatten_idx = i * (this->num_embeddings) + j;
			h_ptr[flatten_idx] = distribution(this->random_engine);
		}
	}
	return;
}

//
// @brief: Add a new item index to item queue locally
//
void Worker::add_item_idx_to_queue(int item_idx) {
	this->item_queue->push(item_idx);
	return;
}

//
// @brief: Perform one iteration of SGD update
//
void Worker::update(int epoch_idx) {
	// printf("---> Proc-id=%02d:\t%d\titem_queue.size()=%d\n", this->proc_id, epoch_idx, (int)this->item_queue->size());
	if (this->item_queue->empty() == false) {
		// Get the first item index from the queue
		int item_idx = this->item_queue->front();
		this->item_queue->pop();

		// Compute new value of W and H
		// Remote update H using global_ptr
		this->update_value_W_and_H(item_idx);

		// Transfer the item to another process
		// int receiver_id = this->randomer(this->random_engine);
		int receiver_id = this->get_priority_process_index();
		this->transfer_item(receiver_id, item_idx).wait();

		// Print to debug the transfer process
		// if (upcxx::rank_me() == this->proc_id)
			// printf("Proc-id=%02d: item=%02d\t|||\treceiver=%02d\n", this->proc_id, item_idx, receiver_id);
	}
	return;
}

//
// @brief: Compute and update the new value of H and W
//
void Worker::update_value_W_and_H(int item_index) {
	for(int i=0;i<user_index.size(); i++){
		if(A[i][item_index] == 0.0) continue;

		// Compute the learning rate w.r.t to the time step
		int t = ++(this->update_step[i*this->num_items + item_index]);
		double lr = compute_learning_rate(this->update_step[t]);

		// printf("--->time_idx=%02d\tt=%d\tlr=%.4f\n",this->update_step[i*this->num_items + item_index],t,lr);

		double A_ij = A[i][item_index];

		vector<double> W_i(this->num_embeddings);
		upcxx::global_ptr<double> W_glptr = this->W.fetch(this->proc_id).wait();
		assert(W_glptr.is_local());
		double* W_ptr = W_glptr.local();

		vector<double> H_j(this->num_embeddings);
		upcxx::global_ptr<double> H_glptr = this->H.fetch(0).wait();
		assert(H_glptr.is_local());
		double* H_ptr = H_glptr.local();

		for(int k=0;k<this->num_embeddings;k++){
			W_i[k] = W_ptr[i*this->num_embeddings + k];
			H_j[k] = H_ptr[item_index*this->num_embeddings + k];
		}

		W_i =	vec_vec_subtract(
					W_i,
					vec_scalar_multiply(
						vec_vec_add(
							vec_scalar_multiply(
								H_j, 
								A_ij - vec_vec_multiply(W_i, H_j)
							), 
							vec_scalar_multiply(W_i, this->_lambda_)
						), 
						lr
					)
				);
		
		H_j =	vec_vec_subtract(
					H_j,
					vec_scalar_multiply(
						vec_vec_add(
							vec_scalar_multiply(
								W_i, 
								A_ij - vec_vec_multiply(W_i, H_j)
							), 
							vec_scalar_multiply(H_j, this->_lambda_)
						), 
						lr
					)
				);		
		
		for(int k=0;k<this->num_embeddings;k++){
			// Push W
			upcxx::rput(
				W_i[k],
				W_glptr + i*this->num_embeddings + k
			).wait();

			// Push H
			upcxx::rput(
				H_j[k],
				H_glptr + item_index*this->num_embeddings + k
			).wait();
		}
	}

	return;
}

//
// @brief: Push the item index to another process
//
upcxx::future<> Worker::transfer_item(int worker_id, int item_index) {
	return upcxx::rpc(
		worker_id,
		[](upcxx::dist_object<queue<int>> &item_queue, int item_idx) {
			item_queue->push(item_idx);
		},
		item_queue, item_index);
}

//
// @brief: Dynamic Load Balancing: Get the index of 
// process/worker that have the least amount of item indices
//
int Worker::get_priority_process_index() {
	int min_capac = this->num_items;
	int min_proc_id = -1;
	for (int proc_id = 0; proc_id < upcxx::rank_n(); proc_id++) {
		int remote_capac = upcxx::rpc(
							   proc_id,
							   [](upcxx::dist_object<queue<int>> &item_queue) {
								   return (int)item_queue->size();
							   },
							   item_queue)
							   .wait();
		if (remote_capac <= min_capac) {
			min_capac = remote_capac;
			min_proc_id = proc_id;
		}
	}
	return min_proc_id;
}

//
// @brief: Compute the new learning rate w.r.t to the time step
//
double Worker::compute_learning_rate(int time) {
	double lr = this->_alpha_ / (1.0 + this->_beta_ * pow((double)1.0 * time, 1.5));
	return lr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Math: Linear algebra functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// @brief: Element-wise multiplication of 2 vectors
//
double Worker::vec_vec_multiply(vector<double> vec1, vector<double> vec2) {
	assert(vec1.size() == vec2.size());

	double ans = 0.0;
	for (int i = 0; i < vec1.size(); i++) {
		ans += vec1[i] * vec2[i];
	}
	return ans;
}

//
// @brief: Multiplication of a vector and a scalar value
//
vector<double> Worker::vec_scalar_multiply(vector<double> vec, double scalar) {
	vector<double> ans = vec;
	for (int i = 0; i < ans.size(); i++) {
		ans[i] *= scalar;
	}
	return ans;
}

//
// @brief: Addition of 2 vectors
//
vector<double> Worker::vec_vec_add(vector<double> vec1, vector<double> vec2) {
	assert(vec1.size() == vec2.size());

	vector<double> ans(vec1.size(), 0.0);
	for (int i = 0; i < ans.size(); i++) {
		ans[i] = vec1[i] + vec2[i];
	}
	return ans;
}

//
// @brief: Subtraction of 2 vectors
//
vector<double> Worker::vec_vec_subtract(vector<double> vec1, vector<double> vec2) {
	assert(vec1.size() == vec2.size());

	vector<double> ans(vec1.size(), 0.0);
	for (int i = 0; i < ans.size(); i++) {
		ans[i] = vec1[i] - vec2[i];
	}
	return ans;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Debugging functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// @brief: Print to debug the value in: local rows A, working rows of W
// and all matrix H (performed on process-0)
//
void Worker::print_debug_matrix(bool print_A = false, bool print_W = false, bool print_H = false) {
	printf("----> proc-id = %d\n", this->proc_id);

	if (print_A == true) {
		printf(" ** Segment of A ** \n");
		for (int i = 0; i < this->A.size(); i++) {
			printf("user-id = %02d\t", this->user_index.at(i));
			for (int j = 0; j < this->A.at(i).size(); j++)
				printf("%.0f  ", this->A.at(i).at(j));
			printf("\n");
		}
	}

	if (print_W == true) {
		upcxx::global_ptr<double> w_obj = this->W.fetch(this->proc_id).wait();

		printf(" ** Segment of W ** \n");
		for (int i = 0; i < (int)this->user_index.size(); i++) {
			printf("user-id = %02d\t", this->user_index.at(i));
			for (int j = 0; j < this->num_embeddings; j++) {
				int flatten_idx = i * (this->num_embeddings) + j;
				double val = upcxx::rget(w_obj + flatten_idx).wait();

				printf("%.2f  ", val);
			}
			printf("\n");
		}
	}

	if (print_H == true && this->proc_id == 0) {
		upcxx::global_ptr<double> h_obj = this->H.fetch(this->proc_id).wait();

		printf(" ** Segment of H ** \n");
		for (int i = 0; i < (int)this->num_items; i++) {
			printf("item-no = %02d\t", i);
			for (int j = 0; j < this->num_embeddings; j++) {
				int flatten_idx = i * (this->num_embeddings) + j;
				double val = upcxx::rget(h_obj + flatten_idx).wait();

				printf("%.2f  ", val);
			}
			printf("\n");
		}
	}

	printf("\n");
	return;
}

//
// @brief: Print the content int the processing queue
//
void Worker::print_debug_queue() {
	vector<int> back_up_queue;

	printf("The queue of proc-id=%d:\t", this->proc_id);
	while (this->item_queue->empty() == false) 	{
		printf("%02d  ", this->item_queue->front());
		back_up_queue.push_back(this->item_queue->front());
		this->item_queue->pop();
	}
	printf("\n");

	for (auto v : back_up_queue)
		this->item_queue->push(v);

	return;
}



