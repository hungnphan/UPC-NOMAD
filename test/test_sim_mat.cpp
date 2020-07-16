#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>
#include <chrono>
#include <cmath>
#include <ctime>
using namespace std;

//
// @brief: Calculate similarity matrix among users
//
vector<vector<double>> calculate_simarity_matrix(vector<vector<double>> &arr_data) {
    // Calculate mean of each user
    vector<double>mean_usr(arr_data.size(), 0.0);
    for(int i=0; i<arr_data.size();i++){
        double sum = 0.0;
        int non_zero_cnt = 0;
        for(int j=0;j<arr_data[i].size();j++){
            if(arr_data[i][j] != -1.0){
                sum += arr_data[i][j];
                non_zero_cnt++;
            }
        }
        mean_usr[i] = sum / (1.0*non_zero_cnt);
        printf("%.2f  ", mean_usr[i]);
    }
    printf("\n--------------------------\n\n");

    // Normalize the data: X = X - mean
    vector<vector<double>> norm_data = arr_data;
    for(int i=0; i<norm_data.size();i++){
        for(int j=0;j<norm_data[i].size();j++){
            if(norm_data[i][j] != -1.0){
                norm_data[i][j] -=  mean_usr[i];
            }
            else norm_data[i][j] =  0.0;
            printf("%.2f  ", norm_data[i][j]);
        }
        printf("\n");
    }
    printf("\n--------------------------\n\n");

    // Calculate similarity matrix
    vector<vector<double>> sim_mat;
    sim_mat.resize(norm_data.size());
    for(int i=0; i<norm_data.size();i++){
        sim_mat[i].resize(norm_data.size());
        for(int j=0;j<norm_data.size();j++){
            sim_mat[i][j] = 0.0;
            double dis_i = 0.0;
            double dis_j = 0.0;
            for(int k=0;k<norm_data[i].size();k++){
                sim_mat[i][j] += (norm_data[i][k] * norm_data[j][k]);
                dis_i += (norm_data[i][k] * norm_data[i][k]);
                dis_j += (norm_data[j][k] * norm_data[j][k]);
            }
            sim_mat[i][j] = sim_mat[i][j] / (sqrt(dis_i) * sqrt(dis_j));
        }
    }


    return sim_mat;
}


int main(){
    int arr[7][5] = {  
        {5, 4,-1, 2, 2} ,
        {5,-1, 4, 2, 0} ,
        {2,-1, 1, 3, 4} ,
        {0, 0,-1, 4,-1} ,
        {1,-1,-1, 4,-1} ,
        {-1,2, 1,-1,-1} ,
        {-1,-1,1, 4, 5} 
    };

    vector<vector<double>> v(7);
    for(int i=0;i<7;i++){
        for(int j=0;j<5;j++){
            v[i].push_back(1.0*arr[i][j]);
        }
    }

    vector<vector<double>>ans = calculate_simarity_matrix(v);
    for(int i=0;i<7;i++){
        for(int j=0;j<7;j++){
            printf("%.2f  ",ans[i][j]);
        }
        printf("\n");
    }

    return 0;

}