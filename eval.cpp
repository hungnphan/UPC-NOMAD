#include <iostream>
#include <cstdio>
#include <vector>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <cmath>
using namespace std;

vector<vector<double>> pred_;
vector<vector<double>> true_;

// Argument:
//  + argv[1]   =   pred_result (char*, e.g. "matrix.txt")
//  + argv[2]   =   true_label (char*, e.g. "matrix.txt")
int main(int argc, char **argv) {
    // Collect program arguments
    string output_pred(argv[1]);
    string output_true(argv[2]);

    // Read predicted output
    ifstream pred_file(output_pred, ios::in);
    int ROWS, COLS;
    pred_file >> ROWS >> COLS;
    pred_.resize(ROWS);
    for(int i=0; i<ROWS; i++){
        pred_[i].resize(COLS);
        for(int j=0;j<COLS;j++){
            pred_file >> pred_[i][j];
        }
    }
    pred_file.close();

    // Read groundtruth values
    ifstream true_file(output_true, ios::in);
    int ROWS_, COLS_;
    true_file >> ROWS_ >> COLS_;
    true_.resize(ROWS_);
    for(int i=0; i<ROWS_; i++){
        true_[i].resize(COLS_);
        for(int j=0;j<COLS_;j++){
            true_file >> true_[i][j];
        }
    }
    true_file.close();

    // Calculate Mean squared error
    assert((ROWS == ROWS_) && (COLS == COLS_));
    int instance_cnt = 0;
    double total_dif = 0.0;
    for(int i=0; i<ROWS; i++){
        for(int j=0;j<COLS;j++){
            if(true_[i][j] != 0){
                instance_cnt++;
                total_dif += pow(abs(pred_[i][j] - true_[i][j]),2.0);
            }
        }
    }

    cout << "Total instance checked = " << instance_cnt << endl;
    cout << "MSE = " << fixed << setprecision(4) << sqrt(total_dif/(1.0*instance_cnt)) << endl;

    return 0;
}
