#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <fstream>
#define bug(x) cout << #x << " = " << x << endl
using namespace std;

int get_random_rating(){
    return 1+(rand()%5);
}

int get_amount_zero(int max_num_zero){
    int min_num_zero = 0.35 * max_num_zero;
    int ans = rand()%(max_num_zero-min_num_zero+1) + min_num_zero;
    return ans;
}

int main(int argc, char **argv){
    // Collect program arguments
    if (argc < 3)
        exit(0);
    string file_output(argv[1]);
    int NROW = atoi(argv[2]);
    int NCOL = atoi(argv[3]);;

    srand (time(NULL));   

    // int a[NROW][NCOL] = {0};
    vector<vector<int>>a(NROW);
    int zeros_row[NROW] = {NCOL};
    int zeros_col[NCOL] = {NROW};

    // Fill matrix a with random int value [1,5]
    for(int i=0;i<NROW;i++){
        for(int j=0;j<NCOL;j++){
            // a[i][j] = get_random_rating();
            a[i].push_back(get_random_rating());
        }
    }

    int max_num_zeros = NROW*NCOL - max(NROW,NCOL);
    int num_zeros = get_amount_zero(max_num_zeros);

    // bug(max_num_zeros);
    // bug(num_zeros);

    // for(int i=0;i<NROW;i++){
    //     for(int j=0;j<NCOL;j++){
    //         cout << a[i][j] << "  ";
    //     }
    //     cout << endl;
    // }

    int zero_cnt = 0;
    while(zero_cnt < num_zeros){
        int r = rand()%NROW;
        int c = rand()%NCOL;

        if(a[r][c]==0) continue;

        while(zeros_row[r] == 1){
            r = (r+1)%NROW;
            // bug(r);bug(zeros_row[r]);
        }
        while(zeros_col[c] == 1){
            c = (c+1)%NCOL;
            // bug(c);bug(zeros_col[c]);
        }

        zeros_row[r]--;
        zeros_col[c]--;
        a[r][c] = 0;

        zero_cnt++;
    }
    

    ofstream ofile(file_output, ios::out);
    ofile << NROW << ' ' << NCOL << endl;
    for(int i=0;i<NROW;i++){
        for(int j=0;j<NCOL;j++){
            ofile << a[i][j] << "  ";
        }
        ofile << endl;
    }
    ofile.close();

    return 0;

}
