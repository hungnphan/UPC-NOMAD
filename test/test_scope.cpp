#include <iostream>
#include <vector>
#include <cstdio>
#include <upcxx/upcxx.hpp>
using namespace std;

int main(){
    upcxx::init();

    int n = upcxx::rank_me();
    vector<int>vec;
    upcxx::global_ptr<int>ptr(upcxx::new_<int>(upcxx::rank_me()*10+5));
    int* lc_ptr = ptr.local();

    for(int i=0;i<upcxx::rank_n();i++){
        if(upcxx::rank_me() == i){
            n = upcxx::rank_me();
            for(int j=0;j<upcxx::rank_me();j++)
                vec.push_back(upcxx::rank_me() + j);
        }
        upcxx::barrier();
    }

    upcxx::barrier();
    
    printf("proc-id=%d ||| &n=%d ||| n=%02d ||| *(&n)=%02d ||| &vec=%d ||| vec.size()=%d\n",upcxx::rank_me(), &n, n, *(&n), &vec, vec.size());

    upcxx::barrier();

    printf("proc-id=%d ||| lc_ptr=%d ||| *lc_ptr=%02d\n",upcxx::rank_me(), lc_ptr, *lc_ptr);



    upcxx::finalize();
}