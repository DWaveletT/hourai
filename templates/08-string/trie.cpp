#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXM= 10 + 3;

namespace Trie{
    const int SIZ = 1e6 + 3;
    int M[SIZ][MAXM], s, h = 10;

    void extend(int &last, char c){
        int e = c - 'a';
        if(M[last][e]){
            last = M[last][e];
        } else {
            last = M[last][e] = ++ s;
        }
    }
    void insert(char *S){
        int p = 0;
        for(int i = 0;S[i];++ i){
            int e = S[i] - 'a';
            if(M[p][e]){
                p = M[p][e];
            } else 
                p = M[p][e] = ++ s;
        }
    }
}