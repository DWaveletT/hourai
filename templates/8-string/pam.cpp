#include<bits/stdc++.h>
using namespace std;

const int MAXM = 26 + 3;
namespace PAM{
    const int SIZ = 5e5 + 3;
    int n, s, F[SIZ], L[SIZ], D[SIZ];
    int M[SIZ][MAXM];
    char S[SIZ];
    void init(){
        S[0] = '$', n = 1;
        F[s = 0] = -1, L[0] = -1, D[0] = 0;
        F[s = 1] =  0, L[1] =  0, D[1] = 0;
    }
    void extend(int &last, char c){
        S[++ n] = c;
        int e = c - 'a';
        int a = last;
        while(c != S[n - 1 - L[a]])
            a = F[a];
        if(M[a][e]){
            last = M[a][e];
        } else {
            int cur = M[a][e] = ++ s;
            L[cur] = L[a] + 2;
            if(a == 0){
                F[cur] = 1;
            } else {
                int b = F[a];
                while(c != S[n - 1 - L[b]])
                    b = F[b];
                F[cur] = M[b][e];
            }
            D[cur] = D[F[cur]] + 1;
            last = cur;
        }
    }
}

const int MAXN = 5e5 + 3;
char T[MAXN];
int main(){
    PAM :: init();
    int m = 0, last = 0, lastans = 0;
    for(char c = getchar();isalpha(c);c = getchar()){
        char d = (c - 97 + lastans) % 26 + 97;
        PAM :: extend(last, d);
        cout << (lastans = PAM :: D[last]) << " ";
    }
    return 0;
}
/*
azzzyyzyyx
1 2 1 2 3 2 2 2 3 3
*/