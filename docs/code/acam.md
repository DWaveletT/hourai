```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;

const int MOD = 1e9 + 7;

namespace ACAM{
    const int MAXN =1e6 + 3;
    const int MAXM = 26 + 3;
    int C[MAXN][MAXM], o;
    
    void insert(char *S){
        int p = 0, len = 0;
        for(int i = 0;S[i];++ i){
            int e = S[i] - 'a';
            if(C[p][e]){
                p = C[p][e];
            } else {
                p = C[p][e] = ++ o;
            }
            ++ len;
        }
    }
    int F[MAXN];
    void build(){
        queue <int> Q; Q.push(0);
        while(!Q.empty()){
            int u = Q.front(); Q.pop();
            for(int i = 0;i < 26;++ i){
                int v = C[u][i];
                if(v == 0)
                    continue;
                int p = F[u];
                while(!C[p][i] && p != 0)
                    p = F[p];
                if(C[p][i] && C[p][i] != v)
                    F[v] = C[p][i];
                Q.push(v);
            }
        }
    }
}
```
