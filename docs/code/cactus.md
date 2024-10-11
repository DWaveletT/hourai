```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN= 5e5 + 3;

struct edge{int u, v, w;};
vector <edge> V1[MAXN];
vector <edge> V2[MAXN];
vector <int> H[MAXN];
int n, D[MAXN], W[MAXN], F[MAXN];
int o, X[MAXN], L[MAXN];
bool E[MAXN];
void dfs(int u, int f){
    D[u] = D[f] + 1, F[u] = f;
    for(auto &e : V1[u]) if(e.v != f){
        if(D[e.v] && D[e.v] < D[u]){
            int a = e.u;
            int b = e.v;
            int c = ++ o, t = c + n;
            H[c].push_back(a);
            L[c] = W[a] - W[b] + e.w;
            while(a != b)
                E[a] = true, a = F[a], H[c].push_back(a);
            for(auto &x : H[c]){
                int w = min(W[x] - W[b], L[c] - W[x] + W[b]);
                V2[x].push_back(edge{x, t, w});
                V2[t].push_back(edge{t, x, w});
            }
        } else if(!D[e.v]){
            W[e.v] = W[u] + e.w, dfs(e.v, u);
        }
    }
    for(auto &e : V1[u]) if(D[e.v] > D[u]){
        if(!E[e.v]){
            V2[e.u].push_back({e.u, e.v, e.w});
            V2[e.v].push_back({e.v, e.u, e.w});
        }
    }
}
```
