```cpp
#include<bits/stdc++.h>
using namespace std;
using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;
const int MAXN= 5e5 + 3;
vector <vector<int>> A;
vector <int> V[MAXN];
stack  <int> S;
int D[MAXN], L[MAXN], o; bool I[MAXN];
void dfs(int u, int f){
    D[u] = L[u] = ++ o; I[u] = true, S.push(u); int s = 0;
    for(auto &v : V[u]) if(v != f){
        if(D[v]){
            if(I[v])   L[u] = min(L[u], D[v]);
        } else {
            dfs(v, u), L[u] = min(L[u], L[v]), ++ s;
            if(L[v] >= D[u]){
                vector <int> T;
                while(S.top() != v){
                    int t = S.top(); S.pop();
                    T.push_back(t), I[t] = false;
                }
                T.push_back(v), S.pop(), I[v] = false;
                T.push_back(u);
                A.push_back(T);
            }
        }
    }
    if(f == 0 && s == 0){
        A.push_back({u});
    }
}

```
