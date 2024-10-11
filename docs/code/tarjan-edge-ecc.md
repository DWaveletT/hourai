```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN= 5e5 + 3;
vector <vector<int>> A;
vector <pair<int, int>> V[MAXN];
stack  <int> S;
int D[MAXN], L[MAXN], o;
bool I[MAXN];
void dfs(int u, int l){
    D[u] = L[u] = ++ o; I[u] = true, S.push(u); int s = 0;
    for(auto &p : V[u]) {
        int v = p.first, id = p.second;
        if(id != l){
            if(D[v]){
                if(I[v])    L[u] = min(L[u], D[v]);
            } else {
                dfs(v, id), L[u] = min(L[u], L[v]), ++ s;
            }
        }
    }
    if(D[u] == L[u]){
        vector <int> T;
        while(S.top() != u){
            int v = S.top(); S.pop();
            T.push_back(v), I[v] = false;
        }
        T.push_back(u), S.pop(), I[u] = false;
        A.push_back(T);
    }
}

```
