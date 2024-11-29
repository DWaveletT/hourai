```cpp
#include<bits/stdc++.h>
using namespace std;
const int MAXN = 5e5 + 3;
vector<pair<int, int> > E[MAXN];
namespace LCA{
    const int SIZ = 5e5 + 3;
    int D[SIZ], H[SIZ], F[SIZ];
    int P[SIZ], Q[SIZ], o;
    void dfs(int u, int f){
        P[u] = ++ o;
        Q[o] = u;
        F[u] = f;
        D[u] = D[f] + 1;
        for(auto &[v, w] : E[u]) if(v != f){
            H[v] = H[u] + w, dfs(v, u);
        }
    }
    const int MAXH = 18 + 3;
    int h = 18;
    int ST[SIZ][MAXH];
    int cmp(int a, int b){
        return D[a] < D[b] ? a : b;
    }
    int T[SIZ], n;
    void init(int _n, int root){
        n = _n;
        dfs(root, 0);
        for(int i = 1;i <= n;++ i)
            ST[i][0] = Q[i];
        for(int i = 2;i <= n;++ i)
            T[i] = T[i >> 1] + 1;
        for(int i = 1;i <= h;++ i){
            for(int j = 1;j <= n;++ j) if(j + (1 << i - 1) <= n){
                ST[j][i] = cmp(ST[j][i - 1], ST[j + (1 << i - 1)][i - 1]);
            }
        }
    }
    int lca(int a, int b){
        if(a == b)
            return a;
        int l = P[a];
        int r = P[b];
        if(l > r)
            swap(l, r);
        ++ l;
        int d = T[r - l + 1];
        return F[cmp(ST[l][d], ST[r - (1 << d) + 1][d])];
    }
    int dis(int a, int b){
        return H[a] + H[b] - 2 * H[lca(a, b)];
    }
}
bool cmp(int a, int b){
    return LCA :: P[a] < LCA :: P[b];
}
bool I[MAXN];
vector <int> E1[MAXN];
vector <int> V1;
void solve(vector <int> &V){
    using LCA :: lca;
    using LCA :: D;
    stack <int> S;
    sort(V.begin(), V.end(), cmp);
    S.push(1);
    int v, l;
    for(auto &u : V) I[u] = true;
    for(auto &u : V) if(u != 1){
        int f = lca(u, S.top());
        l = -1;
        while(D[v = S.top()] > D[f]){
            if(l != -1)
                E1[v].push_back(l);
            V1.push_back(l = v), S.pop();
        }
        if(l != -1)
            E1[f].push_back(l);
        if(f != S.top())
            S.push(f);
        S.push(u);
    }
    l = -1;
    while(!S.empty()){
        v = S.top();
        if(l != -1)
            E1[v].push_back(l);
        V1.push_back(l = v), S.pop();
    }
    // dfs(1, 0); // SOLVE HERE !!!
    for(auto &u : V1)
        E1[u].clear(), I[u] = false;
    V1.clear();
}

```
