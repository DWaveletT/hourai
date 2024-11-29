#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN = 3e3 + 3;

vector<pair<int, int> > E[MAXN];
i64 D[MAXN];

void dij(int n, int s){
    using nd = pair<i64, int>;
    priority_queue <nd, vector<nd>, greater<nd> > Q;
    for(int i = 1;i <= n;++ i)
        D[i] = INFL;
    Q.push(make_pair(0, s));
    while(!Q.empty()){
        auto [d, u] = Q.top();
        Q.pop();
        if(D[u] == INFL) {
            D[u] = d;
            for(auto &[v, w] : E[u]){
                if(D[v] == INFL)
                    Q.push(make_pair(d + w, v));
            }
        }
    }
}

bool inQ[MAXN];
int C[MAXN];

bool spfa(int n, int s){
    queue <int> Q;
    for(int i = 1;i <= n;++ i)
        D[i] = INFL;
    D[s] = 0;
    Q.push(s), inQ[s] = true, ++ C[s];
    while(!Q.empty()){
        int u = Q.front();
        if(C[u] > n)
            return false;
        Q.pop(), inQ[u] = false;
        for(auto &[v, w] : E[u]){
            if(D[u] + w < D[v]){
                D[v] = D[u] + w;
                if(!inQ[v]){
                    inQ[v] = true;
                    Q.push(v), ++ C[v];
                }
            }
        }
    }
    return true;
}

i64 T[MAXN];
i64 H[MAXN][MAXN];
bool johnson(int n){
    for(int i = 1;i <= n;++ i){
        E[n + 1].push_back(make_pair(i, 0));
    }
    bool res = spfa(n + 1, n + 1);
    if(res == false){
        E[n + 1].clear();
        return false;
    } else {
        for(int i = 1;i <= n;++ i)
            T[i] = D[i];
        for(int u = 1;u <= n;++ u){
            for(auto &[v, w] : E[u]){
                w = w + T[u] - T[v];
            }
        }
        for(int i = 1;i <= n;++ i){
            dij(n, i);
            for(int j = 1;j <= n;++ j){
                if(D[j] == INFL)
                    H[i][j] = INFL;
                else
                    H[i][j] = D[j] - T[i] + T[j];
            }
        }
        E[n + 1].clear();
        return true;
    }
}