```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN= 5e5 + 3;

vector <int> V[MAXN];
stack  <int> S;
int D[MAXN], L[MAXN], C[MAXN], o, s;
bool F[MAXN], I[MAXN];
void add(int u, int v){ V[u].push_back(v); }
void dfs(int u){
    L[u] = D[u] = ++ o, S.push(u), I[u] = F[u] = true;
    for(auto &v : V[u]){
        if(F[v]){
            if(I[v]) L[u] = min(L[u], D[v]);
        } else {
            dfs(v),  L[u] = min(L[u], L[v]);
        }
    }
    if(L[u] == D[u]){
        int c = ++ s;
        while(S.top() != u){
            int v = S.top(); S.pop();
            I[v] = false;
            C[v] = c;
        }
        S.pop(), I[u] = false, C[u] = c;
    }
}

vector <int> ANS[MAXN];
int main(){

    int n, m;
    cin >> n >> m;
    for(int i = 1;i <= m;++ i){
        int u, v;
        cin >> u >> v;
        V[u].push_back(v);
    }
    for(int i = 1;i <= n;++ i)
        if(!F[i])
            dfs(i);
    for(int i = 1;i <= n;++ i){
        ANS[C[i]].push_back(i);
    }
    cout << s << endl;
    for(int i = 1;i <= n;++ i) if(F[i]){
        int c = C[i];
        sort(ANS[c].begin(), ANS[c].end());
        for(auto &u : ANS[c])
            cout << u << " ", F[u] = false;
        cout << endl;
    }
    return 0;
}
```
