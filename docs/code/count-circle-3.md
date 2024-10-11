```cpp
#include<bits/stdc++.h>
using namespace std;

const int MAXN = 2e5 + 3;
vector <int> E[MAXN];
int D[MAXN], U[MAXN], V[MAXN];
bool F[MAXN];

int main(){
    int n, m;
    cin >> n >> m;
    for(int i = 1;i <= m;++ i){
        int u, v;
        cin >> u >> v;
        D[u] ++;
        D[v] ++;
        U[i] = u, V[i] = v;
    }
    for(int i = 1;i <= m;++ i){
        int u = U[i];
        int v = V[i];
        if(D[u] > D[v] || (D[u] == D[v] && u > v))
            swap(u, v);
        E[u].push_back(v);
    }

    int ans = 0;
    for(int u = 1;u <= n;++ u){
        for(auto &v: E[u])
            F[v] = 1;
        for(auto &v: E[u]){
            for(auto &w: E[v]){
                ans += F[w];
            }
        }
        for(auto &v: E[u])
            F[v] = 0;
    }
    cout << ans << "\n";

    return 0;
}
```
