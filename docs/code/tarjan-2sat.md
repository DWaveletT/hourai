```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;
namespace SCC{
    const int MAXN= 2e6 + 3;

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
}
const int MAXN = 1e6 + 3;
int X[MAXN][2], o;
int main(){
    ios :: sync_with_stdio(false);
    int n, m;
    cin >> n >> m;
    
    for(int i = 1;i <= n;++ i)
        X[i][0] = ++ o;
    for(int i = 1;i <= n;++ i)
        X[i][1] = ++ o;
    for(int i = 1;i <= m;++ i){
        int a, x, b, y;
        cin >> a >> x >> b >> y;
        SCC :: add(X[a][!x], X[b][y]);
        SCC :: add(X[b][!y], X[a][x]);
    }
    for(int i = 1;i <= o;++ i)
        if(!SCC :: F[i])
            SCC :: dfs(i);
    bool ok = true;
    for(int i = 1;i <= n;++ i){
        if(SCC :: C[X[i][0]] == SCC :: C[X[i][1]])
            ok = false;
    }
    if(ok){
        cout << "POSSIBLE" << endl;
        for(int i = 1;i <= n;++ i){
            int a = SCC :: C[X[i][0]];
            int b = SCC :: C[X[i][1]];
            if(a < b)
                cout << 0 << " ";
            else 
                cout << 1 << " ";
        }
        cout << endl;
    } else {
        cout << "IMPOSSIBLE" << endl;
    }
    return 0;
}
```
