```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long i64;
const int MAXN = 2e3 + 3;
vector<int> E[MAXN];
int W[MAXN];
int F[MAXN][MAXN], S[MAXN];
void dfs(int u, int f){
    F[u][1] = W[u];
    S[u]    = 1;
    for(auto &v : E[u]) if(v != f){
        dfs(v, u);
        for(int i = S[u];i >= 1;-- i)
            for(int j = S[v];j >= 1;-- j)
                F[u][i + j] = max(F[u][i + j], F[u][i] + F[v][j]);
        S[u] += S[v];
    }
}
int main(){
    int n, m;
    cin >> n >> m;
    for(int i = 1;i <= n;++ i){
        int f;
        cin >> f >> W[i];
        E[f].push_back(i);
    }
    dfs(0, 0);
    cout << F[0][m + 1] << endl;
    return 0;
}
```
