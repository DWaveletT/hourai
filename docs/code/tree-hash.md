```cpp
#include<bits/stdc++.h>
using namespace std;

using u64 = unsigned long long;
const int MAXN = 1e6 + 3;

u64 xor_shift(u64 x){
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}
u64 H[MAXN];

vector <int> E[MAXN];
void dfs(int u, int f){
    H[u] = 1;
    for(auto &v: E[u]) if(v != f){
        dfs(v, u);
        H[u] += H[v];
    }
    H[u] = xor_shift(H[u]); // !important
}

int main(){
    int n;
    cin >> n;
    for(int i = 2;i <= n;++ i){
        int u, v;
        cin >> u >> v;
        E[u].push_back(v);
        E[v].push_back(u);
    }
    dfs(1, 0);

    sort(H + 1, H + 1 + n);
    cout << (unique(H + 1, H + 1 + n) - H - 1) << endl;
    return 0;
}
```
