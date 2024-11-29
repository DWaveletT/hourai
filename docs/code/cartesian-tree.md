```cpp
#include <bits/stdc++.h>
using namespace std;
using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;
int qread();
const int MAXN = 1e7 + 3;
int n, L[MAXN], R[MAXN], A[MAXN];
void build(){
    stack <int> S;
    A[n + 1] = -1e9;
    for(int i = 1;i <= n + 1;++ i){
        int v = 0;
        while(!S.empty() && A[S.top()] > A[i]){
            auto u = S.top();
            R[u] = v;
            v    = u;
            S.pop();
        }
        L[i] = v;
        S.push(i);
    }
}
int main(){
    n = qread();
    for(int i = 1;i <= n;++ i)
        A[i] = qread();
    build();
    long long ans1 = 0, ans2 = 0;
    for(int i = 1;i <= n;++ i){
        // cout << L[i] << " " << R[i] << endl;
        ans1 ^= 1ll * i * (L[i] + 1);
        ans2 ^= 1ll * i * (R[i] + 1);
    }
    cout << ans1 << " " << ans2 << endl;
    return 0;
}
```
