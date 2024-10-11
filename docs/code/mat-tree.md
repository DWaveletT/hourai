```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN = 300 + 3;
const int MOD  = 1e9 + 7;

struct Mat{
    int n, m;
    int W[MAXN][MAXN];
    Mat(int _n = 0, int _m = 0){
        n = _n;
        m = _m;
        for(int i = 1;i <= n;++ i)
            for(int j = 1;j <= m;++ j)
                W[i][j] = 0;
    }
};

int mat_det(Mat a){
    int ans = 1;

    const int &n = a.n;
    for(int i = 1;i <= n;++ i){
        int f = -1;
        for(int j = i;j <= n;++ j) if(a.W[j][i] != 0){
            f = j;
            break;
        }
        if(f == -1){
            return 0;
        }
        if(f != i){
            for(int j = 1;j <= n;++ j)
                swap(a.W[i][j], a.W[f][j]);
            ans = MOD - ans;
        }
        for(int j = i + 1;j <= n;++ j) if(a.W[j][i]){
            while(a.W[j][i]){
                int u = a.W[i][i];
                int v = a.W[j][i];
                if(u > v){
                    for(int k = 1;k <= n;++ k)
                        swap(a.W[i][k], a.W[j][k]);
                    ans = MOD - ans;
                    swap(u, v);
                }
                int rate = v / u;
                for(int k = 1;k <= n;++ k){
                    a.W[j][k] = (a.W[j][k] - 1ll * rate * a.W[i][k] % MOD + MOD) % MOD;
                }
            }
        }
    }
    for(int i = 1;i <= n;++ i)
        ans = 1ll * ans * a.W[i][i] % MOD;
    return ans;
}

int D[MAXN];
int W[MAXN][MAXN];

int main(){
    int n, m, t;
    cin >> n >> m >> t;
    for(int i = 1;i <= m;++ i){
        int u, v, w;
        cin >> u >> v >> w;
        if(u != v){
            if(t == 0){ // 无向图
                D[u] = (D[u] + w) % MOD;
                D[v] = (D[v] + w) % MOD;
                W[u][v] = (W[u][v] + w) % MOD;
                W[v][u] = (W[v][u] + w) % MOD;
            } else {    // 叶向树
                D[v] = (D[v] + w) % MOD;
                W[u][v] = (W[u][v] + w) % MOD;
            }
        }
    }
    Mat A(n - 1, n - 1);
    for(int i = 2;i <= n;++ i)
        for(int j = 2;j <= n;++ j)  // 以 1 为根的叶向树
            A.W[i - 1][j - 1] = MOD - W[i][j];
    for(int i = 2;i <= n;++ i)
        A.W[i - 1][i - 1] = (D[i] + A.W[i - 1][i - 1]) % MOD;
    cout << mat_det(A) << endl;
    return 0;
}
```
