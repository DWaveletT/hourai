#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN = 600 + 3;
int MOD;

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

int main(){
    int n;
    cin >> n >> MOD;
    Mat A(n, n);
    for(int i = 1;i <= n;++ i)
        for(int j = 1;j <= n;++ j)
            cin >> A.W[i][j], A.W[i][j] %= MOD;
    cout << mat_det(A) << endl;
    return 0;
}