#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN = 400 + 3;
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

int power(int a, int b){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % MOD;
        b >>= 1,  a = 1ll * a * a % MOD;
    }
    return r;
}

int inv(int x){
    return power(x, MOD - 2);
}

bool mat_inv(Mat &a){
    const int &n = a.n;
    Mat b(n, n);
    for(int i = 1;i <= n;++ i)
        b.W[i][i] = 1;
    for(int i = 1;i <= n;++ i){
        int f = -1;
        for(int j = i;j <= n;++ j) if(a.W[j][i] != 0){
            f = j; 
            break;
        }
        if(f == -1){
            return false;
        }
        if(f != i){
            for(int j = 1;j <= n;++ j)
                swap(a.W[i][j], a.W[f][j]),
                swap(b.W[i][j], b.W[f][j]);
        }
        int invp = inv(a.W[i][i]);
        for(int j = i + 1;j <= n;++ j){
            int rate = 1ll * a.W[j][i] * invp % MOD;
            for(int k = 1;k <= n;++ k){
                a.W[j][k] = (a.W[j][k] - 1ll * rate * a.W[i][k] % MOD + MOD) % MOD;
                b.W[j][k] = (b.W[j][k] - 1ll * rate * b.W[i][k] % MOD + MOD) % MOD;
            }
        }
    }
    for(int i = n;i >= 1;-- i){
        int invp = inv(a.W[i][i]);
        for(int j = 1;j <= n;++ j){
            a.W[i][j] = 1ll * a.W[i][j] * invp % MOD;
            b.W[i][j] = 1ll * b.W[i][j] * invp % MOD;
        }
        for(int j = i - 1;j >= 1;-- j){
            int rate = 1ll * a.W[j][i] % MOD;
            for(int k = 1;k <= n;++ k){
                a.W[j][k] = (a.W[j][k] - 1ll * rate * a.W[i][k] % MOD + MOD) % MOD;
                b.W[j][k] = (b.W[j][k] - 1ll * rate * b.W[i][k] % MOD + MOD) % MOD;
            }
        }
    }
    for(int i = 1;i <= n;++ i)
        for(int j = 1;j <= n;++ j)
            a.W[i][j] = b.W[i][j];
    return true;
}

int X[MAXN];

int main(){
    int n;
    cin >> n;
    Mat A(n, n);
    for(int i = 1;i <= n;++ i)
        for(int j = 1;j <= n;++ j)
            cin >> A.W[i][j];
    bool res = mat_inv(A);
    if(res == false){
        cout << "No Solution" << endl;
    } else {
        for(int i = 1;i <= n;++ i)
            for(int j = 1;j <= n;++ j)
                cout << A.W[i][j] << " \n"[j == n];
    }
    return 0;
}