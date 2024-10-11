/**
## 用法

离线计算 $x = [x_1, x_2, \cdots, x_n]$ 在模 $p$ 意义下的逆元。
**/
#include<bits/stdc++.h>
using namespace std;

int power(int a, int b, int p){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % p;
        b >>= 1,  a = 1ll * a * a % p;
    }
    return r;
}

const int MAXN = 5e6 + 3;
int A[MAXN], B[MAXN];
int P[MAXN], Q[MAXN];

int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);

    int n, p, K, S = 1;
    cin >> n >> p >> K;
    P[0] = 1;
    for(int i = 1;i <= n;++ i){
        cin >> A[i];
        P[i] = 1ll * P[i - 1] * A[i] % p;
    }
    Q[n] = power(P[n], p - 2, p);
    for(int i = n;i >= 1;-- i){
        Q[i - 1] = 1ll * Q[i] * A[i] % p;
        B[i] = 1ll * Q[i] * P[i - 1] % p;
    }
    int ans = 0;
    for(int i = 1;i <= n;++ i){
        S = 1ll * S * K % p;
        ans = (ans + 1ll * S * B[i]) % p;
    }
    cout << ans << "\n";
    
    return 0;
}