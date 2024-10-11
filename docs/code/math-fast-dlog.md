```cpp
#include<bits/stdc++.h>
using namespace std;

int power(int a, int b, int  p){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % p;
        b >>= 1,  a = 1ll * a * a % p;
    }
    return r;
}
namespace BSGS {
    unordered_map <int, int> M;
    int B, U, P, g;
    void init(int g, int P0, int B0){
        M.clear();
        B = B0;
        P = P0;
        U = power(power(g, B, P), P - 2, P);
        int w = 1;
        for(int i = 0;i < B;++ i){
            M[w] = i;
            w = 1ll * w * g % P;
        }
    }
    int solve(int y){
        int w = y;
        for(int i = 0;i < P / B;++ i){
            if(M.count(w)){
                return i * B + M[w];
            }
            w = 1ll * w * U % P;
        }
        return -1;
    }
}

const int MAXN = 1e5 + 3;
int H[MAXN], P[MAXN], H0, p, h, g, mod;
bool V[MAXN];

int solve(int x){
    if(x <= h){
        return H[x];
    }
    int v = mod / x, r = mod % x;
    if(r < x - r){
        return ((H0 + solve(r)) % (mod - 1) - H[v] + mod - 1) % (mod - 1);
    } else {
        return (solve(x - r) - H[v + 1] + mod - 1) % (mod - 1);
    }
}

int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);
    
    int T;
    cin >> g >> mod;
    h = sqrt(mod) + 1;

    BSGS :: init(g, mod, sqrt(1ll * mod * sqrt(mod) / log10(mod)));
    H0 = BSGS :: solve(mod - 1);

    H[1] = 0;
    for(int i = 2;i <= h;++ i){
        if(!V[i]){
            P[++ p] = i;
            H[i] = BSGS :: solve(i);
        }
        for(int j = 1;j <= p && P[j] <= h / i;++ j){
            int &p = P[j];
            H[i * p] = (H[i] + H[p]) % (mod - 1);
            V[i * p] = true;
            if(i % p == 0)
                break;
        }
    }

    cin >> T;
    while(T --){
        int x, tmp = 0;
        cin >> x;
        cout << solve(x) << "\n";
    }
    return 0;
}
/*
3 998244353
9
1
11
111
1111
11111
111111
1111111
11111111
111111111

5 1000000007
9
1
12
123
1234
12345
123456
1234567
12345678
123456789
*/
```
