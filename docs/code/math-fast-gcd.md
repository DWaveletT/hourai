```cpp
#include<bits/stdc++.h>
#define up(l, r, i) for(int i = l, END##i = r;i <= END##i;++ i)
#define dn(r, l, i) for(int i = r, END##i = l;i >= END##i;-- i)
using namespace std;
typedef long long i64;
const int INF = 2147483647;
const int MAXN= 5e3 + 3;
const int MAXT= 1e6 + 3;
const int MAXM= 1e3 + 3;
int G[MAXM][MAXM];
int T[MAXT][3];
int A[MAXN], B[MAXN], o = 1e6, h = 1e3, V[MAXT];
int tgcd(int a, int b){
    if(a <= h && b <= h) return G[a][b];
    return a == b ? a : 1;
}
int qgcd(int a, int b){
    int ans = 1;
    up(0, 2, i){
        if(T[b][i] > h){
            if(a % T[b][i] == 0) a /= T[b][i], ans *= T[b][i];
        } else {
            int d = G[a % T[b][i]][T[b][i]];
            a /= d, ans *= d;
        }
    }
    return ans;
}
const int MOD = 998244353;
int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);
    
    up(1, h, i) G[0][i] = G[i][0] = i;
    up(1, h, i) up(1, h, j){
        if(i >= j) G[i][j] = G[i - j][j];
            else   G[i][j] = G[i][j - i];
    }
    up(2, o, i) if(!V[i]){
        V[i] = i;
        for(int j = 2;i * j <= o;++ j)
            if(!V[i * j]) V[i * j] = i;
    }
    T[1][0] = T[1][1] = T[1][2] = 1;
    up(2, o, i){
        int p = V[i];
        int a = T[i / p][0];
        int b = T[i / p][1];
        int c = T[i / p][2];
        int x, y, z;
        if(p >= h){
            x = 1, y = i / p, z = p;
        } else {
            if(c * p <= h){
                x = a, y = b, z = c * p;
            }
            else if(b * p <= h){
                x = a, y = b * p, z = c;
                if(y > z) swap(y, z);
            }
            else if(a * p <= h){
                x = a * p, y = b, z = c;
                if(x > y) swap(x, y);
                if(y > z) swap(y, z);
            } else {
                x = a * b, y = c, z = p;
                if(x > y) swap(x, y);
                if(y > z) swap(y, z);
                if(x > z) swap(x, z);
            }
        }
        T[i][0] = x;
        T[i][1] = y;
        T[i][2] = z;
    }
    int n;
    cin >> n;
    up(1, n, i) cin >> A[i];
    up(1, n, i) cin >> B[i];
    up(1, n, i){
        int s = 0, u = 1;
        up(1, n, j){
            int d = qgcd(A[i], B[j]);
            u = 1ll * u * i % MOD;
            s = (s + 1ll * d * u) % MOD;
        }
        printf("%d\n", s);
    }
    return 0;
}
```
