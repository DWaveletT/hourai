```cpp
#include<bits/stdc++.h>
#define up(l, r, i) for(int i = l, END##i = r;i <= END##i;++ i)
#define dn(r, l, i) for(int i = r, END##i = l;i >= END##i;-- i)
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAX_ = (1 << 19) + 3;

template <typename T>
struct cplx0{
    T a, b; cplx0(T _a = 0, T _b = 0) :a(_a), b(_b){}
    cplx0 operator +(cplx0 t){ return cplx0(a + t.a, b + t.b); }
    cplx0 operator -(cplx0 t){ return cplx0(a - t.a, b - t.b); }
    cplx0 operator *(cplx0 t){ return cplx0(a * t.a - b * t.b, a * t.b + b * t.a); }
    cplx0 operator *(int t) { return cplx0(a * t, b * t); }
};
using cplx = cplx0<double>;

const long double pi = acos(-1);
namespace Poly{
    void FFT(int n, cplx Z[]){
        static int W[MAX_];
        int l = 1; W[0] = 0;
        while (n >>= 1)
            up(0, l - 1, i)
                W[l++] = W[i] << 1 | 1, W[i] <<= 1;
        up(0, l - 1, i)
            if(W[i] > i) swap(Z[i], Z[W[i]]);
        for (n = l >> 1, l = 1;n;n >>= 1, l <<= 1){
            cplx* S = Z;
            cplx0<long double> o(cosl(pi / l), sinl(pi / l));
            up(0, n - 1, i){
                cplx0<long double> s(1, 0);
                up(0, l - 1, j){
                    cplx x = S[j] + cplx(s.a, s.b) * S[j + l];
                    cplx y = S[j] - cplx(s.a, s.b) * S[j + l];
                    S[j] = x, S[j + l] = y, s = s * o;
                }
                S += l << 1;
            }
        }
    }
    void IFFT(int n, cplx Z[]){
        FFT(n, Z); reverse(Z + 1, Z + n);
        up(0, n - 1, i)
            Z[i].a /= 1.0 * n, Z[i].b /= 1.0 * n;
    }
}

const int MAXN  = (1 << 19) + 3;
const int BLOCK = 32768;
cplx A1[MAXN], A2[MAXN], B1[MAXN], B2[MAXN];
int n, m, L, mod;

cplx P[MAXN], Q[MAXN];

void FFTFFT(int L, cplx X[], cplx Y[]){
    for(int i = 0;i < L;++ i){
        P[i].a = X[i].a;
        P[i].b = Y[i].a;
    }
    Poly :: FFT(L, P);
    for(int i = 0;i < L;++ i){
        Q[i] = (i == 0 ? P[0] : P[L - i]);
        Q[i].b = -Q[i].b;
    }
    for(int i = 0;i < L;++ i){
        X[i] = (P[i] + Q[i]);
        Y[i] = (Q[i] - P[i]) * cplx(0, 1);
        X[i].a /= 2.0, X[i].b /= 2.0;
        Y[i].a /= 2.0, Y[i].b /= 2.0;
    }
}

int main(){
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n >> m >> mod;
    for(int i = 0;i <= n;++ i){
        int a; cin >> a;
        a %= mod;
        A1[i].a = a / BLOCK;
        A2[i].a = a % BLOCK;
    }
    for(int i = 0;i <= m;++ i){
        int a; cin >> a;
        a %= mod;
        B1[i].a = a / BLOCK;
        B2[i].a = a % BLOCK;
    }
    for(L = 1;L <= n + m;L <<= 1);
    FFTFFT(L, A1, A2);
    FFTFFT(L, B1, B2);
    for(int i = 0;i < L;++ i){
        P[i] = A1[i] * B1[i] + cplx(0, 1) * A2[i] * B1[i];
        Q[i] = A1[i] * B2[i] + cplx(0, 1) * A2[i] * B2[i];
    }
    Poly :: IFFT(L, P);
    Poly :: IFFT(L, Q);
    for(int i = 0;i < L;++ i){
        long long a1b1 = P[i].a + 0.5;
        long long a2b1 = P[i].b + 0.5;
        long long a1b2 = Q[i].a + 0.5;
        long long a2b2 = Q[i].b + 0.5;

        long long w = ((a1b1 % mod * (BLOCK * BLOCK % mod)) + ((a2b1 + a1b2) % mod) * BLOCK + a2b2) % mod;

        if(i <= n + m)
            cout << w << " ";
    }

    return 0;
}
```
