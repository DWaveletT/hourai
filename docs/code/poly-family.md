## 用法

多项式全家桶。

- 包含基础多项式算法：快速傅里叶变换（`FFT`）及其逆变换（`IFFT`）、快速数论变换（`NTT`）及其逆变换（`INTT`）；
- 包含基于 NTT 的扩展多项式算法：多项式乘法（`MUL`）、多项式乘法逆元（`INV`）、多项式微分（`DIF`）、多项式积分（`INT`）、多项式对数（`LN`）、多项式指数（`EXP`）、多项式开根（`SQT`）、多项式平移（即计算 $G(x) = F(x + c)$，`SHF`）。

```cpp
#include<bits/stdc++.h>
#define up(l, r, i) for(int i = l, END##i = r;i <= END##i;++ i)
#define dn(r, l, i) for(int i = r, END##i = l;i >= END##i;-- i)
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MOD = 998244353;

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

const int MAX_ = (1 << 19) + 3;
struct cplx{
    double a, b; cplx(double _a = 0, double _b = 0) :a(_a), b(_b){}
    cplx operator +(cplx t){ return cplx(a + t.a, b + t.b); }
    cplx operator -(cplx t){ return cplx(a - t.a, b - t.b); }
    cplx operator *(cplx t){ return cplx(a * t.a - b * t.b, a * t.b + b * t.a); }
    cplx operator *(int t) { return cplx(a * t, b * t); }
};

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
            cplx* S = Z, o(cos(pi / l), sin(pi / l));
            up(0, n - 1, i){
                cplx s(1, 0);
                up(0, l - 1, j){
                    cplx x = S[j] + s * S[j + l];
                    cplx y = S[j] - s * S[j + l];
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
    void NTT(int n, int Z[]){
        static int W[MAX_];
        int g = 3, l = 1;
        W[0] = 0;
        while (n >>= 1)
            up(0, l - 1, i)
                W[l++] = W[i] << 1 | 1, W[i] <<= 1;
        up(0, l - 1, i)
            if (W[i] > i)swap(Z[i], Z[W[i]]);
        for (n = l >> 1, l = 1;n;n >>= 1, l <<= 1){
            int* S = Z, o = power(g, (MOD - 1) / l / 2);
            up(0, n - 1, i){
                int s = 1;
                up(0, l - 1, j){
                    int x = (S[j] + 1ll * s * S[j + l] % MOD      ) % MOD;
                    int y = (S[j] - 1ll * s * S[j + l] % MOD + MOD) % MOD;
                    S[j] = x, S[j + l] = y;
                    s = 1ll * s * o % MOD;
                }
                S += l << 1;
            }
        }
    }
    void INTT(int n, int Z[]){
        NTT(n, Z); reverse(Z + 1, Z + n);
        int o = inv(n);
        up(0, n - 1, i)
            Z[i] = 1ll * Z[i] * o % MOD;
    }
    void MUL(int n, int A[], int B[]){          // 乘法
        NTT(n, A), NTT(n, B);
        up(0, n - 1, i)
            A[i] = 1ll * A[i] * B[i] % MOD;
        INTT(n, A);
    }
    void INV(int n, int Z[], int T[]){          // 乘法逆
        static int A[MAX_];
        up(0, n - 1, i)
            T[i] = 0;
        T[0] = power(Z[0], MOD - 2);
        for (int l = 1;l < n;l <<= 1){
            up(    0, 2 * l - 1, i) A[i] = Z[i];
            up(2 * l, 4 * l - 1, i) A[i] = 0;
            NTT(4 * l, A), NTT(4 * l, T);
            up(0, 4 * l - 1, i)
                T[i] = (2ll * T[i] - 1ll * A[i] * T[i] % MOD * T[i] % MOD + MOD) % MOD;
            INTT(4 * l, T);
            up(2 * l, 4 * l - 1, i)
                T[i] = 0;
        }
    }
    void DIF(int n, int Z[], int T[]){          // 微分
        up(0, n - 2, i)
            T[i] = 1ll * Z[i + 1] * (i + 1) % MOD;
        T[n - 1] = 0;
    }
    void INT(int n, int c, int Z[], int T[]){   // 积分
        up(1, n - 1, i)
            T[i] = 1ll * Z[i - 1] * inv(i) % MOD;
        T[0] = c;
    }
    void LN(int n, int* Z, int* T){           // 求对数
        static int A[MAX_];
        static int B[MAX_];
        up(0, 2 * n - 1, i)
            A[i] = B[i] = 0;
        DIF(n, Z, A);
        INV(n, Z, B);
        MUL(2 * n, A, B);
        INT(n, 0, A, T);
    }
    void EXP(int n, int* Z, int* T){          // 求指数
        static int A[MAX_];
        static int B[MAX_];
        up(1, 2 * n - 1, i) T[i] = 0;
        T[0] = 1;
        for (int l = 1;l < n;l <<= 1){
            LN (2 * l, T, A);
            up(    0, 2 * l - 1, i)
                B[i] = (-A[i] + Z[i] + MOD) % MOD;
            B[0] = (B[0] + 1) % MOD;
            up(2 * l, 4 * l - 1, i)
                T[i] = B[i] = 0;
            MUL(4 * l, T, B);
        }
    }
    void SQT(int n, int* Z, int* T){          // 开根
        static int A[MAX_];
        static int B[MAX_];
        up(1, 2 * n - 1, i) T[i] = 0;
        T[0] = 1;
        int o = inv(2);
        for (int l = 1;l < n;l <<= 1){
            INV(2 * l, T, A);
            up(0, 2 * l - 1, i)
                B[i] = Z[i];
            up(2 * l, 4 * l - 1, i)
                A[i] = B[i] = 0;
            MUL(4 * l, A, B);
            up(0, 2 * l - 1, i)
                T[i] = 1ll * (T[i] + A[i]) * o % MOD;
        }
    }
    void SHF(int n, int c, int* Z, int* T){   // 平移
        static int A[MAX_];
        static int B[MAX_];
        static int F[MAX_];
        static int G[MAX_];
        int o = 1;
        up(1, n - 1, i)
            F[i] = 1ll * F[i - 1] *     i  % MOD,
            G[i] = 1ll * G[i - 1] * inv(i) % MOD;
        up(0, n - 1, i)
            A[i] = 1ll * Z[n - 1 - i] * F[n - 1 - i] % MOD;
        up(0, n - 1, i){
            B[i] = 1ll * G[i] * o % MOD;
            o = 1ll * o * c % MOD;
        }
        int l = 1; while (l < 2 * n - 1) l <<= 1;
        up(n, l - 1, i)
            A[i] = B[i] = 0; 
        MUL(l, A, B);
        up(0, n - 1, i)
            T[n - 1 - i] = 1ll * G[n - 1 - i] * A[i] % MOD;
    }
}
```
