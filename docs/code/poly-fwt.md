```cpp
#include<bits/stdc++.h>
#define up(l, r, i) for(int i = l, END##i = r;i <= END##i;++ i)
#define dn(r, l, i) for(int i = r, END##i = l;i >= END##i;-- i)
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MOD = 998244353;

namespace Solve1{   // and 卷积
    void FWT(int n, int *A){
        for(int l = 1 << n, u = 2, v = 1;u <= l;u <<= 1, v <<= 1)
            for(int j = 0;j < l;j += u)
                for(int k = 0;k < v;++ k)
                    A[j + v + k] = (A[j + v + k] + A[j + k]) % MOD;
    }
    void IFWT(int n, int *A){
        for(int l = 1 << n, u = l, v = l / 2;u > 1;u >>= 1, v >>= 1)
            for(int j = 0;j < l;j += u)
                for(int k = 0;k < v;++ k)
                    A[j + v + k] = (A[j + v + k] - A[j + k] + MOD) % MOD;
    }
}
namespace Solve2{   // or  卷积
    void FWT(int n, int *A){
        for(int l = 1 << n, u = 2, v = 1;u <= l;u <<= 1, v <<= 1)
            for(int j = 0;j < l;j += u)
                for(int k = 0;k < v;++ k)
                    A[j + k] = (A[j + k] + A[j + v + k]) % MOD;
    }
    void IFWT(int n, int *A){
        for(int l = 1 << n, u = l, v = l / 2;u > 1;u >>= 1, v >>= 1)
            for(int j = 0;j < l;j += u)
                for(int k = 0;k < v;++ k)
                    A[j + k] = (A[j + k] - A[j + v + k] + MOD) % MOD;
    }
}
namespace Solve3{   // xor 卷积
    void FWT(int n, int *A){
        for(int l = 1 << n, u = 2, v = 1;u <= l;u <<= 1, v <<= 1)
            for(int j = 0;j < l;j += u)
                for(int k = 0;k < v;++ k){
                    int a = A[j + k];
                    int b = A[j + v + k];
                    A[j + k    ] = (a + b + MOD) % MOD;
                    A[j + v + k] = (a - b + MOD) % MOD;
                }
    }
    void IFWT(int n, int *A){
        int div2 = (MOD + 1) / 2;
        for(int l = 1 << n, u = l, v = l / 2;u > 1;u >>= 1, v >>= 1)
            for(int j = 0;j < l;j += u)
                for(int k = 0;k < v;++ k){
                    int a = A[j + k];
                    int b = A[j + v + k];
                    A[j + k    ] = 1ll * (a + b + MOD) * div2 % MOD;
                    A[j + v + k] = 1ll * (a - b + MOD) * div2 % MOD;
                }
    }
}
```
