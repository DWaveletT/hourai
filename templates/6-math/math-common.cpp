#include<bits/stdc++.h>
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

int power(int a, int b, int m){
    int r = 1;
    while(b){
        if(b & 1) r = 1ll * r * a % m;
        b >>= 1,  a = 1ll * a * a % m;
    }
    return r;
}

int multi(int a, int b, int m){
    int r = 1;
    while(b){
        if(b & 1) r = (r + a) % m;
        b >>= 1,  a = (a + a) % m;
    }
    return r;
}

int exgcd(int a, int b, int &x, int &y){
    if(a == 0){
        x = 0, y = 1; return b;
    } else {
        int x0 = 0, y0 = 0;
        int d = exgcd(b % a, a, x0, y0);
        x = y0 - (b / a) * x0;
        y = x0;
        return d;
    }
}

void inv(int n, int T[]){
    T[1] = 1;
    for(int i = 2;i <= n;++ i)
        T[i] = 1ll * (MOD - MOD / i) * T[MOD % i] % MOD;
}