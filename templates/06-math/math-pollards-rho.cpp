/**
## 用法

- 调用 `test(n)` 判断 $n$ 是否是质数；
- 调用 `rho(n)` 计算 $n$ 分解质因数后的结果，不保证结果有序。
**/
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

i64 step(i64 a, i64 c, i64 m){
    return ((__int128)a * a + c) % m;
}
i64 multi(i64 a, i64 b, i64 m){
    return (__int128) a * b % m;
}
i64 power(i64 a, i64 b, i64 m){
    i64 r = 1;
    while(b){
        if(b & 1) r = multi(r, a, m);
        b >>= 1,  a = multi(a, a, m);
    }
    return r;
}
mt19937_64 MT;
bool test(i64 n){
    if(n < 3 || n % 2 == 0)
        return n == 2;
    i64 u = n - 1, t = 0;
    while(u % 2 == 0)
        u /= 2,
        t += 1;
    int test_time = 20;
    for(int i = 1; i <= test_time;++ i){
        i64 a = MT() % (n - 2) + 2;
        i64 v = power(a, u, n);
        if(v == 1){
            continue;
        }
        int s;
        for(s = 0;s < t;++ s){
            if(v == n - 1)
                break;
            v = multi(v, v, n);
        }
        if(s == t)
            return false;
    }
    return true;
}
basic_string<i64> rho(i64 n){
    if(n == 1)
        return {};
    if(test(n)){
        return {n};
    }
    i64 a  = MT() % (n - 1) + 1;
    i64 x1 = MT() % (n - 1);
    i64 x2 = x1;
    for(int i = 1;;i <<= 1){
        i64 tot = 1;
        for(int j = 1;j <= i;++ j){
            x2 = step(x2, a, n);
            tot = multi(tot, llabs(x1 - x2), n);
            if(j % 127 == 0){
                i64 d = __gcd(tot, n);
                if(d > 1)
                    return rho(d) + rho(n / d);
            }
        }
        i64 d = __gcd(tot, n);
        if(d > 1)
            return rho(d) + rho(n / d);
        x1 = x2;
    }
}

// ===== TEST =====

int main(){
    int T;
    cin >> T;
    for(int _ = 1;_ <= T;++ _){
        i64 n, p = 0;
        cin >> n;
        auto res = rho(n);
        for(auto &u : res)
            p = max(p, u);
        if(res.size() == 1)
            cout << "Prime" << endl;
        else 
            cout << p << endl;
    }
    return 0;
}