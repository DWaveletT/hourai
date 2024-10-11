```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long i64;

struct Line{ int id; double k, b; Line() = default;};
namespace LCSeg{
    const int SIZ = 2e5 + 3;
    struct Line T[SIZ];
    #define lc(t) (t << 1)
    #define rc(t) (t << 1 | 1)
    bool cmp(int p, Line x, Line y){
        double w1 = x.k * p + x.b;
        double w2 = y.k * p + y.b;
        double d = w1 - w2;
        if(fabs(d) < 1e-8) return x.id > y.id;
        return d < 0;
    }
    void merge(int t, int a, int b, Line x, Line y){
        int c = a + b >> 1;
        if(cmp(c, x, y)) swap(x, y);
        if(cmp(a, y, x)){
            T[t] = x; if(a != b) merge(rc(t), c + 1, b, T[rc(t)], y);
        } else {
            T[t] = x; if(a != b) merge(lc(t), a,     c, T[lc(t)], y);
        }
    }
    void modify(int t, int a, int b, int l, int r, Line x){
        if(l <= a && b <= r) merge(t, a, b, T[t], x);
        else {
            int c = a + b >> 1;
            if(l <= c) modify(lc(t), a,     c, l, r, x);
            if(r >  c) modify(rc(t), c + 1, b, l, r, x);
        }
    }
    void query(int t, int a, int b, int p, Line &x){
        if(cmp(p, x, T[t])) x = T[t];
        if(a != b){
            int c = a + b >> 1;
            if(p <= c) query(lc(t), a,     c, p, x);
            if(p >  c) query(rc(t), c + 1, b, p, x);
        }
    }
}
const int MOD1 = 39989;
const int MOD2 = 1e9;
int qread();
int m = 39989, o;
int main(){
    int n = qread(), last_ans = 0;
    for(int i = 1;i <= n;++ i){
        int op = qread(); if(op == 0){
            int k = (qread() + last_ans - 1) % MOD1 + 1;
            Line x = {0, 0, 0}; LCSeg :: query(1, 1, m, k, x);
            printf("%d\n", last_ans = x.id);
        } else {
            int _x1 = (qread() + last_ans - 1) % MOD1 + 1;
            int _y1 = (qread() + last_ans - 1) % MOD2 + 1;
            int _x2 = (qread() + last_ans - 1) % MOD1 + 1;
            int _y2 = (qread() + last_ans - 1) % MOD2 + 1;
            if(_x1 > _x2) swap(_x1, _x2), swap(_y1, _y2);
            double k, b; int d = ++ o;
            if(_x1 == _x2) k = 0, b = max(_y1, _y2);
                else k = 1.0 * (_y2 - _y1) / (_x2 - _x1), b = _y1 - k * _x1;
            Line x = {d, k, b}; LCSeg :: modify(1, 1, m, _x1, _x2, x);
        }
    }
    return 0;
}
```
