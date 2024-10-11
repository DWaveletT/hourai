```cpp
#include<bits/stdc++.h>
#define up(l, r, i) for(int i = l, END##i = r;i <= END##i;++ i)
#define dn(r, l, i) for(int i = r, END##i = l;i >= END##i;-- i)
using namespace std;
typedef long long i64;
const int INF = 2147483647;
const int MAXN= 1e5 + 3;
const int MAXM= (1 << 14) + 3;
int n, m, k, maxt = 16383, X[MAXM], C[MAXM], t;
const int BUF_SIZE = 1e6;
char *p1, *p2, BUF[BUF_SIZE];
inline char readc(){
    if(p1 == p2) p1 = BUF, p2 = BUF + fread(BUF, 1, BUF_SIZE, stdin);
    return p1 == p2 ? EOF : *p1 ++;
}
inline int qread(){
    int w = 1, c, ret;
    while((c = readc()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = readc()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}
int A[MAXN], bsize; i64 B[MAXN], R[MAXN];
struct Qry1{ int l, r, id; }O[MAXN];
struct Qry2{ int id, l, r; };
struct Qry3{ int id, l, r; };
bool cmp(Qry1 a, Qry1 b){
    return a.l / bsize == b.l / bsize ? a.r < b.r : a.l < b.l;
}
vector <Qry2> P[MAXN];
vector <Qry3> Q[MAXN];
int main(){
    n = qread(), m = qread(), k = qread(), bsize = sqrt(m + 1);
    up(1, n, i) A[i] = qread();
    up(1, m, i){
        int l = qread(), r = qread(); O[i] = {l, r, i};
    }
    sort(O + 1, O + 1 + m, cmp);
    int l = 1, r = 0;
    up(1, m, i){
        int p = O[i].l, q = O[i].r;
        if(r < q){
            P[r    ].push_back({ i, r + 1, q});
            Q[l - 1].push_back({-i, r + 1, q});
        }
        if(r > q){
            P[q    ].push_back({-i, q + 1, r});
            Q[l - 1].push_back({ i, q + 1, r});
        }
        r = q;
        if(l > p){
            P[p].push_back({-i, p, l - 1});
            Q[r].push_back({ i, p, l - 1});
        }
        if(l < p){
            P[l].push_back({ i, l, p - 1});
            Q[r].push_back({-i, l, p - 1});
        }
        l = p;
    }
    up(0, maxt, i) if(__builtin_popcount(i) == k) X[++ t] = i;
    up(0, n, i){
        up(1, t, j) ++ C[A[i] ^ X[j]];
        for(auto &o : P[i]){
            if(o.id > 0) R[ o.id] += C[A[o.l]];
            else         R[-o.id] -= C[A[o.l]];
            if(o.l < o.r)
                P[i + 1].push_back({o.id, o.l + 1, o.r});
        }
        for(auto &o : Q[i]){
            up(o.l, o.r, j){
                if(o.id > 0) R[ o.id] += C[A[j]];
                else         R[-o.id] -= C[A[j]];
            }
        }
        P[i].clear(), Q[i].clear();
        P[i].shrink_to_fit();
        Q[i].shrink_to_fit();
    }
    i64 ans = 0;
    up(1, m, i){ ans += R[i], B[O[i].id] = ans; }
    up(1, m, i) printf("%lld\n", B[i]);
    return 0;
}
```
