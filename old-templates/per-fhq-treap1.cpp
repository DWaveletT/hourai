#include<bits/stdc++.h>
using namespace std;
typedef long long i64;
typedef unsigned int       u32;
typedef unsigned long long u64;
const int INF = 2147483647;
mt19937_64 MT(114514);
const int MAXN = 5e5 + 3;
int R[MAXN];
namespace FhqTreap{
    struct Pair3{ int a, b, c; };
    #define lc(x) X[x][0]
    #define rc(x) X[x][1]
    const int SIZ = 5e7 + 3;
    int C[SIZ], S[SIZ], W[SIZ], X[SIZ][2], size;
    u64 O[SIZ];
    int  new_node(int w){
        ++ size, S[size] = C[size] = 1, W[size] = w, O[size] = MT();
        return size;
    }
    void push_down(int x){
        
    }
    void push_up(int x){
        S[x] = C[x] + S[lc(x)] + S[rc(x)];
    }
    int  clone(int x){
        int p = ++ size;
        C[p] = C[x], S[p] = S[x], W[p] = W[x], O[p] = O[x];
        X[p][0] = X[x][0];
        X[p][1] = X[x][1];
        return p;
    }
    int  merge(int x, int y){
        if(x == 0 || y == 0) return x | y;
        if(O[x] < O[y]){
            int _y = clone(y);
            lc(_y) = merge(x, lc(_y)), push_up(_y); return _y;
        } else {
            int _x = clone(x);
            rc(_x) = merge(rc(_x), y), push_up(_x); return _x;
        }
    }
    Pair3 split_by_val(int x, int w){
        if(x == 0) return {0, 0, 0}; int _x = clone(x);
        if(W[x] <  w){
            Pair3 p = split_by_val(rc(_x), w);
            rc(_x) = p.a, push_up(_x); return {_x, p.b, p.c};
        } else if(W[x] == w){
            Pair3 ret = {lc(_x), _x, rc(_x)};
            lc(_x) = rc(_x) = 0, push_up(_x); return ret;
        } else {
            Pair3 p = split_by_val(lc(_x), w);
            lc(_x) = p.c, push_up(_x); return {p.a, p.b, _x};
        }
    }
    Pair3 split_by_rnk(int x, int w){
        if(x == 0) return {0, 0, 0}; int _x = clone(x);
        if(S[lc(x)] + C[x] < w){
            Pair3 p = split_by_rnk(rc(_x), w - S[lc(_x)] - C[_x]);
            rc(_x) = p.a, push_up(_x); return {_x, p.b, p.c};
        } else if(S[lc(x)] < w){
            Pair3 ret = {lc(_x), _x, rc(_x)};
            lc(_x) = rc(_x) = 0; return ret;
        } else {
            Pair3 p = split_by_rnk(lc(_x), w);
            lc(_x) = p.c, push_up(_x); return {p.a, p.b, _x};
        }
    }
    void insert(int &r, int w){
        if(r == 0){r = new_node(w); return;}
        Pair3 u = split_by_val(r, w); int p = 0;
        if(u.b) p = clone(u.b), ++ C[p], ++ S[p];
            else p = new_node(w);
        r = merge(merge(u.a, p), u.c);
    }
    void erase(int &r, int w){
        Pair3 u = split_by_val(r, w); int p = 0;
        if(u.b) p = clone(u.b), p = u.b, -- C[p], -- S[p];
            else p = 0;
        r = merge(merge(u.a, p), u.c);
    }
    int  get_rnk(int &r, int w){
        Pair3 u = split_by_val(r, w); int ret = S[u.a] + 1;
        r = merge(merge(u.a, u.b), u.c); return ret;
    }
    int  get_kth(int &r, int w){
        Pair3 u = split_by_rnk(r, w); int ret = W[u.b];
        r = merge(merge(u.a, u.b), u.c); return ret;
    }
    int  get_pre(int &r, int w){
        int t = get_rnk(r, w) - 1; return get_kth(r, t);
    }
    int  get_suc(int &r, int w){
        int t = get_rnk(r, w + 1); return get_kth(r, t);
    }
}
int main(){ // 可持久化平衡树
    using namespace FhqTreap;
    insert(R[0], -INF), insert(R[0], INF);
    int n;
    cin >> n;
    for(int i = 1;i <= n;++ i){
        int ver, op, x;
        cin >> ver >> op >> x;
        R[i] = clone(R[ver]);
        switch(op){
            case 1 : insert(R[i], x); break;
            case 2 : erase (R[i], x); break;
            case 3 : cout << (get_rnk(R[i], x) - 1) << endl; break;
            case 4 : cout << (get_kth(R[i], x + 1)) << endl; break;
            case 5 : cout << (get_pre(R[i], x)) << endl; break;
            case 6 : cout << (get_suc(R[i], x)) << endl; break;
        }
    }
    return 0;
}