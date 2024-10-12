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
    struct Pair2{ int a, b;    };
    #define lc(x) X[x][0]
    #define rc(x) X[x][1]
    const int SIZ = 5e7 + 3;
    int C[SIZ], S[SIZ], W[SIZ], X[SIZ][2], size; bool T[SIZ];
    u64 O[SIZ]; i64 H[SIZ];
    int  new_node(int w){
        ++ size,
        S[size] = C[size] = 1;
        W[size] = H[size] = w;
        O[size] = MT();
        return size;
    }
    int  clone(int x){
        if(x == 0) return 0;
        int p = ++ size;
        C[p] = C[x], S[p] = S[x];
        W[p] = W[x], H[p] = H[x];
        T[p] = T[x], O[p] = O[x];
        X[p][0] = X[x][0];
        X[p][1] = X[x][1];
        return p;
    }
    void push_down(int x){
        if(!T[x]) return;
        int _l = clone(lc(x)); if(_l) swap(lc(_l), rc(_l)), T[_l] ^= 1;
        int _r = clone(rc(x)); if(_r) swap(lc(_r), rc(_r)), T[_r] ^= 1;
        lc(x) = _l, rc(x) = _r, T[x] = 0;
    }
    void push_up(int x){
        S[x] = C[x] + S[lc(x)] + S[rc(x)];
        H[x] = W[x] + H[lc(x)] + H[rc(x)];
    }
    int  merge(int x, int y){
        if(x == 0 || y == 0) return x | y;
        if(O[x] < O[y]){
            int _y = clone(y); push_down(_y);
            lc(_y) = merge(x, lc(_y)), push_up(_y); return _y;
        } else {
            int _x = clone(x); push_down(_x);
            rc(_x) = merge(rc(_x), y), push_up(_x); return _x;
        }
    }
    Pair2 split_by_rnk(int x, int w){
        if(x == 0) return {0, 0}; int _x = clone(x); push_down(_x);
        if(S[lc(_x)] + 1 < w){
            Pair2 p = split_by_rnk(rc(_x), w - S[lc(_x)] - 1);
            rc(_x) = p.a; push_up(_x); return {_x, p.b};
        } else if(S[lc(_x)] + 1 == w){
            int t = rc(_x);
            rc(_x) =   0; push_up(_x); return {_x,   t};
        } else {
            Pair2 p = split_by_rnk(lc(_x), w);
            lc(_x) = p.b; push_up(_x); return {p.a, _x};
        }
    }
    void insert(int &r, int p, int w){
        Pair2 u = split_by_rnk(r, p);
        int t = new_node(w);
        r = merge(merge(u.a, t), u.b);
    }
    void erase(int &r, int p){
        Pair2 u = split_by_rnk(r, p - 1);
        Pair2 v = split_by_rnk(u.b, 1);
        r = merge(u.a, v.b);
    }
    void reverse(int &r, int a, int b){
        int l = b - a + 1;
        Pair2 u = split_by_rnk(  r, a - 1);
        Pair2 v = split_by_rnk(u.b, l    );
        int t = clone(v.a); T[t] ^= 1, swap(lc(t), rc(t));
        r = merge(merge(u.a, t), v.b);
    }
    i64  query(int &r, int a, int b){
        int l = b - a + 1;
        Pair2 u = split_by_rnk(  r, a - 1);
        Pair2 v = split_by_rnk(u.b, l    );
        i64 ret = H[v.a];
        r = merge(merge(u.a, v.a), v.b);
        return ret;
    }
}
int main(){ // 可持久化文艺平衡树
    ios :: sync_with_stdio(false);
    cin.tie(nullptr);
    
    using namespace FhqTreap;
    int n; cin >> n; i64 last_ans = 0;
    for(int i = 1;i <= n;++ i){
        int ver, op;
        cin >> ver >> op;
        R[i] = R[ver] ? clone(R[ver]) : 0;
        if(op == 1){        // 在 p 后面插入 w
            i64 p, w; cin >> p >> w;
            p ^= last_ans;
            w ^= last_ans;
            insert(R[i], p, w);
        } else if(op == 2){ // 删除第 p 个数
            i64 p; cin >> p;
            p ^= last_ans;
            erase(R[i], p);
        } else if(op == 3){ // 区间翻转
            i64 l, r; cin >> l >> r;
            l ^= last_ans;
            r ^= last_ans;
            reverse(R[i], l, r);
        } else {            // 查询区间和
            i64 l, r; cin >> l >> r;
            l ^= last_ans;
            r ^= last_ans;
            last_ans = query(R[i], l, r);
            cout << last_ans << "\n";
        }
    }
    return 0;
}