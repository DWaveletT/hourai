```cpp
#include<bits/stdc++.h>

using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

namespace Splay{
    const int SIZ = 1e6 + 1e5 + 3;
    int F[SIZ], C[SIZ], S[SIZ], W[SIZ], X[SIZ][2], sz;
    bool is_root(int x){ return   F[x]     == 0;}
    bool is_rson(int x){ return X[F[x]][1] == x;}
    int  newnode(int w){
        W[++ sz] = w, C[sz] = S[sz] = 1, F[sz] = 0;
        return sz;
    }
    void pushup(int x){
        S[x] = C[x] + S[X[x][0]] + S[X[x][1]];
    }
    void cut(int x, int y){
        X[x][is_rson(y)] = 0;
        F[y] = 0;
    }
    void link(int x, int y, bool isr){
        X[x][isr] = y;
        F[y] = x;
    }
    void rotate(int x){
        int y = F[x], z = F[y];
        bool f = is_rson(x);
        bool g = is_rson(y);
        int &t = X[x][!f];
        if(z){ X[z][g] = x; }
        if(t){ F[t]    = y; }
        X[y][f] = t, t = y;
        F[y] = x, pushup(y);
        F[x] = z, pushup(x);
    }
    void splay(int &root, int x){
        for(int f = F[x];f = F[x], f;rotate(x))
            if(F[f]) rotate(is_rson(x) == is_rson(f) ? f : x);
        root = x;
    }
    void insert(int &root, int w){
        if(root == 0) {root = newnode(w); return;}
        int x = root, o = x;
        for(;x;o = x, x = X[x][w > W[x]]){
            ++ S[x]; if(w == W[x]){ ++ C[x], o = x; break;}
        }
        if(W[o] != w){
            if(w < W[o]) X[o][0] = newnode(w), link(o, sz, 0);
            else         X[o][1] = newnode(w), link(o, sz, 1);
        }
        splay(root, o);
    }
    void erase(int &root, int w){
        int val = S[root];
        int x = root, o = x;
        for(;x;o = x, x = X[x][w > W[x]]){
            -- S[x]; if(w == W[x]){ -- C[x], o = x; break;}
        }
        splay(root, o);
        if(C[o] == 0){
            if(X[o][0] == 0 || X[o][1] == 0){
                int u = X[o][0] | X[o][1];
                if(u != 0) F[root = u] = 0;
            } else {
                int l = X[x][0];
                int r = X[x][1];
                cut(o, l), cut(o, r);
                int p = l;
                while(X[p][1]) p = X[p][1];
                splay(l, p), link(l, r, 1);
                root = l;
            }
        }
    }
    int  find_rank(int &root, int w){
        int x = root, o = x, a = 0;
        for(;x;){
            if(w <  W[x])
                o = x, x = X[x][0];
            else {
                a += S[X[x][0]];
                if(w == W[x]){
                    o = x; break;
                }
                a += C[x];
                o = x, x = X[x][1];
            }
        }
        splay(root, o); return a + 1;
    }
    int  find_kth(int &root, int w){
        int x = root, o = x, a = 0;
        for(;x;){
            if(w <= S[X[x][0]])
                o = x, x = X[x][0];
            else {
                w -= S[X[x][0]];
                if(w <= C[x]){
                    o = x; break;
                }
                w -= C[x];
                o = x, x = X[x][1];
            } 
        }
        splay(root, o); return W[x];
    }
    int  find_pre(int &root, int w){
        return find_kth(root, find_rank(root, w) - 1);
    }
    int  find_suc(int &root, int w){
        return find_kth(root, find_rank(root, w + 1));
    }
}

// ===== TEST =====

int qread(){
    int w=1,c,ret;
    while((c = getchar()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}
int main(){
    using namespace Splay;
    int n = qread(), m = qread(), root = 0;
    for(int i = 1;i <= n;++ i){
        int a = qread(); insert(root, a);
    }
    int last_ans = 0, ans = 0;
    for(int i = 1;i <= m;++ i){
        int op = qread(), x = qread() ^ last_ans;
        switch(op){
            case 1 : insert(root, x); break;
            case 2 : erase (root, x); break;
            case 3 : ans ^= (last_ans = find_rank(root, x)); break;
            case 4 : ans ^= (last_ans = find_kth (root, x)); break;
            case 5 : ans ^= (last_ans = find_pre (root, x)); break;
            case 6 : ans ^= (last_ans = find_suc (root, x)); break;
        }
    }
    printf("%d\n", ans);
    return 0;
}
```
