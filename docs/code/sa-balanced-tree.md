```cpp
#include<bits/stdc++.h>
#define up(l, r, i) for(int i = l, END##i = r;i <= END##i;++ i)
#define dn(r, l, i) for(int i = r, END##i = l;i >= END##i;-- i)
using namespace std;
typedef long long i64;
const int INF = 2147483647;
typedef unsigned int       u32;
typedef unsigned long long u64;
mt19937_64 MT(114514);

const int MAXN = 5e5 + 3;

struct Suffix {
    int i; char c; u64 h;
};
Suffix U[MAXN];

bool cmp(int a, int b){
    return U[a].c == U[b].c ? U[U[a].i - 1].h < U[U[b].i - 1].h : U[a].c < U[b].c;
}

namespace Treap{
    const int SIZ = 1e6 + 1e5 + 3;
    int F[SIZ], C[SIZ], S[SIZ], W[SIZ], X[SIZ][2], sz;
    u64 H[SIZ], L[SIZ], R[SIZ];
    bool is_root(int x){ return   F[x]     == 0;}
    bool is_rson(int x){ return X[F[x]][1] == x;}
    int  newnode(int w){
        W[++ sz] = w, C[sz] = S[sz] = 1; H[sz] = MT();
        return sz;
    }
    void pushup(int x){
        S[x] = C[x] + S[X[x][0]] + S[X[x][1]];
    }
    void pushdown(int x){
        if(L[x]){
            
        }
    }
    void rotate(int &root, int x){
        int y = F[x], z = F[y];
        bool f = is_rson(x);
        bool g = is_rson(y);
        int &t = X[x][!f];
        if(z){ X[z][g] = x; } else root = x;
        if(t){ F[t]    = y; }
        X[y][f] = t, t = y;
        F[y] = x, pushup(y);
        F[x] = z, pushup(x);
    }
    void insert(int &root, int w){
        if(root == 0) {root = newnode(w); return;}
        int x = root, o = x;
        for(;x;o = x, x = X[x][w > W[x]]){
            ++ S[x]; if(w == W[x]){ ++ C[x], o = x; break;}
        }
        if(W[o] != w){
            if(w < W[o]) X[o][0] = newnode(w), F[sz] = o, o = sz;
            else         X[o][1] = newnode(w), F[sz] = o, o = sz;
        }
        while(!is_root(o) && H[o] < H[F[o]])
            rotate(root, o);
    }
    void erase(int &root, int w){
        int x = root, o = x;
        for(;x;o = x, x = X[x][w > W[x]]){
            -- S[x]; if(w == W[x]){ -- C[x], o = x; break;}
        }
        if(C[o] == 0){
            while(X[o][0] || X[o][1]){
                u64 wl = X[o][0] ? H[X[o][0]] : ULLONG_MAX;
                u64 wr = X[o][1] ? H[X[o][1]] : ULLONG_MAX;
                if(wl < wr){
                    int p = X[o][0]; rotate(root, p);
                } else {
                    int p = X[o][1]; rotate(root, p);
                }
            }
            if(is_root(o)){
                root = 0;
            } else {
                X[F[o]][is_rson(o)] = 0;
            }
        }
    }
}
int main(){

    return 0;
}
```
