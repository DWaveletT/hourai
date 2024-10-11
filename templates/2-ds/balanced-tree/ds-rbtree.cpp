#include<bits/stdc++.h>
using namespace std;
typedef long long i64;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

namespace RBT{
    #define BK 0
    #define RD 1 
    const int SIZ = 1e5 + 1e6 + 3;
    int sz, X[SIZ][2], C[SIZ], S[SIZ], W[SIZ], F[SIZ];
    bool H[SIZ];
    bool is_root(int x){ return   F[x]     == 0;}
    bool is_rson(int x){ return X[F[x]][1] == x;}
    void pushup(int t){
        S[t] = S[X[t][0]] + S[X[t][1]] + C[t];
    }
    int newnode(int w){
        ++ sz;
        X[sz][0] = 0, X[sz][1] = 0;
        C[sz] = S[sz] = 1;
        W[sz] =  w,
        H[sz] = RD;
        return sz; 
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
    void maintain1(int &root, int u){
        if(F[  u ] == 0) return H[  u ] = BK, void();   // Case 1
        if(H[F[u]] == 0) return;                        // Case 2
        int v = F[u], w = F[v];
        bool f = is_rson(u);
        bool g = is_rson(v);
        int x = X[w][!g];
        if(H[x] == RD){ // Case 3
            H[x] = BK, H[v] = BK, H[w] = RD;
            maintain1(root, w);
        } else {
            // Case 4 :
            if(f != g)
                rotate(root, u), swap(u, v), f = !f;

            // Case 5:
            rotate(root, v);
            H[w] = RD, H[v] = BK;
        }
    }
    void insert(int &root, int w){
        if(root == 0) {root = newnode(w), H[root] = BK; return;}
        int x = root, o = x;
        for(;x;o = x, x = X[x][w > W[x]]){
            ++ S[x]; if(w == W[x]){ ++ C[x], o = x; break;}
        }
        if(W[o] != w){
            if(w < W[o]) X[o][0] = newnode(w), F[sz] = o, o = sz;
            else         X[o][1] = newnode(w), F[sz] = o, o = sz;
            maintain1(root, o);
        }
    }
    void maintain2(int &root, int u){
        // Case 1 :
        if(F[u] == 0) return;

        int v = F[u]; bool f = is_rson(u);
        int h = X[v][!f];
        int a = X[h][ f];
        int b = X[h][!f];
        // Case 2 :
        if(H[a] == BK && H[b] == BK && H[h] == BK && H[v] == BK){
            H[h] = RD;
            maintain2(root, v);
            return;
        }
        // Case 3 :
        if(H[h] == RD){
            rotate(root, h);
            H[h] = BK;
            H[v] = RD;
            h = a;
            a = X[h][ f];
            b = X[h][!f];
        }
        // Case 4 :
        if(H[v] == RD && H[a] == BK && H[b] == BK){
            H[v] = BK;
            H[h] = RD;
            return;
        }
        // Case 5 :
        if(H[a] == RD && H[b] == BK){
            rotate(root, a);
            H[a] = BK;
            H[h] = RD;
            h = a;
            a = X[h][ f];
            b = X[h][!f];
        }
        // Case 6 :
        {
            rotate(root, h);
            swap(H[h], H[v]);
            H[b] = BK;
            return;
        }
    }
    void erase(int &root, int w){
        int sss = S[root];
        int x = root, o = x;
        for(;x;o = x, x = X[x][w > W[x]]){
            -- S[x]; if(w == W[x]){ -- C[x], o = x; break;}
        }
        if(C[o] == 0){
            if(X[o][0] != 0 && X[o][1] != 0){
                int y = X[o][1];
                while(X[y][0]) y = X[y][0];
                swap(C[o], C[y]);
                swap(W[o], W[y]);
                for(int p = y;p != o;p = F[p])
                    pushup(p);
                pushup(o), o = y;
            }
            if(X[o][0] == 0 && X[o][1] == 0){
                if(F[o] == 0) root = 0;
                else {
                    if(H[o] == BK)
                        maintain2(root, o);
                    X[F[o]][is_rson(o)] = 0;
                }
            } else {
                int s = X[o][0] ? X[o][0] : X[o][1];
                H[s] = BK;
                F[s] = F[o];
                if(F[o])
                    X[F[o]][is_rson(o)] = s,
                    pushup(F[o]);
                else
                    root = s;
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
        return a + 1;
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
        return W[x];
    }
    int  find_pre(int &root, int w){
        return find_kth(root, find_rank(root, w) - 1);
    }
    int  find_suc(int &root, int w){
        return find_kth(root, find_rank(root, w + 1));
    }
}
int qread();

int main(){
    using namespace RBT;
    int n = qread(), m = qread(), root = 0;
    for(int i = 1;i <= n;++ i){
        int a = qread(); insert(root, a);
    }
    int lastans = 0, ans = 0;
    for(int i = 1;i <= m;++ i){
        int op = qread(), x = qread() ^ lastans;
        switch(op){
            case 1 : insert(root, x); break;
            case 2 : erase (root, x); break;
            case 3 : ans ^= (lastans = find_rank(root, x)); break;
            case 4 : ans ^= (lastans = find_kth (root, x)); break;
            case 5 : ans ^= (lastans = find_pre (root, x)); break;
            case 6 : ans ^= (lastans = find_suc (root, x)); break;
        }
    }
    printf("%d\n", ans);
    return 0;
}