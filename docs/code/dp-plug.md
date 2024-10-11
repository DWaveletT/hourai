```cpp
#include<bits/stdc++.h>
#define up(l, r, i) for(int i = l, END##i = r;i <= END##i;++ i)
#define dn(r, l, i) for(int i = r, END##i = l;i >= END##i;-- i)
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN = 20 + 3;
const int MAXM = 67108864 + 3;

namespace HashT{
    const int SIZ = 19999997;
    int H[SIZ], V[SIZ], N[SIZ], t;
    bool F[SIZ];
    i64  W[SIZ];
    void add(int u, int v, bool f, i64 w){
        V[++ t] = v, N[t] = H[u], F[t] = f, W[t] = w, H[u] = t;
    }
    i64& find(int u, bool f){
        for(int p = H[u % SIZ];p;p = N[p])
            if(V[p] == u && F[p] == f)
                return W[p];
        add(u % SIZ, u, f, 0);
        return W[t];
    }
}
char S[MAXN][MAXN];
int qread(){
    int w = 1, c, ret;
    while((c = getchar()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}
int n, m;
vector <pair<pair<int, bool>, i64> > M[2];
int getp(int s, int p){
    return (s >> (2 * p - 2)) & 3;
}
int setw(int s, int p, int w){
    return (s & ~(3 << (2 * p - 2))) | (w << (2 * p - 2));
}
int findr(int s, int p){
    int c = 0;
    for(int q = p;q <= m + 1;++ q){
        if(((s >> (2 * q - 2)) & 3) == 1) ++ c;
        if(((s >> (2 * q - 2)) & 3) == 2) -- c;
        if(c == 0)
            return q;
    }
    return -1;
}
int findl(int s, int p){
    int c = 0;
    for(int q = p;q >= 1;-- q){
        if(((s >> (2 * q - 2)) & 3) == 2) ++ c;
        if(((s >> (2 * q - 2)) & 3) == 1) -- c;
        if(c == 0)
            return q;
    }
    return -1;
}
void state(int s){
    return ;
    up(1, m + 1, i){
        switch(getp(s, i)){
            case 0 : putchar('#'); break;
            case 1 : putchar('('); break;
            case 2 : putchar(')'); break;
            case 3 : putchar('E');
        }
    }
    puts("");
}
int main(){
    n = qread(), m = qread();
    up(1, n, i)
        scanf("%s", S[i] + 1);
    int o = 0;
    #define X M[ o]
    #define Y M[!o]
    vector <pair<int, bool> > T;
    X.push_back({{0, 0}, 1});
    up(1, n, i){
        Y.clear();
        for(auto &u : X){
            auto [s0, c] = u;
            auto [s, f] = s0;
            if(getp(s, m + 1) == 0)
                Y.push_back({{s << 2, f}, c});
        }
        o ^= 1;
        up(1, m, j){
            int x = j, y = j + 1;
            for(auto &u : X){
                auto [s0, c] = u;
                auto [s, f] = s0;
                int a = getp(s, x);
                int b = getp(s, y);
                int t = setw(setw(s, x, 0), y, 0);
                #define update(t, c) HashT :: find(t, f) += c, T.push_back({t, f})
                if(S[i][j] == '.'){     // 经过该格
                    if(a == 1 && b == 1){
                        t = setw(t, findr(s, y), 1),
                        update(t, c);
                    } else
                    if(a == 2 && b == 2){
                        t = setw(t, findl(s, x), 2),
                        update(t, c);
                    } else 
                    if(a == 1 && b == 2){
                        if(f == false)  // 还没有闭合回路
                            f = true, update(t, c);
                    } else
                    if(a == 2 && b == 1){
                        update(t, c);
                    } else
                    if(a == 0 && b == 0){
                        t = setw(t, x, 1);
                        t = setw(t, y, 2);
                        update(t, c);
                    } else {    // a == 0 || b == 0
                        int t1 = setw(t, x, a | b);
                        int t2 = setw(t, y, a | b);
                        update(t1, c);
                        update(t2, c);
                    }
                }
                if(S[i][j] == '*'){ // 不经过该格
                    if(a == 0 && b == 0)
                        update(t, c);
                }
            }
            Y.clear();
            for(auto &u : T){
                auto [s, f] = u;
                if(HashT :: find(s, f) != 0){
                    Y.push_back({{s, f}, HashT :: find(s, f)});
                    HashT :: find(s, f) = 0;
                }
            }
            T.clear(), o ^= 1;
        }
    }
    i64 ans = 0;
    for(auto &u : X){
        auto [s0, c] = u;
        auto [s, f] = s0;
        bool g = true;
        up(1, m + 1, i)
            g &= getp(s, i) == 0;
        f &= g;
        if(f)
            ans = c;
    }
    printf("%lld\n", ans);
    return 0;
}
```
