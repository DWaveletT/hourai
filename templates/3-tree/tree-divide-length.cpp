#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN= 5e5 + 3;
const int MAXM=  19 + 3;
vector <int> P[MAXN];
vector <int> Q[MAXN];
vector <int> E[MAXN];
int h = 19;
int L[MAXN], F[MAXN], G[MAXN], D[MAXN], S[MAXM][MAXN];
void dfs1(int u, int f){
    L[u] = 1, S[0][u] = f;
    F[u] = f, D[u] = D[f] + 1;
    for(int i = 1;i <= h;++ i)
        S[i][u] = S[i - 1][S[i - 1][u]];
    for(auto &v : E[u]) if(v != f){
        dfs1(v, u);
        if(L[v] > L[G[u]])
            G[u] = v;
        L[u] = max(L[u], L[v] + 1);
    }
}
int T[MAXN];
void dfs2(int u, int f){
    if(u == G[f]){
        T[u] = T[f];
        P[T[u]].push_back(u);
        Q[T[u]].push_back(F[Q[T[u]].back()]);
    } else {
        T[u] = u;
        P[u].push_back(u);
        Q[u].push_back(u);
    }
    if(G[u]) dfs2(G[u], u);
    for(auto &v : E[u]) if(v != f && v != G[u])
        dfs2(v, u);
}
typedef unsigned int       u32;
typedef unsigned long long u64;
int n, q; u32 s;
u32 get(u32 x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return s = x; 
}
int qread(){
    int w = 1, c, ret;
    while((c = getchar()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}
int H[MAXN];
int main(){
    scanf("%d%d%u", &n, &q, &s);
    int root = 0; H[0] = -1;
    for(int i = 1;i <= n;++ i){
        int f = qread();
        if(f == 0)
            root = i;
        else {
            E[f].push_back(i);
            E[i].push_back(f);
        }
        H[i] = H[i >> 1] + 1;
    }
    dfs1(root, 0);
    dfs2(root, 0);
    int lastans = 0;
    i64 realans = 0;
    for(int i = 1;i <= q;++ i){
        int x = (get(s) ^ lastans) % n + 1;
        int k = (get(s) ^ lastans) % D[x];
        
        if(k == 0){
            lastans = x;
        } else {
            int h = H[k];
            k -= 1 << h;
            x = S[h][x];
            int t = T[x];
            k -= D[x] - D[t];
            if(k > 0){
                x = Q[t][k];
            } else {
                x = P[t][-k];
            }
            lastans = x;
        }
        realans ^= 1ll * i * lastans;
        
    }
    printf("%lld\n", realans);
    return 0;
}