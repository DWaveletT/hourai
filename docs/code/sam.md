```cpp
#include<bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXM= 26 + 3;
namespace SAM{
    const int SIZ = 2e6 + 3;
    int M[SIZ][MAXM];
    int L[SIZ], F[SIZ], S[SIZ];
    int last = 0, s = 0, h = 25;
    void init(){
        F[0] = -1, last = s = 0;
    }
    void extend(char c){
        int cur = ++ s, e = c - 'a';
        L[cur] = L[last] + 1;
        S[cur] = 1;
        int p = last;
        while(p != -1 && !M[p][e])
            M[p][e] = cur, p = F[p];
        if(p == -1){
            F[cur] = 0;
        } else {
            int q = M[p][e];
            if(L[p] + 1 == L[q]){
                F[cur] = q;
            } else {
                int clone = ++ s;
                L[clone] = L[p] + 1;
                F[clone] = F[q];
                S[clone] = 0;
                for(int i = 0;i <= h;++ i)
                    M[clone][i] = M[q][i];
                while(p != -1 && M[p][e] == q)
                    M[p][e] = clone, p = F[p];
                F[cur] = F[q] = clone;
            }
        }
        last = cur;
    }
    vector <int> E[SIZ];
    void build(){
        for(int i = 1;i <= s;++ i){
            E[F[i]].push_back(i);
        }
    }
    i64 ans = 0;
    void dfs(int u){
        for(auto &v : E[u]){
            dfs(v), S[u] += S[v];
        }
        if(S[u] > 1)
            ans = max(ans, 1ll * S[u] * L[u]);
    }
}
const int MAXN = 1e6 + 3;
char S[MAXN];
int main(){
    SAM :: init();
    scanf("%s", S); int n = strlen(S);
    for(int i = 0;i < n;++ i)
        SAM :: extend(S[i]);
    SAM :: build( );
    SAM :: dfs  (0);
    printf("%lld\n", SAM :: ans);
    return 0;
}

```
