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
    int s = 0, h = 25;
    void init(){
        F[0] = -1, s = 0;
    }
    void extend(int &last, char c){
        int e = c - 'a';
        int cur = ++ s;
        L[cur] = L[last] + 1;
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
                for(int i = 0;i <= h;++ i)
                    M[clone][i] = M[q][i];
                while(p != -1 && M[p][e] == q)
                    M[p][e] = clone, p = F[p];
                F[cur] = F[q] = clone;
            }
        }
        last = cur;
    }
    void solve(){
        i64 ans = 0;
        for(int i = 1;i <= s;++ i)
            ans += L[i] - L[F[i]];
        cout << ans << endl;
    }
}

namespace Trie{
    const int SIZ = 1e6 + 3;
    int M[SIZ][MAXM], s, h = 25;

    void insert(char *S){
        int p = 0;
        for(int i = 0;S[i];++ i){
            int e = S[i] - 'a';
            if(M[p][e]){
                p = M[p][e];
            } else 
                p = M[p][e] = ++ s;
        }
    }
    int O[SIZ];

    void build_sam(){
        queue <int> Q;
        Q.push(0);
        while(!Q.empty()){
            int u = Q.front(); Q.pop();
            for(int i = 0;i <= h;++ i){
                char c = i + 'a';
                if(M[u][i]){
                    int v = M[u][i];
                    O[v] = O[u];
                    SAM :: extend(O[v], c);
                    Q.push(v);
                }
            }
        }
    }
}
const int MAXN = 1e6 + 3;
char S[MAXN];
int main(){
    SAM :: init();
    int n, last = 0;
    cin >> n;
    for(int i = 1;i <= n;++ i){
        scanf("%s", S);
        Trie :: insert(S);
    }
    Trie :: build_sam();
    SAM :: solve();
    cout << SAM :: s + 1 << endl;
    return 0;
}

```
