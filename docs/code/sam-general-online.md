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
        if(M[last][e]){
            int p = last;
            int q = M[last][e];
            if(L[q] == L[last] + 1){
                last = q;
            } else {
                int clone = ++ s;
                L[clone] = L[p] + 1;
                F[clone] = F[q];
                for(int i = 0;i <= h;++ i)
                    M[clone][i] = M[q][i];
                while(p != -1 && M[p][e] == q)
                    M[p][e] = clone, p = F[p];
                F[q] = clone;
                last = clone;
            }
        } else {
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
    }
    void solve(){
        i64 ans = 0;
        for(int i = 1;i <= s;++ i)
            ans += L[i] - L[F[i]];
        cout << ans << endl;
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
        int m = strlen(S);
        last = 0;
        for(int j = 0;j < m;++ j){
            SAM :: extend(last, S[j]);
        }
    }
    SAM :: solve();
    cout << SAM :: s + 1 << endl;
    return 0;
}

```
