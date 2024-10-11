```cpp
#include<bits/stdc++.h>
using namespace std;

typedef long long i64;
const int INF  =  1e9;
const i64 INFL = 1e18;

const int MAXN = 1e5 + 3;

using edge = tuple<int, int, int>;

vector <edge> E[MAXN];
vector <edge> W;
vector <int>  C;

edge F[MAXN];
bool V[MAXN];
int  I[MAXN], o;

void dfs0(int u, int e){
    V[u] = true;
    I[u] = ++ o;
    for(auto &[i, v, w] : E[u]) if(i != e){
        if(V[v]){
            if(I[v] < I[u]){
                for(int p = u;p != v;){
                    auto &[j, f, x] = F[p];
                    C.push_back(p);
                    W.push_back({j, p, x});
                    p = f;
                }
                C.push_back(v);
                W.push_back({i, v, w});
            }
        } else {
            F[v] = {i, u, w};
            dfs0(v, i);
        }
    }
}

namespace Problem1{
// ===== 求直径 =====
    int H[MAXN], A[MAXN], B[MAXN];
    
    i64 L[MAXN];
    i64 dis = 0;
    void dfs1(int u, int e){
        for(auto &[i, v, w] : E[u]) if(i != e){
            if(!V[v]){
                dfs1(v, i);
                dis = max(dis, L[u] + w + L[v]);
                L[u] = max(L[u], L[v] + w);
            }
        }
    }
    int main(){
        int n;
        cin >> n;
        for(int i = 1;i <= n;++ i){
            int u, v, w;
            cin >> u >> v >> w;
            E[u].push_back({i, v, w});
            E[v].push_back({i, u, w});
        }
        dfs0(1, 0);
        memset(V, 0, sizeof(V));
        for(auto &u : C)
            V[u] = true;
        for(auto &u : C){
            dfs1(u, 0);
        }
        for(int i = 0;i < C.size();++ i){
            int x = C[i];
            if(i > 0)
                H[i] = H[i - 1] + get<2>(W[i - 1]);
            A[i] = L[x] + H[i];
            B[i] = L[x] - H[i];
        }
        int h = H[C.size() - 1] + get<2>(W.back());
        int j = 0;
        multiset <int, greater<int> > X, Y;
        for(int i = 0;i < C.size();++ i){
            int x = C[i];
            while(j < i && (H[i] - H[j]) * 2 >= h){
                X.insert(A[j]);
                Y.erase (Y.find(B[j]));
                ++ j;
            }
            if(!X.empty())
                dis = max(dis, L[x] - H[i] + *X.begin() + h);
            if(!Y.empty())
                dis = max(dis, L[x] + H[i] + *Y.begin() + 0);
            Y.insert(B[i]);
        }
        cout << dis << endl;
        return 0;
    }
}

namespace Problem2{
// ===== 删除环上第 i 条边，求直径 =====
    i64 H[MAXN], A1[MAXN], B1[MAXN], A2[MAXN], B2[MAXN], A3[MAXN], B3[MAXN];
    
    i64 L[MAXN];
    i64 dis = 0;
    void dfs1(int u, int e){
        for(auto &[i, v, w] : E[u]) if(i != e){
            if(!V[v]){
                dfs1(v, i);
                dis = max(dis, L[u] + w + L[v]);
                L[u] = max(L[u], L[v] + w);
            }
        }
    }
    int main(){
        int n;
        cin >> n;
        for(int i = 1;i <= n;++ i){
            int u, v, w;
            cin >> u >> v >> w;
            E[u].push_back({i, v, w});
            E[v].push_back({i, u, w});
        }
        dfs0(1, 0);
        memset(V, 0, sizeof(V));
        for(auto &u : C)
            V[u] = true;
        for(auto &u : C){
            dfs1(u, 0);
        }
        int l = 0, r = C.size() - 1;
        for(int i = l;i <= r;++ i){
            int x = C[i];
            if(i > 0)
                H[i] = H[i - 1] + get<2>(W[i - 1]);
            A1[i] = L[x] + H[i];
            B1[i] = L[x] - H[i];
            A2[i] = L[x] - H[i];
            B2[i] = L[x] + H[i];
        }
        i64 h = H[r] + get<2>(W.back());
        for(int i = l;i <= r;++ i)
            A1[i] = max(i == l ? -INFL : A1[i - 1], L[C[i]] + H[i]),
            A2[i] = max(i == l ? -INFL : A2[i - 1], L[C[i]] - H[i]);
        for(int i = r;i >= l;-- i)
            B1[i] = max(i == r ? -INFL : B1[i + 1], L[C[i]] - H[i]),
            B2[i] = max(i == r ? -INFL : B2[i + 1], L[C[i]] + H[i]);
        A3[l] = -INFL, B3[r] = -INFL;
        for(int i = l + 1;i <= r;++ i){
            int x = C[i];
            i64 w = A2[i - 1] + L[x] + H[i];
            A3[i] = max(A3[i - 1], w);
        }
        for(int i = r - 1;i >= l;-- i){
            int x = C[i];
            i64 w = B2[i + 1] + L[x] - H[i];
            B3[i] = max(B3[i + 1], w);
        }
        i64 t =  INFL;
        for(int i = l;i < r;++ i){
            i64 d = A1[i] + B1[i + 1] + h;
            i64 g = A2[i] + B2[i + 1] + 0;
            d = max({d, dis, A3[i], B3[i + 1]});
            t = min(t, d);
        }
        t = min(t, max(A3[r], dis));

        if(t % 2 == 0)
            cout << t / 2 << ".0" << endl;
        if(t % 2 == 1)
            cout << t / 2 << ".5" << endl;
        return 0;
    }
}

namespace Problem3{
// ===== 求最大点权独立集 =====

    int A[MAXN];
    i64 X[MAXN], Y[MAXN];
    i64 P[MAXN][2], Q[MAXN][2];
    void dfs1(int u, int e){
        for(auto &[i, v, w] : E[u]) if(i != e){
            if(!V[v]){
                dfs1(v, i);
                Y[u] += max(X[v], Y[v]);
                X[u] += Y[v];
            }
        }
        X[u] += A[u];
    }

    int main(){
        int n;
        cin >> n;
        for(int i = 1;i <= n;++ i){
            cin >> A[i];
        }
        for(int i = 1;i <= n;++ i){
            int u, v;
            cin >> u >> v;
            ++ u, ++ v;
            E[u].push_back({i, v, 0});
            E[v].push_back({i, u, 0});
        }
        double p;
        cin >> p;
        dfs0(1, 0);
        memset(V, 0, sizeof(V));
        for(auto &u : C)
            V[u] = true;
        for(auto &u : C){
            dfs1(u, 0);
        }
        int l = 0, r = C.size() - 1;
        P[0][1] = X[C[0]];
        P[0][0] = -INFL;
        Q[0][0] = Y[C[0]];
        Q[0][1] = -INFL;
        for(int i = l + 1;i <= r;++ i){
            int x = C[i];
            P[i][1] = X[x] + P[i - 1][0];
            P[i][0] = Y[x] + max(P[i - 1][0], P[i - 1][1]);
            Q[i][1] = X[x] + Q[i - 1][0];
            Q[i][0] = Y[x] + max(Q[i - 1][0], Q[i - 1][1]);
        }
        i64 ans = max({P[r][0], Q[r][0], Q[r][1]});
        cout << fixed << setprecision(1) << ans * p << endl;
        return 0;
    }
}

int main(){
    return Problem3 :: main();
}
```
