#include <bits/stdc++.h>
using namespace std;

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

namespace LeftHeap{
    const int SIZ = 1e5 + 3;
    int W[SIZ], D[SIZ];
    int L[SIZ], R[SIZ];
    int F[SIZ], s;

    bool E[SIZ];

    int merge(int u, int v){
        if(u == 0 || v == 0)
            return u | v;
        if(W[u] > W[v] || (W[u] == W[v] && u > v))
            swap(u, v);
        int &lc = L[u];
        int &rc = R[u];
        rc = merge(rc, v);
        if(D[lc] < D[rc])
            swap(lc, rc);
        D[u] = min(D[lc], D[rc]) + 1;
        if(lc != 0) F[lc] = u;
        if(rc != 0) F[rc] = u;
        return u;
    }
    void pop(int &root){
        int root0 = merge(L[root], R[root]);
        F[root0] = root0;
        F[root ] = root0;
        E[root ] = true;
        root = root0;
    }
    int top(int &root){
        return W[root];
    }
    int getfa(int u){
        return u == F[u] ? u : F[u] = getfa(F[u]);
    }
    int newnode(int w){
        ++ s;
        W[s] = w;
        F[s] = s;
        D[s] = 1;
        return s;
    }
}

// ===== TEST =====

int qread();

const int MAXN = 1e5 + 3;
int A[MAXN], O[MAXN];

int main(){
    int n, m;
    cin >> n >> m;
    for(int i = 1;i <= n;++ i){
        cin >> A[i];
        O[i] = LeftHeap :: newnode(A[i]);
    }
    for(int i = 1;i <= m;++ i){
        int op;
        cin >> op;
        if(op == 1){
            int x, y;
            cin >> x >> y;
            if(LeftHeap :: E[O[x]])
                continue;
            if(LeftHeap :: E[O[y]])
                continue;
            int fx = LeftHeap :: getfa(O[x]);
            int fy = LeftHeap :: getfa(O[y]);
            if(fx != fy){
                LeftHeap :: merge(fx, fy);
            }
        } else {
            int x;
            cin >> x;
            if(LeftHeap :: E[O[x]]){
                cout << -1 << endl;
                continue;
            }
            int fx = LeftHeap :: getfa(O[x]);
            cout << LeftHeap :: top(fx) << endl;
            LeftHeap :: pop(fx);
        }
    }
    return 0;
}