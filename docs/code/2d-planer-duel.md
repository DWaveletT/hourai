```cpp
#include "2d.cpp"

namespace DSU{
    const int MAXN = 1e5 + 3;
    int F[MAXN];
    int getfa(int u){
        return u == F[u] ? u : F[u] = getfa(F[u]);
    }
}

namespace Dual{
    const int MAXN = 1e5 + 3;
    const int MAXM = 1e5 + 3;
    int A[MAXM], B[MAXM], W[MAXM], I[MAXM], n, m;
    int outer;
    bool cmp(int a, int b){
        return W[a] < W[b];
    }

    vector <pair<int, int> > E[MAXN];
    const int MAXT = 20 + 3;
    int F[MAXN][MAXT], G[MAXN][MAXT], D[MAXN], h = 20;
    void dfs(int u, int f){
        D[u] = D[f] + 1;
        for(int i = 1;i <= h;++ i)
            F[u][i] = F[F[u][i - 1]][i - 1],
            G[u][i] = max(G[u][i - 1], G[F[u][i - 1]][i - 1]);
        for(auto &[v, w] : E[u]) if(v != f){
            G[v][0] = w;
            F[v][0] = u;
            dfs(v, u);
        }
    }
    void build(){
        for(int i = 1;i <= n;++ i)
            DSU :: F[i] = i;
        for(int i = 1;i <= m;++ i)
            I[i] = i;
        sort(I + 1, I + 1 + m, cmp);
        for(int i = 1;i <= m;++ i){
            int a = A[I[i]];
            int b = B[I[i]];
            int w = W[I[i]];
            int fa = DSU :: getfa(a);
            int fb = DSU :: getfa(b);
            if(fa != fb){
                DSU :: F[fa] = fb;
                E[a].push_back({b, w});
                E[b].push_back({a, w});
            }
        }
        dfs(1, 0);
    }
    int solve(int u, int v){
        if(u == outer || v == outer)
            return -1;
        int ans = 0;
        if(D[u] < D[v]) swap(u, v);
        for(int i = h;i >= 0;-- i)
            if(D[F[u][i]] >= D[v]){
                ans = max(ans, G[u][i]);
                u = F[u][i];
            }
        if(u == v) return ans;
        for(int i = h;i >= 0;-- i)
            if(F[u][i] != F[v][i]){
                ans = max(ans, G[u][i]);
                ans = max(ans, G[v][i]);
                u = F[u][i];
                v = F[v][i];
            }
        ans = max(ans, G[u][0]);
        ans = max(ans, G[v][0]);
        return ans;
    }
}

namespace Planer{
    const int MAXN = 1e5 + 3 + 3;
    const int MAXE = 2e5 + 3;
    const int MAXG = 1e5 + 3;
    const int MAXQ = 2e5 + 3;
    point P[MAXN];

    using edge = tuple<int, int>;

    double gety(int a, int b, double x){
        return P[a].y + (x - P[a].x) / (P[b].x - P[a].x) * (P[b].y - P[a].y);
    }

    double scanx;
    struct Cmp1{
        bool operator ()(const pair<edge, int> l1, const pair<edge, int> l2) const{
            const edge &e1 = l1.first;
            const edge &e2 = l2.first;
            double h1 = gety(get<0>(e1), get<1>(e1), scanx);
            double h2 = gety(get<0>(e2), get<1>(e2), scanx);
            return h1 < h2;
        };
    };
    struct Cmp2{
        bool operator ()(const pair<edge, int> l1, const pair<edge, int> l2) const{
            if(l1.second == l2.second)
                return false;
            const edge &e1 = l1.first;
            const edge &e2 = l2.first;
            vec v1 = P[get<1>(e1)] - P[get<0>(e1)];
            vec v2 = P[get<1>(e2)] - P[get<0>(e2)];
            if(sign(v1.y) != sign(v2.y)){
                return v1.y > 0;
            } else {
                return sign(mulx(v1, v2)) == 1;
            }
        };
    };

    vector <pair<edge, int> > E[MAXN];
    
    vector <int> G[MAXG];
    int L[MAXE], R[MAXE], W[MAXE], n, m, q, o;
    double theta;

    int outer;

    void rotate(){
        srand(time(0));
        theta = PI * rand() / RAND_MAX;
    }

    int add(double x, double y){
        srand(time(0));
        P[++ n] = rotate(vec(x, y), theta);
        return n;
    }
    int link(int u, int v, int w){
        ++ m;
        E[u].push_back({{u, v}, ++ o});
        L[o] = u, R[o] = v, W[o] = w;
        E[v].push_back({{v, u}, ++ o});
        L[o] = v, R[o] = u, W[o] = w;
        return m;
    }

    int I[MAXE];
    int polys;
    pair<edge, int> findleft(int l, int r){
        auto it = lower_bound(E[r].begin(), E[r].end(), make_pair(edge(r, l), 0), Cmp2());
        if(it == E[r].begin())
            return E[r].back();
        else 
            return *(it - 1);
    }
    void leftmost(){
        for(int i = 1;i <= n;++ i){
            sort(E[i].begin(), E[i].end(), Cmp2());
        }
        for(int p = 1;p <= n;++ p){
            for(auto &[e1, id1] : E[p]){
                auto &[x, y] = e1;
                if(!I[id1]){
                    int l = x;
                    int r = y;
                    I[id1] = ++ polys;
                    G[polys].push_back(id1);
                    while(r != p){
                        auto [e2, id2] = findleft(l, r);
                        auto [a, b] = e2;
                        I[id2] = polys;
                        G[polys].push_back(id2);
                        l = r;
                        r = b;
                    }
                }
            }
        }
        for(int i = 1;i <= polys;++ i){
            double area = 0;
            for(int j = 0;j < G[i].size();++ j){
                area += mulx(P[L[G[i][j]]], P[R[G[i][j]]]);
            }
            if(area < 0)
                outer = i;
        }
    }

    void dual(){
        Dual :: n = polys;
        Dual :: m = 0;
        for(int i = 1;i <= m;++ i){
            int u = I[2 * i - 1], v = I[2 * i], w = W[2 * i];
            if(u == outer || v == outer)
                w = 1e9L + 1;
            ++ Dual :: m;
            Dual :: A[Dual :: m] = u;
            Dual :: B[Dual :: m] = v;
            Dual :: W[Dual :: m] = w;
        }
        Dual :: build();
        Dual :: outer = outer;
    }

    set <pair<edge, int>, Cmp1> S;
    
    vector <pair<double, int> > T;
    vector <pair<double, int> > Q;

    double X[MAXQ], Y[MAXQ];
    int    Z[MAXQ];

    int ask(double x, double y){
        ++ q;
        point p = rotate(vec(x, y), theta);
        X[q] = p.x;
        Y[q] = p.y;
        return q;
    }

    void locate(){
        T.clear(), Q.clear(), S.clear();
        for(int i = 1;i <= q;++ i){
            Q.push_back(make_pair(X[i], i));
        }
        for(int i = 1;i <= polys;++ i){
            for(auto &e : G[i]){
                int u = L[e];
                int v = R[e];
                if(P[u].x > P[v].x){
                    T.push_back(make_pair(P[v].x + 1e-5,  e));
                    T.push_back(make_pair(P[u].x - 1e-5, -e));
                }
            }
        }
        sort(T.begin(), T.end());
        sort(Q.begin(), Q.end());

        int p1 = 0, p2 = 0;
        scanx = -1e9;
        Cmp1 CMP;
        while(p1 < Q.size() || p2 < T.size()){
            // for(auto it1 = S.begin(), it2 = next(S.begin()); it2 != S.end();++ it1, ++ it2)
            //     assert(CMP(*it1, *it2));
            double x1 = p1 < Q.size() ? Q[p1].first : 1e9;
            double x2 = p2 < T.size() ? T[p2].first : 1e9;
            scanx = min(x1, x2);
            if(equal(scanx, x1)){
                auto &x = X[Q[p1].second];
                auto &y = Y[Q[p1].second];
                auto &z = Z[Q[p1].second];
                P[n + 1] = point(-1e9, y);
                P[n + 2] = point( 1e9, y);
                auto it = S.lower_bound({{n + 1, n + 2}, 0});
                if(it == S.end())
                    z = outer;
                else 
                    z = it -> second;
                ++ p1;
            }
            if(equal(scanx, x2)){
                int g = T[p2].second;
                if(g > 0){
                    assert(!S.count({{L[g], R[g]}, I[g]}));
                    S.insert({{L[g], R[g]}, I[g]});
                } else {
                    g = -g;
                    assert( S.count({{L[g], R[g]}, I[g]}));
                    S.erase ({{L[g], R[g]}, I[g]});
                }
                ++ p2;
            }
        }
    }
}

const int MAXN = 1e5 + 3;
int A[MAXN], B[MAXN];

int main(){
#ifndef ONLINE_JUDGE
    freopen("test.in", "r", stdin);
    freopen("test.out", "w", stdout);
#endif
    int n, m, q;
    Planer :: rotate();

    cin >> n >> m;
    for(int i = 1;i <= n;++ i){
        double x, y;
        cin >> x >> y;
        Planer :: add(x, y);
    }
    for(int i = 1;i <= m;++ i){
        int u, v, w;
        cin >> u >> v >> w;
        Planer :: link(u, v, w);
    }
    Planer :: leftmost();
    Planer :: dual();
    cin >> q;
    for(int i = 1;i <= q;++ i){
        double a1, b1, a2, b2;
        cin >> a1 >> b1;
        A[i] = Planer :: ask(a1, b1);
        cin >> a2 >> b2;
        B[i] = Planer :: ask(a2, b2);
    }
    Planer :: locate();
    for(int i = 1;i <= q;++ i)
        A[i] = Planer :: Z[A[i]],
        B[i] = Planer :: Z[B[i]];
    for(int i = 1;i <= q;++ i){
        int ans = Dual :: solve(A[i], B[i]);
        cout << ans << endl;
    }
    return 0;
}
```
