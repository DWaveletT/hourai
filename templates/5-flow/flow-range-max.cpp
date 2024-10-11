#include <bits/stdc++.h>
using namespace std;

int qread(){
    int w = 1, c, ret;
    while((c = getchar()) >  '9' || c <  '0') w = (c == '-' ? -1 : 1); ret = c - '0';
    while((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}

using i64 = long long;
const int INF  =  1e9;
const i64 INFL = 1e18;

namespace MCMF{
    const int MAXN = 1e5 + 3;
    const int MAXM = 2e5 + 3;
    int H[MAXN], V[MAXM], N[MAXM], F[MAXM], o = 1, n;
    void add0(int u, int v, int f){
        V[++ o] = v, N[o] = H[u], H[u] = o, F[o] = f;
        V[++ o] = u, N[o] = H[v], H[v] = o, F[o] = 0;
        n = max(n, u);
        n = max(n, v);
    }
    i64 D[MAXN];
    bool bfs(int s, int t){
        queue <int> Q;
        for(int i = 1;i <= n;++ i)
            D[i] = 0;
        Q.push(s), D[s] = 1;
        while(!Q.empty()){
            int u = Q.front(); Q.pop();
            for(int i = H[u];i;i = N[i]){
                const int &v = V[i];
                const int &f = F[i];
                if(f != 0 && !D[v]){
                    D[v] = D[u] + 1;
                    Q.push(v);
                }
            }
        }
        return D[t] != 0;
    }
    int C[MAXN];
    i64 dfs(int s, int t, int u, i64 maxf){
        if(u == t)
            return maxf;
        i64 totf = 0;
        for(int &i = C[u];i;i = N[i]){
            const int &v = V[i];
            const int &f = F[i];
            if(f && D[v] == D[u] + 1){
                i64 f = dfs(s, t, v, min(1ll * F[i], maxf));
                F[i    ] -= f;
                F[i ^ 1] += f;
                totf += f;
                maxf -= f;
                if(maxf == 0){
                    return totf;
                }
            }
        }
        return totf;
    }
    i64 mcmf(int s, int t){
        i64 ans = 0;
        while(bfs(s, t)){
            memcpy(C, H, sizeof(H));
            ans += dfs(s, t, s, INFL);
        }
        return ans;
    }
    int G[MAXN];
    void add(int u, int v, int l, int r){
        G[v] += l;
        G[u] -= l;
        add0(u, v, r - l);
    }
	void clear(){
		for(int i = 1;i <= o;++ i){
			N[i] = F[i] = V[i] = 0;
		}
		for(int i = 1;i <= n;++ i){
			H[i] = G[i] = C[i] = 0;
		}
		o = 1, n = 0;
	}
    bool solve(){
        int s = ++ n;
        int t = ++ n;
        i64 sum = 0;
        for(int i = 1;i <= n - 2;++ i){
            if(G[i] < 0)
                add0(i, t, -G[i]);
            else
                add0(s, i,  G[i]), sum += G[i];
        }
        auto res = mcmf(s, t);
        if(res != sum)
            return true;
        return false;
    }
    i64 solve(int s0, int t0){
        add0(t0, s0, INF);
        int s = ++ n;
        int t = ++ n;
        i64 sum = 0;
        for(int i = 1;i <= n - 2;++ i){
            if(G[i] < 0)
                add0(i, t, -G[i]);
            else
                add0(s, i,  G[i]), sum += G[i];
        }
        auto res = mcmf(s, t);
        if(res != sum)
            return -1;
        return mcmf(s0, t0);
    }
}

const int MAXN = 1e3 + 3;
const int MAXM = 365 + 3;

int G[MAXN], A[MAXN], B[MAXM];

int main(){
	ios :: sync_with_stdio(false);
	cin.tie(nullptr);

	int n, m, o = 0;
	while(cin >> n >> m){

		int s = ++ o;
		int t = ++ o;

		for(int i = 1;i <= m;++ i){
			cin >> G[i];
			A[i] = ++ o;
			MCMF :: add(A[i], t, G[i], INF);
		}
		for(int i = 1;i <= n;++ i){
			B[i] = ++ o;

			int c, d;
			cin >> c >> d;

			MCMF :: add(s, B[i], 0, d);

			for(int j = 1;j <= c;++ j){
				int t, l, r;
				cin >> t >> l >> r;
				t ++;
				MCMF :: add(B[i], A[t], l, r);
			}
		}
		cout << MCMF :: solve(s, t) << "\n\n";

		MCMF :: clear();
	}
	
	return 0;
}
