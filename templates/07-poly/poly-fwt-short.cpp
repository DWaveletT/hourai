/**
有 $n$ 个多项式 $\displaystyle F_i=\sum_{j=1}^{k}w_jx^{a_{i,j}}$，$(w_i)$ 序列固定，并且 $k$ 很小，$a_i\in[0,2^m)$，现在要把他们位运算卷积起来，求最终的序列。

异或版本，复杂度 $O(2^k((n+m)k+2^m(m+\log V)))$，按通常大小关系可以认为是 $O(nk2^k+m2^{m+k})$，最后那个 $\log V$ 是快速幂，可以 $O(n2^k)$ 预处理去掉：
**/
#include "poly-fwt.cpp"
#define vv vector

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	ll n, k, m;
	cin >> n >> m, k = 3;
	vv<ll> w(k);
	for (auto &i : w) cin >> i;
	vv<vv<ll>> a(n, vv<ll>(k));
	for (auto &i : a) for (auto &j : i) cin >> j;
	ll uk = 1 << k, V = 1 << m;
	vv<vv<ll>> c(V, vv<ll>(uk));
	vv<ll> val(uk);
	for (ll i = 0; i < uk; ++i) {
		for (ll p = 0; p < k; ++p) {
			if ((i >> p) & 1) val[i] = (val[i] - w[p] + MD) % MD;
			else val[i] = (val[i] + w[p]) % MD;
		}
		vv<ll> f(V);
		for (ll j = 0; j < n; ++j) {
			ll z = 0;
			for (ll p = 0; p < k; ++p) if ((i >> p) & 1) z ^= a[j][p];
			++f[z];
		}
		FWT(f.data(), V, 1, 1, 1, MD - 1);
		for (ll j = 0; j < V; ++j) c[j][i] = f[j];
	}
	for (ll i = 0; i < V; ++i) FWT(c[i].data(), uk, inv2, inv2, inv2, MD - inv2);
	vv<ll> Ans(V, 1);
	for (ll i = 0; i < V; ++i) {
		for (ll j = 0; j < uk; ++j) {
			Ans[i] = Ans[i] * qpow(val[j], c[i][j]) % MD;
		}
	}
	FWT(Ans.data(), V, inv2, inv2, inv2, MD - inv2);
	for (ll i = 0; i < V; ++i) cout << Ans[i] << " \n"[i == V - 1];
	return 0;
}
