/**
## 三元环计数

**无向图**：考虑将所有点按度数从小往大排序，然后将每条边定向，由排在前面的指向排在后面的，得到一个有向图。然后考虑枚举一个点，再枚举一个点，暴力数，具体见代码。结论是，这样定向后，每个点的出度是 $O(\sqrt{m})$ 的。复杂度 $O(m\sqrt{m})$。
**有向图**：不难发现，上述方法枚举了三个点，计算有向图三元环也就只需要处理下方向的事，这个由于算法够暴力，随便改改就能做了。
**/

// 无向图
ll n, m; cin >> n >> m;
vector<pair<ll, ll>> Edges(m);
vector<vector<ll>> G(n + 2);
vector<ll> deg(n + 2);
for (auto &[i, j] : Edges) cin >> i >> j, ++deg[i], ++deg[j];
for (auto [i, j] : Edges) {
	if (deg[i] > deg[j] || (deg[i] == deg[j] && i > j)) swap(i, j);
	G[i].emplace_back(j);
}
vector<ll> val(n + 2);
ll ans = 0;
for (ll i = 1; i <= n; ++i) {
	for (auto j : G[i]) ++val[j];
	for (auto j : G[i]) for (auto k : G[j]) ans += val[k];
	for (auto j : G[i]) val[j] = 0;
}

// 有向图
ll n, m; cin >> n >> m;
vector<pair<ll, ll>> Edges(m);
vector<vector<pll>> G(n + 2);
vector<ll> deg(n + 2);
for (auto &[i, j] : Edges) cin >> i >> j, ++deg[i], ++deg[j];
for (auto [i, j] : Edges) {
	ll flg = 0;
	if (deg[i] > deg[j] || (deg[i] == deg[j] && i > j)) swap(i, j), flg = 1;
	G[i].emplace_back(j, flg);
}
vector<ll> in(n + 2), out(n + 2);
ll ans = 0;
for (ll i = 1; i <= n; ++i) {
	for (auto [j, w] : G[i]) w ? (++in[j]) : (++out[j]);
	for (auto [j, w1] : G[i]) for (auto [k, w2] : G[j]) {
		if (w1 == w2) ans += w1 ? in[k] : out[k];
	}
	for (auto [j, w] : G[i]) in[j] = out[j] = 0;
}
cout << ans << '\n';
