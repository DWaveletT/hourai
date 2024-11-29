## 四元环计数

_From zpk_

- **无向图**：类似，由于定向后出度结论过于强大，可以暴力。讨论了三种情况。
- **有向图**：缺少题目，但应当类似三元环计数有向形式记录定向边和原边的正反关系。因为此法最强的结论是定向后出度 $O(\sqrt{m})$，实际上方法很暴力，应当不难数有向形式的。

```cpp
ll n, m; cin >> n >> m;
vector<pair<ll, ll>> Edges(m);
vector<vector<ll>> G(n + 2), iG(n + 2);
vector<ll> deg(n + 2);
for (auto &[i, j] : Edges) cin >> i >> j, ++deg[i], ++deg[j];
for (auto [i, j] : Edges) {
	if (deg[i] > deg[j] || (deg[i] == deg[j] && i > j)) swap(i, j);
	G[i].emplace_back(j), iG[j].emplace_back(i);
}
ll ans = 0;
vector<ll> v1(n + 2), v2(n + 2);
for (ll i = 1; i <= n; ++i) {
	for (auto j : G[i]) for (auto k : G[j]) ++v1[k];
	for (auto j : iG[i]) for (auto k : G[j]) ans += v1[k], ++v2[k];
	for (auto j : G[i]) for (auto k : G[j]) ans += v1[k] * (v1[k] - 1) / 2, v1[k] = 0;
	for (auto j : iG[i]) for (auto k : G[j]) {
		if (deg[k] > deg[i] || (deg[k] == deg[i] && k > i)) ans += v2[k] * (v2[k] - 1) / 2;
		v2[k] = 0;
	}
}
cout << ans << '\n';

```
