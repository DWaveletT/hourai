/**
设有一个积性函数 $f(n)$，满足 $f(p^k)$ 可以快速求，考虑搞一个在质数位置和 $f(n)$ 相等的 $g(n)$，满足它有完全积性，并且单点和前缀和都可以快速求，然后通过第一部分筛出 $g$ 在质数位置的前缀和，从而相当于得到 $f$ 在质数位置的前缀和，然后利用它，做第二部分，求出 $f$ 的前缀和。

1. $G_k(n)=\sum_{i=1}^{n}[\text{mindiv}(i)>p_k{~\text{or}~}\text{isprime}(i)]g(i)$（$p_0=1$），则有 $G_k(n)=G_{k-1}(n)-g(p_k) \times (G_{k-1}(n/p_k)-G_{k-1}(p_{k-1}))$，复杂度 $O({n^{3/4}}/{\log n})$。
2. $F_k(n)=\sum_{i=1}^{n}[\text{mindiv}(i)\ge p_k]f(i) = \sum[h\ge k, p_h^2\le n]\sum[c \ge 1, p_h^{c+1}\le n](f(p_h^c)F_{h+1}(n/p_h^c)+f(p_h^{c+1}))+F_{\text{prime}}(n)-F_{\text{prime}}(p_{k-1})$，在 $n\le 10^{13}$ 可以证明复杂度 $O(n^{3/4}/\log n)$。

常见细节问题：

- 由于 $n$ 通常是 $10^{10}$ 到 $10^{11}$ 的数，导致 $n$ 会爆 int，$n^2$ 会爆 long long，而且往往会用自然数幂和，更容易爆，所以要小心。
- 记 $s=\lfloor \sqrt{n}\rfloor$，由于 $F$ 递归时会去找 $F_{h+1}$，会访问到 $s$ 以内最大的质数往后的一个质数，而已经证明对于所有 $n\in\mathbb{N}^+$，$[n+1,2n]$ 中有至少一个质数，所以只需要筛到 $2s$ 即可。
- 注意补回 $f(1)$。
**/

// 预处理，$1$ 所在的块也算进去了
namespace init {
	ll init_n, sqrt_n;
	vector<ll> np, p, id1, id2, val;
	ll cnt;

	void main(ll n) {
		init_n = n, sqrt_n = sqrt(n);
		ll M = sqrt_n * 2; // 筛出一个 > floor(sqrt(n)) 的质数, 避免后续讨论边界
		
		np.resize(M + 1), p.resize(M + 1);
		for (ll i = 2; i <= M; ++i) {
			if (!np[i]) p[++p[0]] = i;
			for (ll j = 1; j <= p[0]; ++j) {
				if (i * p[j] > M) break;
				np[i * p[j]] = 1;
				if (i % p[j] == 0) break;
			}
		}
		p[0] = 1;

		id1.resize(sqrt_n + 1), id2.resize(sqrt_n + 1);
		val.resize(1);
		for (ll l = 1, r, v; l <= n; l = r + 1) {
			v = n / l, r = n / v;
			if (v <= sqrt_n) id1[v] = ++cnt;
			else id2[init_n / v] = ++cnt;
			val.emplace_back(v);
		}
	}

	ll id(ll n) {
		if (n <= sqrt_n) return id1[n];
		else return id2[init_n / n];
	}
}
using namespace init;

// 计算 $G_k$，两个参数分别是 $g$ 从 $2$ 开始的前缀和和 $g$
auto calcG = [&] (auto&& sum, auto&& g) -> vector<ll> {
	vector<ll> G(cnt + 1);
	for (int i = 1; i <= cnt; ++i) G[i] = sum(val[i]);
	ll pre = 0;
	for (int i = 1; p[i] * p[i] <= n; ++i) {
		for (int j = 1; j <= cnt; ++j) {
			if (p[i] * p[i] > val[j]) break;
			ll tmp = id(val[j] / p[i]);
			G[j] = (G[j] - g(p[i]) * (G[tmp] - pre)) % MD;
		}
		pre = (pre + g(p[i])) % MD;
	}
	for (int i = 1; i <= cnt; ++i) G[i] = (G[i] % MD + MD) % MD;
	return G;
};

// 计算 $F_k$，直接搜，不用记忆化。`fp` 是 $F_{\text{prime}}$，`pc` 是 $p^c$，其中 `f(p[h] ^ c)` 要替换掉。
function<ll(ll, int)> calcF = [&] (ll m, int k) {
	if (p[k] > m) return 0;
	ll ans = (fp[id(m)] - fp[id(p[k - 1])]) % MD;
	for (int h = k; p[h] * p[h] <= m; ++h) {
		ll pc = p[h], c = 1;
		while (pc * p[h] <= m) {
			ans = (ans + calcF(m / pc, h + 1) * f(p[h] ^ c)) % MD;
			++c, pc = pc * p[h], ans = (ans + f(p[h] ^ c)) % MD;
		}
	}
	return ans;
};
