#include "../header.cpp"
using Poly = vector<ll>;
#define lg(x) ((x) == 0 ? -1 : __lg(x))
#define Size(x) int(x.size())

namespace NTT_ns {
	const long long G = 3, invG = inv(G);

	void Root(ll x = MD) {
		ll _tp1 = x - 1, _tp2 = 0;
		while (_tp1 % 2 == 0) _tp1 >>= 1, ++_tp2;
		fprintf(stderr, "MD = %lld * 2 ^ %lld + 1\n", _tp1, _tp2);
		vector<ll> p; _tp1 = x - 1;
		for (ll i = 2; i * i <= _tp1; ++i)
      if (_tp1 % i == 0) {
        p.emplace_back(i);
        while (_tp1 % i == 0) _tp1 /= i;
      }
		if (_tp1 != 1) p.emplace_back(_tp1);
		for (ll r = 1; r < x; ++r) {
			int cnt = 0;
			for (auto& i : p)
        if (qpow(r, (x - 1) / i) == 1) ++cnt;
			if (cnt == 0)
        {fprintf(stderr, "%lld\n", r); break;}
		}
	}

	vector<int> rev;
	void NTT(ll* F, int len, int sgn) {
		rev.resize(len);
		for (int i = 1; i < len; ++i) {
			rev[i] = (rev[i >> 1] >> 1) |
               ((i & 1) * (len >> 1));
			if (i < rev[i]) swap(F[i], F[rev[i]]);
		}
		for (int tmp = 1; tmp < len; tmp <<= 1) {
			ll w1 = qpow(sgn ? G : invG,
        (MD - 1) / (tmp << 1));
			for (int i = 0; i < len; i += tmp << 1) {
				for(ll j = 0, w = 1; j < tmp; ++j,
            w = w * w1 % MD) {
					ll x = F[i + j];
          ll y = F[i + j + tmp] * w % MD;
					F[i + j] = (x + y) % MD;
          F[i + j + tmp] = (x - y + MD) % MD;
				}
			}
		}
		if (sgn == 0) {
			ll inv_len = inv(len);
			for (int i = 0; i < len; ++i)
        F[i] = F[i] * inv_len % MD;
		}
	}

	vector<ll> Iv(2, 1), jc(1, 1), ijc(1, 1);
	void Add_Inv(int len) {
		Iv[0] = 0, Iv[1] = 1, len += 10;
    if (len < Size(Iv)) return;
		int i = Size(Iv); Iv.resize(len);
		while (i < len)
      Iv[i] = (MD - MD / i * Iv[MD % i] % MD) % MD, ++i;
	}

	void Add_Fac(int len) {
		len += 10, Add_Inv(len);
    if (len < Size(jc)) return;
		int i = Size(jc);
    jc.resize(len), ijc.resize(len);
		for (; i < len; ++i)
      jc[i] = jc[i - 1] * i % MD,
      ijc[i] = ijc[i - 1] * Iv[i] % MD;
	}

	ll Inv(int x) {
		return Add_Inv(x), Iv[x];
	}

	ll Binom(int n, int m) {
		Add_Fac(max(n, m));
		return n < m ? 0 : jc[n] * ijc[m] % MD * ijc[n - m] % MD;
	}
}

Poly operator * (Poly F, Poly G) {
	int siz = Size(F) + Size(G) - 1;
  int len = 1 << (lg(siz - 1) + 1);
	if (siz <= 300) {
		Poly H(siz);
		for (int i = Size(F) - 1; ~i; --i)
      for (int j = Size(G) - 1; ~j; --j)
		    H[i + j] = (H[i + j] + F[i] * G[j]) % MD;
		return H;
	}
	/*
		建议写完 NTT 先删掉暴力卷积，测一下对不对。
		别搞半天发现原来一直用的暴力卷积，NTT 根本就不对。
	*/
	using NTT_ns::NTT;
  F.resize(len), G.resize(len);
	NTT(F.data(), len, 1), NTT(G.data(), len, 1);
	for (int i = 0; i < len; ++i)
    F[i] = F[i] * G[i] % MD;
  NTT(F.data(), len, 0), F.resize(siz);
	return F;
}

Poly operator + (Poly F, Poly G) {
	int siz = max(Size(F), Size(G));
  F.resize(siz), G.resize(siz);
	for (int i = 0; i < siz; ++i)
    F[i] = (F[i] + G[i]) % MD;
	return F;
}

Poly operator - (Poly F, Poly G) {
	int siz = max(Size(F), Size(G));
  F.resize(siz), G.resize(siz);
	for (int i = 0; i < siz; ++i)
    F[i] = (F[i] - G[i] + MD) % MD;
	return F;
}

Poly lsh(Poly F, int k) {
	F.resize(Size(F) + k);
	for (int i = Size(F) - 1; i >= k; --i) F[i] = F[i - k];
	for (int i = 0; i < k; ++i) F[i] = 0;
	return F;
}

Poly rsh(Poly F, int k) {
	int siz = Size(F) - k;
	for (int i = 0; i < siz;++i)F[i] = F[i + k];
	return F.resize(siz), F;
}

Poly cut(Poly F, int len) {
	return F.resize(len), F;
}

Poly der(Poly F) {
	int siz = Size(F) - 1;
	for (int i = 0; i < siz; ++i)
    F[i] = F[i + 1] * (i + 1) % MD;
	return F.pop_back(), F;
}

Poly inte(Poly F) {
	F.emplace_back(0);
	for (int i = Size(F) - 1; ~i; --i)
    F[i] = F[i - 1] * NTT_ns::Inv(i) % MD;
	return F[0] = 0, F;
}

Poly inv(Poly F) {
	int siz = Size(F); Poly G{inv(F[0])};
	for (int i = 2; (i >> 1) < siz; i <<= 1) {
		G = G + G - G * G * cut(F, i), G.resize(i);
	}
	return G.resize(siz), G;
}

Poly ln(Poly F) {
	return cut(inte(cut(der(F) * inv(F), Size(F))), Size(F));
}

Poly Exp(Poly F) {
	int siz = Size(F); Poly G{1};
	for (int i = 2; (i >> 1) < siz; i <<= 1) {
		G = G * (Poly{1} - ln(cut(G, i)) + cut(F, i)), G.resize(i);
	}
	return G.resize(siz), G;
}