#include "../header.cpp"
using type = long double;
using cp = complex<type>;
using ll = long long;

vector<int> rev;
vector<cp> omega;   // exp(sgn * pi * k / len)
void FFT(cp* F, int len, int sgn) {
    rev.resize(len);
    for (int i = 1; i < len; ++i) {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) * (len >> 1));
        if (i < rev[i]) swap(F[i], F[rev[i]]);
    }
    const type pi = std::acos(type(-1));
    omega.resize(len);
    for (int i = 0; i < len; ++i) {
        omega[i] = polar(type(1), sgn * i * pi / len);
    }
    for (int tmp = 1; tmp < len; tmp <<= 1) {
        for (int i = 0; i < len; i += tmp << 1) {
            int K = len / tmp, pos = 0;
            for (int j = 0; j < tmp; ++j, pos += K) {
                cp x = F[i + j], y = F[i + j + tmp] * omega[pos];
                F[i + j] = x + y, F[i + j + tmp] = x - y;
            }
        }
    }
    if (sgn == -1) {
        cp inv_len(type(1) / len);
        for (int i = 0; i < len; ++i) F[i] = F[i] * inv_len;
    }
}

ll MD, M; // 输入 MD 后, 需要设置 M 为 sqrt(MD)

using Poly = vector<ll>;
Poly polyMul(Poly F, Poly G, int tmp = 0) { // tmp 用于循环卷积技巧, 卡常.
    for (auto &k : F) k %= MD;
    for (auto &k : G) k %= MD;
    int n = (int)F.size() - 1, m = (int)G.size() - 1;
    if (tmp == 0) tmp = n + m + 1;
    int len = 1;
    while (len < tmp) len <<= 1;
    vector<cp> P(len), tP(len), Q(len);
    for (int i = 0; i <= n; ++i)
        P[i] = cp(F[i] / M, F[i] % M),
        tP[i] = cp(F[i] / M, -(F[i] % M));
    for (int i = 0; i <= m; ++i)
        Q[i] = cp(G[i] / M, G[i] % M);
    for(auto &X: {&P, &Q, &tP})
        FFT(X -> data(), len, 1);
    for (int i = 0; i < len; ++i)
        P[i] *= Q[i], tP[i] *= Q[i];
    FFT(P.data(), len, -1);
    FFT(tP.data(), len, -1);
    vector<ll> H(n + m + 1);
    for (int i = 0; i < tmp; ++i) {
        H[i] = ll((P[i].real() + tP[i].real()) / 2 + 0.5) % MD * M % MD * M % MD
             + ll(P[i].imag() + 0.5) % MD * M % MD
             + ll((tP[i].real() - P[i].real()) / 2 + 0.5) % MD;
        H[i] = (H[i] + MD) % MD;
    }
    return H;
}