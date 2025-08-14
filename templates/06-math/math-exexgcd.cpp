/**
## 类欧几里得（万能欧几里得）

一种神奇递归，对 $\displaystyle y=\left\lfloor \frac{Ax+B}{C}\right\rfloor$ 向右和向上走的每步进行压缩，做到 $O(\log V)$ 复杂度。其中 $A\ge C$ 就是直接压缩，向右之后必有至少 $\lfloor A/C\rfloor$ 步向上。$A<C$ 实际上切换 $x,y$ 轴后，相当于压缩了一个上取整折线，而上取整下取整可以互化，便又可以递归。

代码中从 $(0,0)$ 走到 $(n,\lfloor (An+B)/C\rfloor)$，假设了 $A,B,C\ge 0,C\neq 0$（类欧基本都作此假设），$U,R$ 矩阵是从右往左乘的，对列向量进行优化，和实际操作顺序恰好相反。快速幂的 log 据说可以被递归过程均摊掉，实际上并不会导致变成两个 log。
**/
Matrix solve(ll n, ll A, ll B, ll C, Matrix R, Matrix U) {	// (0, 0) 走到 (n, (An+B)/C)
	if (A >= C) return solve(n, A % C, B, C, U.qpow(A / C) * R, U);
	ll l = B / C, r = (A * n + B) / C;
	if (l == r) return R.qpow(n) * U.qpow(l);	// l = r -> l = r or A = 0 or n = 0.
	ll p = (C * r - B - 1) / A + 1;
	return R.qpow(n - p) * U * solve(r - l - 1, C, C - B % C + A - 1, A, U, R) * U.qpow(l);
}