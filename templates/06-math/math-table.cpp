/**
## 分拆数表

$$
\def\arraystretch{1.5}
\begin{array}{c||c|c|c|c|c|c|c|c|c|c} \hline\hline
n&10&20&30&40&50&60&70&80&90&100\\\hline
p(n)&42&627&5604&37338&204226&966467&4087968&15796476&56634173&190569292\\\hline\hline
\end{array}
$$

## 因数个数表

$$
\def\arraystretch{1.5}
\begin{array}{c||c|c|c|c|c|c|c|c|c} \hline\hline
N             & 10^1 & 10^2 & 10^3 & 10^4 & 10^5 & 10^6 & 10^7 & 10^8 & 10^9\\\hline
\max d(n)     &    4 &   12 &   32 &   64 &  128 &  240 &  448 &  768 & 1344\\
\max \omega(n)&    2 &    3 &    4 &    5 &    6 &    7 &    8 &    8 &    9\\
\hline\hline
N &10^{10}&10^{11}&10^{12}&10^{13}&10^{14}&10^{15}&10^{16}&10^{17}&10^{18}\\\hline
\max d(n)&2304&4032&6720&10752&17280&26880&41472&64512&103680\\
\max \omega(n)&10&10&11&12&12&13&13&14&15\\\hline\hline
\end{array}
$$

## 大质数

$10^{18}$ 级别：

- $P=10^{18}+3$，好记。
- $P=2924438830427668481$，可以进行 NTT，$P = 174310137655 \times 2 ^ 24 + 1$，原根为 $3$。
**/
