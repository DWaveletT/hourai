/**
$$
(FA)_i = \sum_j \prod_{k = n}^0 c(B(i, k), B(j, k)) A_j, c_{*, i} c_{*, j} = c_{*, i\oplus j}
$$
**/

#include "../header.cpp"
void FWT(ll *F, ll len, ll c00, ll c01, ll c10, ll c11) {
  for (ll t = 1; t < len; t <<= 1) {
    for (ll i = 0; i < len; i += t * 2) {
      for (ll j = 0; j < t; ++j) {
        ll x = F[i + j];
        ll y = F[i + j + t];
      F[i + j] = (x * c00 + y * c01) % MD;
      F[i + j + t] = (x * c10 + y * c11) % MD;
      }
    }
  }
}