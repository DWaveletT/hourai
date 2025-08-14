#include "../header.cpp"
vector<int> bm(const vector<int> &a) {
  vector<int> v, ls;
  int k = -1, delta = 0;
  for (int i = 0; i < a.size();i ++) {
    int tmp = 0;
    for (int j = 0;j < v.size();j ++)
      tmp = (tmp + (ll)a[i - j - 1] * v[j]) % MD;
    if (a[i] == tmp) continue;
    if (k < 0) { k = i; delta = (a[i] - tmp + MD) % MD; v = vector<int>(i + 1); continue; }
    vector<int> u = v;
    int val = (ll)(a[i] - tmp + MD) * qpow(delta, MD - 2) % MD;
    v.resize(max(v.size(), ls.size()+ i - k));
    (v[i - k - 1] += val) %= MD;
    for (int j = 0; j < (int)ls.size(); j++){
      v[i - k + j] = (v[i - k + j] - (ll)val * ls[j]) % MD;
      if(v[i - k + j] < 0) v[i - k + j] += MD;
    }
    if (u.size() + k < ls.size() + i){
      ls = u; k = i, delta = a[i] - tmp;
      if (delta < 0) delta += MD;
    }
  }
  for (auto &x : v) x = (MD - x) % MD;
  v.insert(v.begin(), 1);
  return v;
} // $\forall i, \sum_{j = 0} ^ m a_{i - j} v_j = 0$
