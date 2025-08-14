#include "2d.cpp"

struct Hull {
  set<pp> S;
  using Iter = set<pp>::iterator;
  long long ans = 0;
  long long area(pp a, pp b, pp c) {
    return llabs((b - a) * (c - a));
  }
  Iter remove_dir(Iter it, bool dir, pp x){
    Iter nxt;
    while (dir ? (it != S.begin() && (nxt = prev(it), 1)) : ((nxt = next(it)) != S.end())) {
      if (((*it - *nxt) * (x - *it) < 0) == dir) break;
      ans += area(*nxt, *it, x), S.erase(it), it = nxt;
    }
    return it;
  }
  void insert(pp x){
    if (S.empty()){ S.insert(x); return; }
    auto r = S.lower_bound(x);
    if (r == S.end()) {
      remove_dir(prev(r), 1, x);
      S.insert(x);
    } else if (r == S.begin()) {
      remove_dir(r, 0, x);
      S.insert(x);
    } else {
      auto l = prev(r);
      if (((*r - *l) * (x - *l)) < 0)
        return; // 在凸包外侧
      ans += area(*l, *r, x);
      l = remove_dir(l, 1, x);
      r = remove_dir(r, 0, x);
      S.insert(x);
    }
  }
};