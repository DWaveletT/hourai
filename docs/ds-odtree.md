```cpp
#include "../header.cpp"
namespace ODT {
  // <pos_type, value_type>
  map <int, long long> M;
  // 分裂为 [1, p) 和 [p, +inf)，返回后者迭代器
  auto split(int p) {
    auto it = prev(M.upper_bound(p));
    return M.insert(
      it,
      make_pair(p, it -> second)
    );
  }
  // 区间赋值
  void assign(int l, int r, int v) {
    auto it = split(l);
    split(r + 1);
    while (it -> first != r + 1) {
      it = M.erase(it);
    }
    M[l] = v;
  }
  // // 执行操作
  // void perform(int l, int r) {
  //   auto it = split(l);
  //   split(r + 1);
  //   while (it -> first != r + 1) {
  //     // Do something...
  //     it = next(it);
  //   }
  // }
};
```
