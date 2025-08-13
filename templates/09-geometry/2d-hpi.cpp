#include "2d-lines.cpp"

vector<pp> HPI(vector<line> vs) {
  auto cmp = [](line a, line b) {
    return paraS(a, b) ? dis(a) < dis(b)
      : ::cmp(pp(a), pp(b));
  }; 
  sort(vs.begin(), vs.end(), cmp); 
  int ah = 0, at = 0, n = size(vs); 
  vector<line> deq(n + 1); 
  vector<pp> ans(n);
  deq[0] = vs[0]; 
  for(int i = 1;i <= n;++i) { 
    line o = i < n ? vs[i] : deq[ah];
    if(paraS(vs[i - 1], o)) continue;

    // maybe <=
    while(ah < at && check(deq[at - 1], deq[at], o) < 0)
      -- at;
    if(i != n)
    while(ah < at && check(deq[ah], deq[ah + 1], o) < 0)
      ++ ah;
    if(!is_parallel(o, deq[at])) { 
      ans[at] = o & deq[at], deq[++at] = o; 
    }
  }
  if(at - ah <= 2) return {}; 
  return {ans.begin() + ah, ans.begin() + at};
}