#include "2d.cpp"

std::vector<p2> HPI(std::vector<line> vs) { 
  auto cmp = [](line a, line b) { 
    if(paraS(a, b)) return dist(a) < dist(b); 
    return ::cmp(p2(a), p2(b)); 
  }; 
  sort(vs.begin(), vs.end(), cmp); 
  int ah = 0, at = 0, n = size(vs); 
  std::vector<line> deq(n + 1); 
  std::vector<p2> ans(n); 
  deq[0] = vs[0]; 
  for(int i = 1;i <= n;++i) { 
  line o = i < n ? vs[i] : deq[ah]; 
  if(paraS(vs[i - 1], o)) continue; 
  for(;ah < at && check(deq[at - 1], deq[at], o) < 0;) -- at;//maybe <= 
  if(i != n) for(;ah < at && check(deq[ah], deq[ah + 1], o) < 0;) ++ ah; 
  if(!is_para(o, deq[at])) { 
  ans[at] = o & deq[at]; 
  deq[++at] = o; 
  }
  }
  if(at - ah <= 2) return {}; 
  return {ans.begin() + ah, ans.begin() + at};
}