// From Let it Rot. 在整数坐标同样可用
#include "2d.cpp"

using Q = struct Quad *;
const pp arb(LLONG_MAX, LLONG_MAX);
struct Quad { 
  Q rot, o; pp p = arb; bool mark; 
  pp& F() { return r() -> p; } 
  Q& r() { return rot->rot; }
  Q prev() { return rot->o->rot; } 
  Q next() { return r()->prev(); }
} *H;
ll cross(pp a, pp b, pp c) {
  return (b - a) * (c - a);
}
// p 是否在 a, b, c 外接圆中
bool incirc(pp p, pp a, pp b, pp c) { 
  i80 p2 = p.norm(), A = a.norm() - p2, B = b.norm() - p2, C = c.norm() - p2;
  a = a - p, b = b - p, c = c - p; 
  return (a * b) * C + (b * c) * A + (c * a) * B > 0;
}
Q link(pp orig, pp dest) { 
  Q r = H ? H : new Quad{new Quad{new Quad{new Quad{0}}}}; 
  H = r -> o; r -> r() -> r() = r; 
  for(int i = 0;i < 4;++ i) 
    r = r -> rot, r -> p = arb,
    r -> o = i & 1 ? r : r -> r();
  r -> p = orig, r -> F() = dest;
  return r;
}
void splice(Q a, Q b) { 
  swap(a -> o -> rot -> o, b -> o -> rot ->o); 
  swap(a -> o, b -> o);
}
Q conn(Q a, Q b) { 
  Q q = link(a -> F(), b -> p); 
  splice(q, a -> next()); 
  splice(q -> r(), b); 
  return q;
}
pair<Q, Q> rec(const vector<pp> &s){ 
  int N = size(s); 
  if(N <= 3) { 
    Q a = link(s[0], s[1]), b = link(s[1], s.back()); 
    if(N == 2) return {a, a -> r()}; 
    splice(a -> r(), b);
    ll side = cross(s[0], s[1], s[2]); 
    Q c = side ? conn(b, a) : 0; 
    return { side < 0 ? c -> r() : a, side < 0 ? c : b -> r() }; 
  }
#define H(e) e -> F(), e -> p
#define valid(e) (cross(e->F(), H(base)) > 0) 
  int half = N / 2; 
  auto [ra, A]=rec({s.begin(), s.end()-half}); 
  auto [B, rb]=rec({s.end() - half, s.end()}); 
  while((cross(B -> p, H(A)) < 0 && (A = A -> next())) || (cross(A -> p, H(B)) > 0 && (B = B -> r() -> o))); 
  Q base = conn(B -> r(), A); 
  if(A -> p == ra -> p) ra = base -> r(); 
  if(B -> p == rb -> p) rb = base;
#define DEL(e, init, dir) \
  Q e = init -> dir; \
  if(valid(e)) \
    for(;incirc(e -> dir -> F(), H(base), e -> F());) { \
      Q t = e -> dir; \
      splice(e, e -> prev()); \
      splice(e -> r(), e -> r() -> prev()); \
      e -> o = H, H = e, e = t; \
    }
  for(;;) { 
    DEL(LC, base -> r(), o); 
    DEL(RC, base, prev()); 
    if(!valid(LC) && !valid(RC)) break; 
    if(!valid(LC) || (valid(RC) && incirc(H(RC), H(LC))))
      base = conn(RC, base -> r()); 
    else 
      base = conn(base -> r(), LC -> r()); 
  }
  return {ra, rb};
}

 // 返回若干逆时针三角形
vector<pp> deluanay(vector<pp> a){
  if((int)size(a) < 2) return {};
  sort(a.begin(), a.end()); // unique
  Q e = rec(a).first;
  vector<Q> q = {e};
  while(cross(e -> o -> F(), e -> F(), e -> p) < 0) e = e -> o;
  #define ADD { Q c = e; do { c -> mark = 1; a.push_back(c -> p); q.push_back(c -> r()), c = c -> next(); } while(c != e); } 
  ADD; a.clear();
  for(int qi = 0;qi < size(q);)
    if(!(e = q[qi++]) -> mark) ADD; 
  return a;
}
