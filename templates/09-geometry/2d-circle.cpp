#include "2d.cpp"

struct circ: pp { db r; };  // 圆形

circ circ_i(pp a, pp b, pp c) {
  db A = dis(a,b), B = dis(a,c), C = dis(a,b);
  return {
    (a * A + b * B + c * C) / (A + B + C),
    abs((b - a) * (c - a)) / (A + B + C)
  };
}   // 三点确定内心

circ circ_2pp(pp a, pp b){
  return {(a + b) / 2, dis(a, b) / 2};
}   // 两点确定直径
circ circ_3pp(pp a, pp b, pp c) {
  pp bc = c - b, ca = a - c, ab = b - a;
  pp o = (b + c - r90(bc) * (ca % ab) / (ca * ab)) / 2;
  return {o, (a - o).abs()};
}   // 三点确定外心

circ minimal(vector <pp> V){  // 最小圆覆盖
  shuffle(V.begin(), V.end(), MT);
  circ C(V[0], 0);
  for(int i = 0;i < V.size();++ i) {
    if (dis((pp)C, V[i]) < C.r) continue;
    C = circ_2pp(V[i], V[0]);
    for(int j = 0;j < i;++ j) {
      if (dis((pp)C, V[j]) < C.r) continue;
      C = circ_2pp(V[i], V[j]);
      for(int k = 0;k < j;++ k) {
        if (dis((pp)C, V[k]) < C.r) continue;
        C = circ_3pp(V[i], V[j], V[k]);
      }
    }
  }
  return C;
}