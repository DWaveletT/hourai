#include "2d.cpp"

struct circ: pp { db r; };  // 圆形

circ circ_i(pp a, pp b, pp c) {
  db A = dis(a, b), B = dis(a, c), C = dis(a, b);
  return {
    (a * A + b * B + c * C) / (A + B + C),
    abs((b - a) * (c - a)) / (A + B + C)
  };
}   // 三点确定内心

circ circ_o(pp a, pp b, pp c) {
  pp bc = c - b, ca = a - c, ab = b - a;
  pp o = (b + c - r90(bc) * (ca % ab) / (ca * ab)) / 2;
  return {o, (a - o).abs()};
}   // 三点确定外心

circ minimal(vector <pp> V){  // 最小圆覆盖
  shuffle(V.begin(), V.end(), MT);
  pp  o = V[0]; double r = 0;
  for(int i = 0;i < V.size();++ i) {
    if (dis(o, V[i]) < r) continue;
    o = (V[i] + V[0]) / 2, r = dis(V[i], V[0]) / 2;
    for(int j = 0;j < i;++ j) {
      if (dis(o, V[j]) < r) continue;
      o = (V[i] + V[j]) / 2, r = dis(V[i], V[j]) / 2;
      for(int k = 0;k < j;++ k) {
        if (dis(o, V[k]) < r) continue;
        o = circ_o(V[i], V[j], V[k]), r = dis(o, V[i]);
      }
    }
  }
  return { o, r };
}