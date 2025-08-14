#include "../header.cpp"
db simpson(db (*f)(db), db l, db r){
  return (r - l) / 6 *
    (f(l) + 4 * f((l + r) / 2) + f(r));
}
db adapt_simpson(db (*f)(db), db l, db r, db EPS, int step){
  db mid = (l + r) / 2;
  db w0 = simpson(f, l, r);
  db w1 = simpson(f, l, mid);
  db w2 = simpson(f, mid, r);
  if(fabs(w0 - w1 - w2) < EPS && step < 0)
    return w1 + w2;
  else return
    adapt_simpson(f, l, mid, EPS, step - 1) +
    adapt_simpson(f, mid, r, EPS, step - 1);
}
