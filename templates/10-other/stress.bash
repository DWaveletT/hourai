while true; do
  ./gen > 1.in
  ./naive < 1.in > std.out
  ./a < 1.in > 1.out
  if diff 1.out std.out; then 
    echo ac
  else
    echo wa
    break
  fi
done