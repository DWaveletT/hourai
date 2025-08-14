export CXXFLAGS='-g -Wall -fsanitize=address,undefined -Dzwb -std=gnu++20'
mk() { g++ -O2 -Dzwb -std=gnu++20 $1.cpp -o $1; }
ulimit -s 1048576
ulimit -v 1048576
