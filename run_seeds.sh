#for i in 0 1 2 3 4 5 6 7 8 9; do
for i in $(seq 2 9); do
    python python/gps/gps_main.py $1/$i -q
done
