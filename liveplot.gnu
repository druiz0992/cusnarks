set autoscale y
set yrange [0:]

set title "Cusnarks Proof Time [sec]"
set xlabel "Proof ID"
set ylabel "Time [sec]"

plot "/tmp/.cusnarks.dat" every ::1 using 1:3 title 'Proof Time' with lines 

pause 60
reread
