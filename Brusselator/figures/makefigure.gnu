res
se te po co enh eps "Times-Roman,27"
unse key
se xtics 1
se ytics 2.0
se xr [0:4]
se yr [0:8]
se format y "%.1f"
se out "volume.eps"
p 'volume_tgtA.txt' w lp pt 7 ps 2.5 lw 5 lc rgb "#E05451" t '','volume_tgtB.txt' w lp pt 5 ps 2.5 lw 5 lc rgb "#90C98B" t ''
