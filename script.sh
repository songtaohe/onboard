echo "test" > /home/pi/autorun/test
sleep 10
num=`ls -1 /home/pi/autorun/ | wc -l`
st=`expr ${num} % 600`
echo ${st}
mkdir /home/pi/autorun/videos${st}
raspivid -n -o /home/pi/autorun/videos${st}/record%04d.h264 --mode 4 -w 1640 -h 1232 -fps 30  -t 2000000 -sg 20000 -wr 600 -vs -ev -15 -ex antishake -ss 5000 
