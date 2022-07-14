# take the .py file name as input
kill $(ps aux |grep $@ |grep -v grep |awk '{print $2}')
