kill $(ps aux |grep $@ |grep -v grep |awk '{print $2}')
