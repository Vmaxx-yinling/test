grep ' avg' age_0524.log | awk '{print $4}' >avgloss.txt
sed -i '/rate,/d' avgloss.txt
sed -i '/avg,/d' avgloss.txt
