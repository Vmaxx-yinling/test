ls >1117.out
vi 1117.out replace \n with space bar
vi 1117.out paste all files in 1117.out to alllog.txt
source 1117.out
awk '{line=""; for (i=5;i<=NF;i+=6) line=line (" " $i); print line;}' alllog.txt >age.txt
awk '{line=""; for (i=1;i<=NF;i+=2) line=line (" " $i); print line;}' alllog.txt >age.txt

vi mm.sh using the new list
check create_genderracesh.sh
source create_genderracesh.sh
source source_genderrace.sh

the result is saved in gender.out and race.out
