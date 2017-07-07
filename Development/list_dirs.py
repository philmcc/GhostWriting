from os import listdir
from os.path import isfile, join


from subprocess import call


import glob
for f in glob.glob("*.txt"):
	print f
	#f = f[:-4]
	fileout = f[:-4] + "_"
	#args = "100 " + f +".txt " + f +"_"
	#call(["split -l ", args])
	#call (["split", "-l 1000 LawsonHenryArchibaldHertzberg.txt LawsonHenryArchibaldHertzberg_", shell=True])
	#os.system("split -l 1000 LawsonHenryArchibaldHertzberg.txt LawsonHenryArchibaldHertzberg_")
	#textfile = file.open(f, 'r')
	cmd = "split -l 1000 " + f + " " + fileout	
	print cmd
