import hmmhmm.HMM as HMM
import math
import sys

myhmm = HMM.HMM()
mypar = HMM.HMMParameters(open("/home/UNIXHOME/mbrown/mbrown/workspace2017Q1/ccs-hmm/unitem.CCS.170214.dat").read())

print "mypar", mypar

myref= sys.argv[1] #"AAAAAAAA"
for ii in range(len(myref)):
    if ii==0:
        ctx = myref[ii]+myref[ii] # hack 1st context is HP
    else:
        ctx = myref[ii-1:ii+1]
    myhmm.pacbioMBSD("ref%d%s"%(ii,myref[ii]),(ii+1)*5, ctx, mypar)
last = HMM.HMMState()
last.name="last"
last.silent=True
myhmm.state.append(last)


################################

if sys.argv[2] != "-":
    myhmm.targetOutput = sys.argv[2] # "AAAAAAA"
else:
    myhmm.targetOutput = "" # nothing

begin = 0
end = len(myhmm.targetOutput)
result = myhmm.backward(0,begin,end)
print result[2] # sum paths
