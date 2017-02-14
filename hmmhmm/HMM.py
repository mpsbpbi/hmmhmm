import numpy as np
import math

################################
class HMMState:
    def __init__(self):
        self.name = "HMMState"
        self.slient = False
        self.out = []   # emisson prob
        self.nexts = [] # next state
        self.nextp = [] # prob of next state
        self.targetOutput = "A"
        self.prevs = [] # previous state that arrive here
        self.prevp = [] # prob previous state that arrive here
        self.prevComputed=False

    def __str__(self):
        tmp=[]
        tmp.append("################")
        tmp.append("%s silent:%s" % (self.name, self.silent))
        tmp.append("out: %s" % ",".join([str(xx) for xx in self.out]))
        tmp.append("nexts: %s" % ",".join([str(xx) for xx in self.nexts]))
        tmp.append("nextp: %s" % ",".join([str(xx) for xx in self.nextp]))
        tmp.append("prevs: %s" % ",".join([str(xx) for xx in self.prevs]))
        tmp.append("prevp: %s" % ",".join([str(xx) for xx in self.prevp]))
        tmp.append("prevComputed: %s" % str(self.prevComputed))
        return("\n".join(tmp))


################################
class HMMParameters:

        #### define the transisition and output given the base identity
        ## Generic probabilities
        # def ftrans( base ):
        #     """pm,pb,ps,pd for base ACGT"""
        #     return( (0.8,0.05,0.05,0.1) )
        # def foutm( base ):
        #     tmp = [0.0,0.0,0.0,0.0]
        #     tmp[ self.alphaToInd[baseIden] ] =1.0
        #     return(tmp)
        # def foutb( base ):
        #     tmp = [0.0,0.0,0.0,0.0]
        #     tmp[ self.alphaToInd[baseIden] ] =1.0
        #     return(tmp)
        # def fouts( base ):
        #     tmp = [1.0/3.0,1.0/3.0,1.0/3.0,1.0/3.0]
        #     tmp[ self.alphaToInd[baseIden] ] =0.0
        #     return(tmp)

        # # nigel's sim parameters
        # def ftrans( base ):
        #     """pm,pb,ps,pd for base ACGT"""
        #     p_m  = 0.95583140484751283
        #     p_d  = 0.00097238955012494488
        #     p_b  = 0.029256323818866534
        #     p_s  = 0.013939881783495679
        #     return( (p_m,p_b,p_s,p_d) )
        # def foutm( base ):
        #     eps  = 0.00505052456472967
        #     tmp = [eps/3,eps/3,eps/3,eps/3]
        #     tmp[ self.alphaToInd[baseIden] ] =1.0-eps
        #     return(tmp)
        # def foutb( base ):
        #     tmp = [0.0,0.0,0.0,0.0]
        #     tmp[ self.alphaToInd[baseIden] ] =1.0
        #     return(tmp)
        # def fouts( base ):
        #     tmp = [1.0/3.0,1.0/3.0,1.0/3.0,1.0/3.0]
        #     tmp[ self.alphaToInd[baseIden] ] =0.0
        #     return(tmp)

    @staticmethod
    def parseDat( holder, data ):
        for ll in data.splitlines():
            ff = ll.split()
            pp = np.array([float(xx) for xx in ff[1:5]])
            pp = pp/sum(pp) # sum to 1.0 after getting rid of false pseudo-count for "-"
            holder[ff[0]] = pp

    def __init__(self, mPmfdat=None, bPmfdat=None, sPmfdat=None, transdat=None):
        #             A           C           G          T           - N CTX
        if mPmfdat is None:
            mPmfdat = """AA 9.988101e-01 9.373250e-04 1.403341e-04 7.007589e-05 4.219841e-05 0  AA
CC 2.146259e-03 9.976724e-01 1.083980e-04 3.663233e-05 3.631252e-05 0  CC
GG 6.026106e-05 3.617703e-05 9.991735e-01 6.943761e-04 3.570186e-05 0  GG
TT 1.050065e-04 1.228631e-04 9.038600e-04 9.988269e-01 4.138490e-05 0  TT
NA 9.991590e-01 6.687298e-04 8.842021e-05 6.981615e-05 1.403361e-05 0  NA
NC 9.602324e-04 9.981416e-01 6.114106e-05 8.252746e-04 1.176365e-05 0  NC
NG 5.913309e-04 3.908480e-05 9.977593e-01 1.598439e-03 1.184521e-05 0  NG
NT 4.828707e-05 9.150200e-05 1.027760e-03 9.988184e-01 1.410018e-05 0  NT
"""
        self.mPmf = {}
        HMMParameters.parseDat(self.mPmf, mPmfdat)

        if bPmfdat is None:
            bPmfdat = """AA 0.987057271 0.003235682 0.003235682 0.003235682 0.003235682 0  AA
CC 0.004642770 0.981428920 0.004642770 0.004642770 0.004642770 0  CC
GG 0.005929860 0.005929860 0.976280560 0.005929860 0.005929860 0  GG
TT 0.002406837 0.002406837 0.002406837 0.990372650 0.002406837 0  TT
NA 0.991693486 0.002076628 0.002076628 0.002076628 0.002076628 0  NA
NC 0.002883788 0.988464849 0.002883788 0.002883788 0.002883788 0  NC
NG 0.002613835 0.002613835 0.989544660 0.002613835 0.002613835 0  NG
NT 0.002356228 0.002356228 0.002356228 0.990575089 0.002356228 0  NT
"""
        self.bPmf = {}
        HMMParameters.parseDat(self.bPmf, bPmfdat)

        if sPmfdat is None:
            sPmfdat = """AA 0.02011942 0.336736109 0.188141521 0.43488353 0.020119421 0  AA
CC 0.43867758 0.016881951 0.340925420 0.18663310 0.016881951 0  CC
GG 0.32834646 0.131647516 0.026688865 0.48662830 0.026688865 0  GG
TT 0.25124307 0.281135027 0.433736630 0.01694264 0.016942636 0  TT
NA 0.01651072 0.154492514 0.344974992 0.46751106 0.016510719 0  NA
NC 0.34305431 0.008586935 0.331614465 0.30815736 0.008586935 0  NC
NG 0.37580837 0.180552820 0.007853463 0.42793189 0.007853463 0  NG
NT 0.36329025 0.330922178 0.271563220 0.01711218 0.017112177 0  NT
"""
        self.sPmf = {}
        HMMParameters.parseDat(self.sPmf, sPmfdat)

        #  CTX     Match      Branch        Stick       Delete
        if transdat is None:
            transdat = """AA 0.9747334 0.012506070 0.0018391528 0.0109214122
CC 0.9824151 0.007507524 0.0019354064 0.0081419332
GG 0.9837446 0.005661715 0.0011370914 0.0094566157
TT 0.9687964 0.016461413 0.0021650551 0.0125771129
NA 0.9918225 0.006633355 0.0007715309 0.0007726506
NC 0.9942187 0.003997322 0.0013002356 0.0004837361
NG 0.9935532 0.004476972 0.0014435620 0.0005262866
NT 0.9924876 0.005870878 0.0007467905 0.0008947449
"""
        self.trans = {}
        HMMParameters.parseDat(self.trans,transdat)

    def toctx(self, ctx):
        if ctx[0]==ctx[1]:
            return(ctx)
        else:
            ctx = "N"+ctx[1]
            return(ctx)

    def ftrans( self, ctx ):
        return(self.trans[self.toctx(ctx)])

    def foutm( self, ctx ):
        return(self.mPmf[self.toctx(ctx)])

    def foutb( self, ctx ):
        return(self.bPmf[self.toctx(ctx)])

    def fouts( self, ctx ):
        return(self.sPmf[self.toctx(ctx)])


################################        
class HMM:
    def __init__( self ):
        self.state = []
        self.nameToStateIndex = {}
        self.alphabet = "ACGT"
        self.alphaToInd = {"A":0, "C":1, "G":2, "T":3}
        self.memo = {}
        self.memoForward = {}

    def key(self, state,length):
        return("%s-%s" % (str(state),str(length)))

    def key(self, state,begin,end):
        return("%s-%s-%s" % (str(state),str(begin),str(end)))

    ################################
    def mylog(self, xx):
        if (xx<=0.0):
            return(99E+99)
        else:
            return(math.log(xx))

    def emLL(self, xx):
        tmp = 0.0
        for tt in xx:
            if tt>0.0: tmp=tmp+tt*math.log(tt)
        return(tmp)

    def emLL2(self, xx):
        tmp = 0.0
        for tt in xx:
            if tt>0.0: tmp=tmp+tt*math.log(tt)*math.log(tt)
        return(tmp)

    ################################
    def __str__(self):
        tmp=[]
        for ii in range(len(self.state)):
            tmp.append("%d %s" % (ii, str(self.state[ii])))
        return("\n".join(tmp))

    ################################
    def setSize( self, inSize ):
        self.state = [ HMMState() for ii in range(inSize)]

    ################################
    def generate( self ):
        """Generate random strings from the HMM"""
        res = {}
        res["prob"]= 0.0
        #res["seq"]=[]

        prob=1.0
        here = 0
        while here != (len(self.state)-1):
            this = self.state[here]
            if  not this.silent:
                #print "output"
                ch = np.random.choice(len(this.out), p=this.out)
                prob=prob*this.out[ch]
                #res["seq"].append( (self.alphabet[ch], this.name, prob) )
            else:
                #res["seq"].append( ( "-", this.name, prob) )
                pass

            #print "trans"
            ch = np.random.choice(len(this.nextp), p=this.nextp)
            prob=prob*this.nextp[ch]
            here = this.nexts[ch]
            #print "here", here

        res["prob"] = prob
        return(res)

    ################################
    def pacbioMBSD( self, prefix, toIndex, ctx, params ):
        """
Generate 5 state sub-hmm representing Match Branch Stick Delete for one base:
0=Begin
1=Match
2=Branch
3=Stick
4=Delete
        """

        (pm,pb,ps,pd) = params.ftrans(ctx)

        thisInd = len(self.state)

        this = HMMState()
        this.name = prefix+"_begin"
        this.silent = True
        this.nexts.append(thisInd+1)
        this.nextp.append(1.0*pm)
        this.nexts.append(thisInd+2)
        this.nextp.append(1.0*pb)
        this.nexts.append(thisInd+3)
        this.nextp.append(1.0*ps)
        this.nexts.append(thisInd+4)
        this.nextp.append(1.0*pd)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_match"
        this.silent = False
        this.nexts.append(toIndex)
        this.nextp.append(1.0)
        this.out = params.foutm(ctx)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_branch"
        this.silent = False
        this.nexts.append(thisInd+1)
        this.nextp.append(1.0*pm)
        this.nexts.append(thisInd+2)
        this.nextp.append(1.0*pb)
        this.nexts.append(thisInd+3)
        this.nextp.append(1.0*ps)
        this.nexts.append(thisInd+4)
        this.nextp.append(1.0*pd)
        this.out = params.foutb(ctx)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_stick"
        this.silent = False
        this.nexts.append(thisInd+1)
        this.nextp.append(1.0*pm)
        this.nexts.append(thisInd+2)
        this.nextp.append(1.0*pb)
        this.nexts.append(thisInd+3)
        this.nextp.append(1.0*ps)
        this.nexts.append(thisInd+4)
        this.nextp.append(1.0*pd)
        this.out = params.fouts(ctx)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_delete"
        this.silent = True
        this.nexts.append(toIndex)
        this.nextp.append(1.0)
        self.state.append(this)

################################
    def derive( self, thisstate, length):

        """returns the (totalProb, EXP(LL), and EXP[LL^2]) for thisstate deriving length string"""

        if self.key(thisstate,length) in self.memo:
            return self.memo[self.key(thisstate,length)]

        this = self.state[thisstate]

        # at end
        if this.name=="last":
            result = (1.0, 0.0, 0.0)
            return(result)

        if this.silent:
            newlength = length
            expEmLL  = 0.0
            expEmLL2 = 0.0
        else:
            newlength = length-1
            expEmLL = self.emLL(this.out)
            expEmLL2 = self.emLL2(this.out)

        # would require impossible derivation
        if newlength<0:
            result = (0.0, 99E+99, 99E+99)
            self.memo[self.key(thisstate,length)]=result
            return(result)

        # cycle through choices
        overallresult = [0.0, 0.0, 0.0]
        for ch in range(len(this.nexts)):

            transp = this.nextp[ch]
            rhs = self.derive( this.nexts[ch], newlength)

            A = self.mylog(transp) # log transition
            A2 = A*A               # log transistion squared
            B = expEmLL            # expected log emission likelihood
            B2= expEmLL2           # expected log squared emission likelihood
            C = rhs[1]             # EXP[LL] of next
            C2= rhs[2]             # EXP[LL^2] of next
            thisll =  transp*(A+B+C)
            thisll2 = transp*( A2 + B2 + C2 + 2*A*B + 2*A*C +2*B*C)

            thisprob = transp*rhs[0]

            if thisprob>0.0:
                overallresult[0]=overallresult[0]+thisprob
                overallresult[1]=overallresult[1]+thisll
                overallresult[2]=overallresult[2]+thisll2

        self.memo[self.key(thisstate,length)]=overallresult
        return(overallresult)

    ################################
    def backward( self, thisstate, begin, end):

        """returns the backward probability, best choice, summed prob of this
           state deriving [begin:end) of self.targetOutput.
           P(thisstate->[begin,end)), the most natural language
           interpretation (inside). End is usually fixed at the end of
           the string for regular grammars.
        """

        if self.key(thisstate,begin,end) in self.memo:
            return self.memo[self.key(thisstate,begin,end)]

        this = self.state[thisstate]

        #print "exploring", this.name, begin, end

        # at end
        if this.name=="last":
            if (end-begin)>0:
                result = (0.0,-1,0.0)
            else:
                result = (1.0,-1,1.0)
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if (end-begin)<0:
            #print "impossible derivation"
            result = (0.0,-1,0.0)
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if not this.silent:
            if (end-begin)<1:
                #print "impossible for non-silent"
                result = (0.0,-1,0.0)
                self.memo[self.key(thisstate,begin,end)]=result
                return(result)

            probEmission = this.out[self.alphaToInd[self.targetOutput[begin]]]
            nextbegin = begin+1
        else:
            probEmission = 1.0
            nextbegin= begin

        # cycle through choices
        overallresult = (-1.0,-1,-1)
        Zsum = 0.0
        for ch in range(len(this.nexts)):
            # this state make emssion, makes choice, and proceeds
            transp = this.nextp[ch]
            rhs = self.backward( this.nexts[ch], nextbegin, end)
            thisprob = probEmission*transp*rhs[0]
            Zsum = Zsum + probEmission*transp*rhs[2]
            if thisprob>overallresult[0]:
                overallresult = (thisprob, this.nexts[ch],-1)

        storeresult = (overallresult[0],overallresult[1],Zsum)
        self.memo[self.key(thisstate,begin,end)]=storeresult
        return(storeresult)

    ################################
    def computePrev( self, thisstate ):

        """Inital call to the start state (thisstate=0), compute prevs and
        prevp, the previous states that arrive at thisstate for use in
        the forward probability computation
        """

        if self.state[thisstate].prevComputed:
            # already done
            return()

        # cycle through nexts
        for ii in range(len(self.state[thisstate].nexts)):
            nn = self.state[thisstate].nexts[ii]
            pp = self.state[thisstate].nextp[ii]
            self.state[nn].prevs.append(thisstate)
            self.state[nn].prevp.append(pp)

        self.state[thisstate].prevComputed=True

        for ii in range(len(self.state[thisstate].nexts)):
            nn = self.state[thisstate].nexts[ii]
            # print "computePrev", thisstate, ">>", nn
            self.computePrev( nn )


    ################################
    def forward( self, thisstate, begin, end):

        """returns the forward probability, best choice, summed prob of the
        start state deriving [begin, end) and arriving in
        thisstate. begin is usually constant at 0.
        """

        # print "forward exploring", thisstate, begin, end

        if self.key(thisstate,begin,end) in self.memoForward:
            return self.memoForward[self.key(thisstate,begin,end)]

        this = self.state[thisstate]

        # at begining (0 by convention)
        if thisstate == 0:
            if (end-begin)>0:
                result = (0.0,-1,0.0)
            else:
                result = (1.0,-1,1.0)
            self.memoForward[self.key(thisstate,begin,end)]=result
            return(result)

        if (end-begin)<0:
            result = (0.0,-1,0.0)
            self.memoForward[self.key(thisstate,begin,end)]=result
            return(result)

        # cycle through choices
        overallresult = (-1.0,-1,-1)
        Zsum = 0.0
        for ch in range(len(this.prevs)):

            # previous state forward psf, prev make possible emission pse, prev trans to thisstate pst, stop

            prev = self.state[this.prevs[ch]]

            if not prev.silent:
                if (end-begin)<1:
                    continue
                else:
                    pse = prev.out[self.alphaToInd[self.targetOutput[end-1]]]
                    prevend = end-1
            else:
                pse = 1.0
                prevend = end

            pst = this.prevp[ch]

            psf = self.forward( this.prevs[ch], begin, prevend)

            thisprob = pse*pst*psf[0]
            Zsum = Zsum + pse*pst*psf[2]
            if thisprob>overallresult[0]:
                overallresult = (thisprob, this.prevs[ch],-1)

        storeresult = (overallresult[0],overallresult[1],Zsum)
        #print "forward storing", this.name, begin, end, storeresult
        self.memoForward[self.key(thisstate,begin,end)]=storeresult
        return(storeresult)
