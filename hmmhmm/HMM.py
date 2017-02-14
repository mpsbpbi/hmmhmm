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

    @staticmethod
    def parseDat( src, target ):
        """Parse out the data containing model parameters"""
        res = {}
        for ll in src.splitlines():
            # if param is target
            if target in ll:
                ff = ll.split("\t")
                pp = np.array([float(xx) for xx in ff[0:4]])
                pp = pp/sum(pp) # just to make sure
                # store by CTX
                res[ff[6]] = pp
        return(res)

    def __init__(self, modelSpec):
        """Init according to modelSpec. This is the Sequel CCS 99% filtered estimates:
V1	V2	V3	V4	V5	V6	CTX	param	head	ver
0.999534526952176	0.000312886404921852	0.000152586615175579	2.77261500523443e-11	0	0	AA	mPmf	ACGT-N.ctx.param	unitemv1.0.0
3.68828022843349e-38	0.999736325976609	2.01732724887564e-60	0.000263674023391063	0	0	CC	mPmf	ACGT-N.ctx.param	unitemv1.0.0
1.25045994442458e-37	7.03271085106264e-52	1	5.72078625640215e-43	0	0	GG	mPmf	ACGT-N.ctx.param	unitemv1.0.0
2.34463964180455e-05	3.27631398082783e-05	0.000211417188567232	0.999732373275206	0	0	TT	mPmf	ACGT-N.ctx.param	unitemv1.0.0
0.999812477638156	9.03775448080541e-05	9.71448170361884e-05	4.86830324534904e-43	0	0	NA	mPmf	ACGT-N.ctx.param	unitemv1.0.0
6.58462076132639e-05	0.999550385774239	9.26493556594431e-47	0.000383768018147193	0	0	NC	mPmf	ACGT-N.ctx.param	unitemv1.0.0
0.000221126370686052	7.56315270883505e-42	0.999610510073569	0.000168363555744473	0	0	NG	mPmf	ACGT-N.ctx.param	unitemv1.0.0
1.55126181265036e-43	2.66017460649836e-06	0.000353716994026079	0.999643622831367	0	0	NT	mPmf	ACGT-N.ctx.param	unitemv1.0.0
1	0	0	0	0	0	AA	bPmf	ACGT-N.ctx.param	unitemv1.0.0
0	1	0	0	0	0	CC	bPmf	ACGT-N.ctx.param	unitemv1.0.0
0	0	1	0	0	0	GG	bPmf	ACGT-N.ctx.param	unitemv1.0.0
0	0	0	1	0	0	TT	bPmf	ACGT-N.ctx.param	unitemv1.0.0
1	0	0	0	0	0	NA	bPmf	ACGT-N.ctx.param	unitemv1.0.0
0	1	0	0	0	0	NC	bPmf	ACGT-N.ctx.param	unitemv1.0.0
0	0	1	0	0	0	NG	bPmf	ACGT-N.ctx.param	unitemv1.0.0
0	0	0	1	0	0	NT	bPmf	ACGT-N.ctx.param	unitemv1.0.0
0	0.172111133629895	0.26382938398206	0.564059482388045	0	0	AA	sPmf	ACGT-N.ctx.param	unitemv1.0.0
0.340664469091335	0	0.659335530908665	2.00088638711006e-21	0	0	CC	sPmf	ACGT-N.ctx.param	unitemv1.0.0
9.29240715236437e-51	3.24551699746993e-77	0	1	0	0	GG	sPmf	ACGT-N.ctx.param	unitemv1.0.0
0.141235903802323	0.609888250407643	0.248875845790034	0	0	0	TT	sPmf	ACGT-N.ctx.param	unitemv1.0.0
0	2.1699001368425e-09	0.466754387388619	0.533245610441481	0	0	NA	sPmf	ACGT-N.ctx.param	unitemv1.0.0
0.51073043218567	0	0.111713977007138	0.377555590807192	0	0	NC	sPmf	ACGT-N.ctx.param	unitemv1.0.0
0.443062623053773	2.49324006533154e-25	0	0.556937376946227	0	0	NG	sPmf	ACGT-N.ctx.param	unitemv1.0.0
0.616002768789131	0.148654509510346	0.235342721700523	0	0	0	NT	sPmf	ACGT-N.ctx.param	unitemv1.0.0
0.985040177228099	0.00321130155061021	0.000604138661946525	0.0111443825593445	0	0	AA	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
0.991322287541937	0.00132539786463229	0.000250849733490369	0.00710146485994044	0	0	CC	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
0.988641966382265	0.00173427371413266	0.000157835628755547	0.00946592427484642	0	0	GG	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
0.984341388468217	0.00342658264602537	0.000428083599271638	0.0118039452864865	0	0	TT	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
0.999071341714542	0.000663763136640228	0.000138165473496796	0.000126729675320616	0	0	NA	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
0.998768059815449	0.000704672010965535	0.000376647235922316	0.000150620937662864	0	0	NC	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
0.999016680294168	0.000598894382522373	0.000131507644811926	0.000252917678497769	0	0	NG	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
0.998635369888274	0.000809722496585358	0.000282752760086924	0.000272154855054138	0	0	NT	trans	Match.Branch.Stick.Delete.CTX.param	unitemv1.0.0
"""

        self.mPmf = HMMParameters.parseDat( modelSpec, "mPmf")
        self.bPmf = HMMParameters.parseDat( modelSpec, "bPmf")
        self.sPmf = HMMParameters.parseDat( modelSpec, "sPmf")
        self.trans = HMMParameters.parseDat( modelSpec, "trans")

    def __str__(self):
        res = []
        res.append( ">>>> HMMParameters" )
        res.append("----emission ACGT")
        res.append( "--mPmf" )
        for (k,v) in self.mPmf.items():
            res.append("%s %s" % (str(k),str(v)))

        res.append( "--bPmf")
        for (k,v) in self.bPmf.items():
            res.append("%s %s" % (str(k),str(v)))

        res.append( "--sPmf")
        for (k,v) in self.sPmf.items():
            res.append("%s %s" % (str(k),str(v)))
        res.append("----transition Match.Branch.Stick.Delete")
        for (k,v) in self.trans.items():
            res.append("%s %s" % (str(k),str(v)))
        res.append( "<<<< HMMParameters")
        return("\n".join(res))

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
