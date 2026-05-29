"""
Created on Thu Jan 23 14:37:38 2025
@author: Burak
"""

from mosek import Env, variabletype, boundkey, conetype, soltype, streamtype
import sys
import gurobipy as gu
import pyomo.environ as pyo
import random


def RemoveDuplicates(allPoints, tolerance=1.0e-6):
    points = []
    for (x, y, z) in allPoints:
        if not any((x-x_)**2 + (y-y_)**2 + (z-z_)**2 <= tolerance for (x_, y_, z_) in points):
            points.append((x, y, z))
    return points

def NonconvexPlus(y1Lo,y1Up, y2Lo, y2Up, y3Lo, y3Up, beta, kappa, y1Obj, y2Obj, y3Obj):
    model = pyo.ConcreteModel()
    
    model.y1 = pyo.Var(bounds=(y1Lo,y1Up))
    model.y2 = pyo.Var(bounds=(y2Lo,y2Up))
    model.y3 = pyo.Var(bounds=(0,y3Up))
    
    model.constraints = pyo.ConstraintList()    
    model.constraints.add(model.y1 - model.y2 == beta*(model.y3**kappa))
    
    model.objective = pyo.Objective(expr = y1Obj*model.y1 + y2Obj*model.y2 + y3Obj*model.y3)

    solver=pyo.SolverFactory('gurobi')
    solver.solve(model, tee=False)#.write()
    # model.write(filename='plus')
    # model.pprint(ostream=open('fooAfter.txt', 'w'))
    return pyo.value(model.objective)

def NonconvexMinus(y1Lo,y1Up, y2Lo, y2Up, y3Lo, y3Up, beta, kappa, y1Obj, y2Obj, y3Obj):
    model = pyo.ConcreteModel()
    
    model.y1 = pyo.Var(bounds=(y1Lo,y1Up))
    model.y2 = pyo.Var(bounds=(y2Lo,y2Up))
    model.y3 = pyo.Var(bounds=(y3Lo,-0.0))
    
    model.constraints = pyo.ConstraintList()    
    model.constraints.add(model.y2 - model.y1 == beta*((-model.y3)**kappa))
    
    model.objective = pyo.Objective(expr = y1Obj*model.y1 + y2Obj*model.y2 + y3Obj*model.y3)
    
    # model.pprint(ostream=open('fooAfter.txt', 'w'))
    solver=pyo.SolverFactory('gurobi')
    solver.solve(model, tee=False)#.write()
    # model.write(filename='minus')
    # model.pprint(ostream=open('fooAfter.txt', 'w'))
    return pyo.value(model.objective)

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()
    
def AddConvexCombination(task, y, mu, points):
    P = len(points)
    numvar = task.getnumvar()
    task.appendvars(P)
    gamma = [numvar + p for p in range(P)]
    
    numcon = task.getnumcon()
    task.appendcons(1+3)
    
    for p in range(P):
        task.putvarbound(gamma[p], boundkey.lo, 0, +0)
        
    for j in [0,1,2]:
        task.putarow(numcon + j, gamma + [y[j]], [points[p][j] for p in range(P)] + [-1])            
        task.putconbound(numcon + j, boundkey.fx, 0, 0)
    task.putarow(numcon + 3, gamma + [mu], [1]*P + [-1])            
    task.putconbound(numcon + 3, boundkey.fx, 0, 0)
        
        
def AddDisjunctionPlusMsk(task, y, z, mu, beta, kappa, yUp, points):
    for j in range(2):
        task.putvarbound(y[j], boundkey.fr, -0, +0)
        task.putvarbound(z[j], boundkey.lo, 0, +0) 
    task.putvarbound(y[2], boundkey.lo, 0, +0)
        
    numcon = task.getnumcon()
    task.appendcons(2)
    
    task.putarow(numcon, [y[0], y[1], z[0]], [1, -1, -1])            
    task.putconbound(numcon, boundkey.fx, 0, 0)
    
    task.putarow(numcon+1, [mu, z[1]], [beta**(1.0/(1.0-kappa)), -1])            
    task.putconbound(numcon+1, boundkey.fx, 0, 0)
    
    task.appendcone(conetype.ppow, 1.0/kappa, [z[0], z[1], y[2]])
    
    AddConvexCombination(task, y, mu, points)
    
    # task.appendcons(2)
    # task.putarow(numcon+2, [y[2], z[0]], [1, -(yUp**(1-kappa))/beta])  # in the case of identical bounds          
    # task.putconbound(numcon+2, boundkey.lo, 0, +0)
    
def AddDisjunctionMinusMsk(task, y, z, mu, beta, kappa, yLo, points):
    for j in range(2):
        task.putvarbound(y[j], boundkey.fr, -0, +0)
        task.putvarbound(z[j], boundkey.lo, 0, +0) 
    task.putvarbound(y[2], boundkey.up, -0, 0)
        
    numcon = task.getnumcon()
    task.appendcons(2)
    
    task.putarow(numcon, [y[0], y[1], z[0]], [-1, 1, -1])            
    task.putconbound(numcon, boundkey.fx, 0, 0)
    
    task.putarow(numcon+1, [mu, z[1]], [beta**(1.0/(1.0-kappa)), -1])            
    task.putconbound(numcon+1, boundkey.fx, 0, 0)
    
    task.appendcone(conetype.ppow, 1.0/kappa, [z[0], z[1], y[2]])
    
    AddConvexCombination(task, y, mu, points)
    
    # task.appendcons(2)
    # task.putarow(numcon+2, [y[2], z[0]], [1, ((-yLo)**(1-kappa))/beta]) # in the case of identical bounds              
    # task.putconbound(numcon+2, boundkey.up, 0, +0)
    
    
def AddDisjunctionBoundsMsk(task, x, xLo, xUp, mu):
    task.putvarbound(x, boundkey.fr, -0, +0)
        
    numcon = task.getnumcon()
    task.appendcons(2)
    
    task.putarow(numcon, [x, mu], [1, -xUp])            
    task.putconbound(numcon, boundkey.up, -0, 0)
    task.putarow(numcon+1, [x, mu], [1, -xLo])            
    task.putconbound(numcon+1, boundkey.lo, 0, +0)
    

def ExtremePointsPosFlow(y1Lo, y1Up, y2Lo, y2Up, y3Up, beta, kappa):
    # extreme points of 
    # y1lo <= y1 <= y1up
    # y2lo <= y2 <= y2up
    # 0 <= y3 <= y3up
    # y1-y2 <= beta*z^kappa
    
    if y3Up < 0:
        return []
    
    allExtrPointsPosFlow = []
    for y2tilde in [y2Lo, y2Up]:
        for y3tilde in [0, y3Up]:
            y_lo = y1Lo
            y_up = min(y1Up, y2tilde+beta*(y3tilde**kappa))
            if y_lo <= y_up:
                allExtrPointsPosFlow += [(y_lo, y2tilde, y3tilde), (y_up, y2tilde, y3tilde)]
            
    for y1tilde in [y1Lo, y1Up]:
        for y3tilde in [0, y3Up]:
            y_lo = max(y2Lo, y1tilde-beta*(y3tilde**kappa))
            y_up = y2Up
            if y_lo <= y_up:
                allExtrPointsPosFlow += [(y1tilde, y_lo, y3tilde), (y1tilde, y_up, y3tilde)]
            
    for y1tilde in [y1Lo, y1Up]:
        for y2tilde in [y2Lo, y2Up]:
            y_lo = 0
            if y1tilde >= y2tilde:
                y_lo = ((y1tilde-y2tilde)/beta)**(1.0/kappa)
            y_up = y3Up
            if y_lo <= y_up:
                allExtrPointsPosFlow += [(y1tilde, y2tilde, y_lo), (y1tilde, y2tilde, y_up)]
                
    return RemoveDuplicates(allExtrPointsPosFlow)


def ExtremePointsNegFlow(y1Lo, y1Up, y2Lo, y2Up, y3Lo, beta, kappa):
    extrPointsNegFlow = []
    if y3Lo < 0:
        extrPointsNegFlowTemp = ExtremePointsPosFlow(y2Lo, y2Up, y1Lo, y1Up, -y3Lo, beta, kappa)
        for (x,y,z) in extrPointsNegFlowTemp:
            extrPointsNegFlow.append((y,x,-z))
    
    return extrPointsNegFlow
    


def POWr(y1Lo, y1Up, y2Lo, y2Up, y3Lo, y3Up, beta, kappa, y1Obj, y2Obj, y3Obj):
    # Variables [pi, alpha, beta, u, v, w]
    numvar = 15
    y=[0,1,2]
    y_p=[3,4,5] 
    y_m=[6,7,8] 
    z_p=[9,10]
    z_m=[11,12]
    mu=[13,14]
    
    yObj=[y1Obj, y2Obj, y3Obj]
    yLo=[y1Lo, y2Lo, y3Lo]
    yUp=[y1Up, y2Up, y3Up]
    
    objval=0
    env = Env()
    with env.Task() as task:
        # task.set_Stream(streamtype.log, streamprinter)
        task.appendvars(numvar)
        
        for j in range(3):
            task.putvarbound(y[j], boundkey.ra, yLo[j], yUp[j])
            task.putcj(y[j], yObj[j])
            
        
        # add convex combination constraints
        task.appendcons(4)
        for j in range(3):
            task.putarow(j, [y[j],y_p[j],y_m[j]], [-1,1,1])            
            task.putconbound(j, boundkey.fx, 0, 0)
        
        task.putarow(3, mu, [1,1])            
        task.putconbound(3, boundkey.fx, 1, 1)
        
        # add disjunction plus constraints
        task.putvarbound(mu[0], boundkey.lo, 0, +0)
        for j in range(3):
            AddDisjunctionBoundsMsk(task, y_p[j], yLo[j], yUp[j], mu[0])
            
        AddDisjunctionPlusMsk(task, y_p, z_p, mu[0], beta, kappa, yUp[2], ExtremePointsPosFlow(y1Lo, y1Up, y2Lo, y2Up, y3Up, beta, kappa))
        
        
        # add disjunction minus constraints
        task.putvarbound(mu[1], boundkey.lo, 0, +0)
        for j in range(3):
            AddDisjunctionBoundsMsk(task, y_m[j], yLo[j], yUp[j], mu[1])
            
        AddDisjunctionMinusMsk(task, y_m, z_m, mu[1], beta, kappa, yLo[2], ExtremePointsNegFlow(y1Lo, y1Up, y2Lo, y2Up, y3Lo, beta, kappa))
        
    
        # print("sdf")
        # print(task.getnumcon())
        
        
        task.writedata("pow.opf")
        task.optimize()
        # task.writedata("plus1.sol")
        
        
    
        
        # Print a summary containing information
        # about the solution for debugging purposes
        task.solutionsummary(streamtype.msg)
        prosta = task.getprosta(soltype.itr)
        solsta = task.getsolsta(soltype.itr)

        # print(prosta)
        # print(solsta)
        
        # # Output a solution
        # xx = [0.] * numvar
        # task.getxx(soltype.itr,
        #            xx)

        # if solsta == solsta.optimal:
        #     print("Optimal solution: %s" % xx[0:numvar])
        # elif solsta == solsta.dual_infeas_cer:
        #     print("Primal or dual infeasibility.\n")
        # elif solsta == solsta.prim_infeas_cer:
        #     print("Primal or dual infeasibility.\n")
        # elif solsta.unknown:
        #     print("Unknown solution status")
        # else:
        #     print("Other solution status")
        
        objval = task.getprimalobj(soltype.itr)
        
        
        # prosta = task.getprosta(soltype.itg)
        # print(prosta)
        
        # solsta = task.getsolsta(soltype.itg)
        # print(solsta)
        # task.getxxslice(soltype.itg, 0, 1, yy)
        # print(yy)
    return objval
    

def AddDisjunctionV_PlusGu(model, x,y,a,b, kappa, xLo,xUp, mu):
    if xLo > xUp:
        model.addConstr(mu == 0)
        x.ub=0
        y.ub=0
        return
        
    AddDisjunctionBoundsGu(model, x, xLo, xUp, mu)
    temp = ( (xUp/b)**kappa - (xLo/b)**kappa )/( xUp - xLo + 1.0e-6) # separate the case if xUp = xLo
    model.addConstr(a*temp*x - y + (-a*temp*xLo+a*( (xLo/b)**kappa - 1))*mu <= 0 )
    
    u = model.addVar()
    model.addConstr( y == a*(u-mu) )
    
    v1 = model.addVar()
    v2 = model.addVar()
    model.addConstr(u*u<=v1*mu)
    model.addConstr(v1*v1<=v2*mu)
    model.addConstr(b*v2*v2<=x*mu)
    
def AddDisjunctionW_PlusGu(model, x,y,a,b, kappa, xLo,xUp, mu):
    if xLo > xUp:
        model.addConstr(mu == 0)
        x.ub=0
        y.ub=0
        return
        
    AddDisjunctionBoundsGu(model, x, xLo, xUp, mu)
    temp = ( (b/xUp)**kappa - (b/xLo)**kappa )/( xUp - xLo + 1.0e-6) # separate the case if xUp = xLo
    model.addConstr(a*temp*x - y + (-a*temp*xLo+a*( (b/xLo)**kappa - 1))*mu >= 0 )
    
    u = model.addVar()
    model.addConstr( y == a*(u-mu) )
    
    v1 = model.addVar()
    v2 = model.addVar()
    v3 = model.addVar()
    model.addConstr(u*v1>=mu*mu)
    model.addConstr(v2*mu>=v1*v1)
    model.addConstr(v3*mu>=v2*v2)
    model.addConstr(x*u>=v3*v3*b)
    
    # v1 = model.addVar()
    # v2 = model.addVar()
    # model.addConstr(u*x>=v1*v1*b)
    # model.addConstr(u*v1>=v2*v2)
    # model.addConstr(u*v2>=mu*mu)
    
    
    
def AddDisjunctionBoundsGu(model, x, xLo, xUp, mu):
    model.addConstr(x - xUp*mu <= 0)
    model.addConstr(x - xLo*mu >= 0)
    
def SOCrPlus(piLo,piUp, pjLo, pjUp, rLo, rUp, H, kappa, fUp, piObj, pjObj, fObj, lObj):
    model = gu.Model()
    model.setParam('OutputFlag', False)
    
    D=5
    pi_h=model.addVars(D+1, name="pi")
    pj_h=model.addVars(D+1, name="pj")
    f_h=model.addVars(D+1, name="f")
    l_h=model.addVars(D+1, name="l")
    mu_h=model.addVars(D+1, name="mu")
    mu_h[0].ub=0
    
    pi_h[0].lb = piLo
    pi_h[0].ub = piUp
    pi_h[0].Obj = piObj
    
    pj_h[0].lb = pjLo
    pj_h[0].ub = pjUp
    pj_h[0].Obj = pjObj
    
    f_h[0].ub = fUp
    f_h[0].Obj = fObj
    
    l_h[0].Obj = lObj
    
    model.addConstr(rLo*pi_h[0] - pj_h[0] <= 0)
    model.addConstr(pj_h[0] - rUp*pi_h[0] <= 0)
        
    # add convex combination constraints
    model.addConstr( pi_h[0] - gu.quicksum(pi_h[d] for d in range(1,D+1)) == 0)
    model.addConstr( pj_h[0] - gu.quicksum(pj_h[d] for d in range(1,D+1)) == 0)
    model.addConstr( f_h[0] - gu.quicksum(f_h[d] for d in range(1,D+1)) == 0)
    model.addConstr( l_h[0] - gu.quicksum(l_h[d] for d in range(1,D+1)) == 0)
    model.addConstr( gu.quicksum(mu_h[d] for d in range(1,D+1)) == 1)
        
    # add disjunction 1 constraints
    model.addConstr(rLo*pi_h[1] - pj_h[1] <= 0)
    model.addConstr(pj_h[1] - rUp*pi_h[1] <= 0)
        
    AddDisjunctionBoundsGu(model, pi_h[1], piLo, piUp, mu_h[1])
    AddDisjunctionBoundsGu(model, pj_h[1], pjLo, pjUp, mu_h[1])
    f_h[1].ub=0
    l_h[1].ub=0
    
    
    # mu_h[1].ub=0
    # mu_h[2].ub=0
    # mu_h[3].ub=0
    # mu_h[4].ub=0
    # mu_h[5].ub=0
    
    # add disjunction 2 constraints
    AddDisjunctionBoundsGu(model, pi_h[2], piLo, piLo, mu_h[2])
    AddDisjunctionBoundsGu(model, f_h[2], fUp, fUp, mu_h[2])
    AddDisjunctionV_PlusGu(model, pj_h[2],l_h[2],H*fUp,piLo, kappa, 
                          max(pjLo, piLo*rLo),min(pjUp, piLo*rUp), mu_h[2])
    
    
    model.write("socr.lp")
    model.optimize()
    # if(model.status == 4):
    #     model.setParam('DualReductions', 0)
    #     model.optimize()
    #     print(model.status)
    return model.ObjVal

def ShootingExperiment(y1Lo,y1Up, y2Lo, y2Up, y3Lo, y3Up, beta, kappa, N=100):
    nonconvexPlus = []
    nonconvexMinus = []
    powr = []
    socr = []
    for n in range(N):
        y1Obj=random.uniform(-1, 1) 
        y2Obj=random.uniform(-1, 1)
        y3Obj=random.uniform(-1, 1)
        nonconvexPlus.append(NonconvexPlus(y1Lo,y1Up, y2Lo, y2Up, y3Lo, y3Up, beta, kappa, y1Obj, y2Obj, y3Obj))
        nonconvexMinus.append(NonconvexMinus(y1Lo,y1Up, y2Lo, y2Up, y3Lo, y3Up, beta, kappa, y1Obj, y2Obj, y3Obj))
        powr.append(POWr(y1Lo,y1Up, y2Lo, y2Up, y3Lo, y3Up, beta, kappa, y1Obj, y2Obj, y3Obj))
        socr.append(0)
        
        print(str(round( min(nonconvexPlus[n],nonconvexMinus[n]),6)) + "\t" +str(round(powr[n],6)) + "\t" + str(round(socr[n],6)))
    
    # for n in range(N):
    #     # print(str(nonconvex[n]) + "\t" + str(powr[n]))
    #     print(str(nonconvex[n]-powr[n]))
    #     # print(str(round(nonconvex[n],6)) + "\t" + str(round(powr[n],6)))
        
# ShootingExperiment(y1Lo=0,y1Up=4, y2Lo=0, y2Up=4, y3Lo=-2, y3Up=2, beta=1, kappa=2)
ShootingExperiment(y1Lo=1,y1Up=3, y2Lo=0, y2Up=5, y3Lo=-2, y3Up=2, beta=1, kappa=2)