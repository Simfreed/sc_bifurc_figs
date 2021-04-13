import grn_sim as sim
import argparse
import numpy as np
import os 

parser = argparse.ArgumentParser()

parser.add_argument("--dir",        type=str,   help  = "output subdirectory",          default='saddle_node_1') 
parser.add_argument("--nresp",      type=int,   help="# of resp genes",            default=100)
parser.add_argument("--ncell",      type=int,   help="# cells",                         default=100)
parser.add_argument("--scale",      type=float, help="scale for # proteins in cell",    default=20) #higher scale ==> lower noise
parser.add_argument("--hill",       type=float, help="hill coeff for both drivs",     default=2) #1==> no bifurcation
parser.add_argument("--beta",       type=int,   help="weight of resp genes",       default=1)

parser.add_argument("--seed",       type=int,   help="random number seed",              default=None)
parser.add_argument("--dt",         type=float, help="simulation step size",            default=0.01)
parser.add_argument("--dt_save",    type=float, help="step size to save at",            default=1000)
parser.add_argument("--tmax",       type=float, help="total simulation time",           default=500)

parser.add_argument("--m1_range",   type=float, help="m1 range for transcritical bifurc",       default=[2,4.1,0.1],    nargs = 3) 
parser.add_argument("--m2",         type=float, help="m2 for transcritical bifurc",             default=3)
parser.add_argument("--tau",        type=float, help="tau = 1/kd for transcritical bifurc",     default=1)

parser.add_argument("--run_pf",     dest='run_pf', action='store_true') # otherwise does the transcritical
parser.add_argument("--tau_range",  type=float, help="tau = 1/kd range for pitchfork bifurc",   default=[0.2,4.2,0.2],  nargs = 3 ) 
parser.add_argument("--m1",         type=float, help="m1 for pitchfork bifurc",                 default=1)
parser.add_argument("--m2_pf",      type=float, help="m2 for pf bifurc",                        default=1)

parser.add_argument("--init",      type=str,   help="""initial g pos; can be ['rand', 
        'min' (exact solution -- smallest if >1), 
        'max' (exact solution -- largest if >1)]. """,                                          default='rand') 
parser.add_argument("--g_range",   type=float, help="range to sample initial pos from",         default=[0,4],   nargs = 2 ) 

args = parser.parse_args()
np.random.seed(args.seed)

hill  = args.hill # hill coefficient for both genes
ndriv      = 2

nresp    = args.nresp
vtaus    = np.ones(nresp) 
betas    = args.beta*np.ones(nresp)
ks       = np.ones(nresp) 

alphas   = np.random.uniform(0,1,size=nresp)
drv_idxs = np.random.choice([0,1],size=nresp)

dt         = args.dt
dt_save    = args.dt_save #0.01 #500
tmax       = args.tmax
ncell      = args.ncell
nc_save    = ncell
scale      = args.scale
hs         = 2*np.ones(nresp)
ngenes     = ndriv + nresp


taus_var = np.arange(*args.tau_range)
m1s_var  = np.arange(*args.m1_range)

if args.run_pf:
    taus = taus_var
    m1s  = args.m1*np.ones(taus.shape[0])
    m2   = args.m2_pf
else:
    m1s = m1s_var
    m2  = args.m2
    taus = args.tau * np.ones(m1s.shape[0])

nb_var = taus.shape[0]


# directory setup
init = 'rand'


driv_traj   = np.zeros((nb_var, int(np.ceil(tmax/dt_save)), ndriv, nc_save))
resp_traj   = np.zeros((nb_var, int(np.ceil(tmax/dt_save)), nresp, nc_save))

driv_final  = np.zeros((nb_var, ndriv, nc_save))
resp_final  = np.zeros((nb_var, nresp, nc_save))


for i in range(nb_var):
    print(r'tau = {0:.2f}; m1 = {1:.2f}'.format(taus[i],m1s[i]))
    _,utraj,vtraj, u, v  = sim.langevin(m1s[i],m2, hill, hill, taus[i],
                               drv_idxs, alphas, betas, ks, hs, vtaus,
                               scale, ncell, dt, tmax, nc_save, dt_save, args.g_range, args.init)

    driv_traj[i]  = utraj
    resp_traj[i]  = vtraj

    driv_final[i] = u
    resp_final[i] = v

os.makedirs(args.dir, exist_ok=True)
gexp = np.hstack([driv_final, resp_final]).transpose((0,2,1))

np.save('{0}/gexp.npy'.format(args.dir),gexp)
np.save('{0}/taus.npy'.format(args.dir),taus)
np.save('{0}/m1s.npy'.format(args.dir),m1s)
np.save('{0}/bvars.npy'.format(args.dir),taus if args.run_pf else m1s)
np.save('{0}/alphas.npy'.format(args.dir),alphas)
np.save('{0}/betas.npy'.format(args.dir),betas)
np.save('{0}/ks.npy'.format(args.dir),ks)
np.save('{0}/driv_idxs.npy'.format(args.dir),drv_idxs)
np.save('{0}/vtaus.npy'.format(args.dir),vtaus)
np.save('{0}/dtraj.npy'.format(args.dir),driv_traj)
np.save('{0}/rtraj.npy'.format(args.dir),resp_traj)
