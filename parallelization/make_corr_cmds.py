import subprocess
import numpy as np

#### WRITE JOBS TO FILE ####
homedir = '/home/slf3348'
datdir  = '/projects/p31095/simonf/out/weinreb_2020/data'
cmdf    = '{0}/jobs/cmds/cmds.txt'.format(homedir)
acct    = 'p31095' #b1020
part    = 'normal' #b1020
time    = '3:59:59'
sbatchf = '{0}/jobs/run_cmd.sbatch'.format(homedir)
script  = '{0}/bifurc/haem/python/pseudotime_cov.py'.format(homedir)
ncpu    = 16

jobn      = 'neut_pst_corr'

args = {
        'min_cell_gexp':0,
        'min_ncells':9,
        'overlap':0.5
}

bin_szs   = np.array([1000,500,200,100])
tbifs     = np.array([109,218,547,1096])
tfs       = np.array([121, 244, 612, 1225])
tis       = tbifs - (tfs - tbifs)
nmaxt     = np.array([4, 10, 20, 20])

bin_szs   = np.array([500,200])
tis = [198, 492]
tfs = [202, 502]
nmaxt   = np.array([5,10])

param_eval = lambda x: x if type(x) is not list else ' '.join(list(map(str,x))) 
for i in range(len(bin_szs)):
    
    args['bin_sz'] = bin_szs[i]

    outdir   = '{0}/{1}/binsz{bin_sz}/min_nc{min_ncells}_gexp_{min_cell_gexp}'.format(
            datdir, jobn, **args)
    args['wdir'] = outdir

    logdir       = '{0}/log'.format(outdir)
    subprocess.call('mkdir -p {0}'.format(logdir), shell=True)

    for ti in np.arange(tis[i], tfs[i], nmaxt[i]):
        args['t0'] = ti
        args['tf'] = min(ti+nmaxt[i], tfs[i])

        cmd = 'python {0} {1} --quest --skip_if_exists'.format(script, ' '.join(['--{0} {1}'.format(k,param_eval(v)) for k,v in args.items() if v]))
        
        spec         = '{t0}-{tf}'.format(**args)
        pycmdf       = '{0}/pycmd_{1}.txt'.format(logdir,spec)
        sbatch_cmd_f = '{0}/run_cmd_{1}.sbatch'.format(logdir,spec)
        logf         = '{0}/run_log_{1}.txt'.format(logdir,spec)

        f = open(pycmdf, 'w')
        f.write(cmd)
        f.close()

        subprocess.call('cat {0} {1} > {2}'.format(sbatchf, pycmdf, sbatch_cmd_f), shell=True)
        subprocess.call('sbatch -o {0}/sbatch.out -e {0}/sbatch.err -A {1} -p {2} -t {3} -n {4} {5} >{6} 2>{6}'.format(
            logdir, acct, part, time, ncpu, sbatch_cmd_f, logf), shell=True)
