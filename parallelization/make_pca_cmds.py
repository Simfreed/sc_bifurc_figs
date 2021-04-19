import subprocess
import numpy as np

#### WRITE JOBS TO FILE ####
homedir = '/home/slf3348'
outdir  = '/projects/p31095/simonf/out/'
datdir 	= '{0}/weinreb_2020/data'.format(outdir)
cmdf    = '{0}/jobs/cmds/cmds.txt'.format(homedir)
acct    = 'p31095' #'b1020' #'p31095'
part    = 'short' #'b1020' #'normal'
time    = '3:59:59'
ntasks  = 10 
sbatchf = '{0}/jobs/run_cmd.sbatch'.format(homedir)
script  = '{0}/bifurc/haem/python/pca_gene_resample.py'.format(homedir)

jobn      = 'group_nulls'

tii      = 65
tff      = 121
grp_sz   = 10 
nsamp    = 100
odir     = 'cov_eigs_all'

for ti in np.arange(tii, tff, grp_sz):
    tf  = min(ti+grp_sz, tff)
    cmd = 'python {0} --quest --nsamp {1} --dir {2} --ti {3} --tf {4}'.format(script, nsamp, odir, ti, tf) 
    
    metadir      = '{0}/{1}/meta'.format(datdir, odir)
    pycmdf       = '{0}/pycmd_t{1}-{2}.txt'.format(metadir, ti, tf)
    sbatch_cmd_f = '{0}/run_cmd_t{1}-{2}.sbatch'.format(metadir, ti, tf)
    logf         = '{0}/run_log_t{1}-{2}.txt'.format(metadir, ti, tf)

    sbatch_o     = '{0}/sbatch_t{1}-{2}.out'.format(metadir, ti, tf)
    sbatch_e     = '{0}/sbatch_t{1}{2}.err'.format(metadir, ti, tf)

    subprocess.call('mkdir -p {0}'.format(metadir), shell=True)
    f = open(pycmdf, 'w')
    f.write(cmd)
    f.close()

    subprocess.call('cat {0} {1} > {2}'.format(sbatchf, pycmdf, sbatch_cmd_f), shell=True)
    
    subprocess.call('sbatch -o {0} -e {1} -A {2} -p {3} -t {4} -n {5} {6} >{7} 2>{7}'.format(
        sbatch_o, sbatch_e, acct, part, time, ntasks, sbatch_cmd_f, logf), shell=True)
