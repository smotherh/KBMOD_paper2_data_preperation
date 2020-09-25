from os import path
import string
import matcher
import numpy as np

#################################################################################################
#################################           Haydens            ##################################
#################################           scripts            ##################################
#################################################################################################
def link_instcal_files(Pointing_Groups,NEO_src,catalog_src,dest,mode='link',script_name='link_files.sh'):
    """
    This function generates a bash script that creates directories and
    symlinks the correct file names for all pointing_groups.

    DO NOT PUT A TRAILING / IN YOUR FILE PATHS.
    The script adds them in where necessary

    Input
    ---------

    Pointing_Groups:
        An array of pointing group pandas tables loaded in from PickledPointings.pkl

    NEO_src : string
        File path to the folder containing all of the raw instcal files for the
        Lori Allen dataset

    catalog_src : string
        File path to the folder containing all of the reference catalogs needed for
        processing the Lori Allen dataset

    dest : string
        File path destination. All of the folders will be populated into "dest"

    script_name : string
        Optional file name of the script

    Output
    ---------

    none, function will write to a file, named by script_name

    """

    with open(script_name,'w') as link_files:
        link_files.write('# This file links data in a given field\n')
        link_files.write('cd '+dest+'\n')

    for i,Pointing in enumerate(Pointing_Groups):
        if Pointing['stellar_density'][0]>10000.:
            continue

        ra = str(Pointing['ra'][0])
        dec = str(Pointing['dec'][0])
        dir_name = dest+'/Pointing_Group_'+str(i).zfill(3)

        with open(script_name,'a') as link_files:
            link_files.write('\n# Link files for field: '+dir_name+'\n')
            link_files.write('mkdir '+dir_name+'\n')
            link_files.write('mkdir '+dir_name+'/ingest/\n')
            link_files.write('mkdir '+dir_name+'/ingest/instcal\n')
            link_files.write('mkdir '+dir_name+'/ingest/dqmask\n')
            link_files.write('mkdir '+dir_name+'/ingest/wtmap\n')
            link_files.write('mkdir '+dir_name+'/processed_data\n')
            link_files.write('echo lsst.obs.decam.DecamMapper > '+dir_name+'/processed_data/_mapper\n')
            link_files.write('ln -s '+catalog_src+' '+dir_name+'/processed_data/ref_cats\n')

            link_files.write('# Link the images files\n')

        for index,image in Pointing.iterrows():
            if image['survey_night'] is not 3:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+str(image['survey_night']) +
                      '/night_'+str(image['survey_night'])+'/'+image['filename'].decode('UTF-8') +
                      ' '+dir_name+'/ingest/instcal/\n')
            else:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+str(image['survey_night']) +
                      '/night_'+str(image['survey_night'])+'/night_'+str(image['survey_night'])+'/'+image['filename'].decode('UTF-8') +
                      ' '+dir_name+'/ingest/instcal/\n')

        with open(script_name,'a') as link_files:
            link_files.write('# Link the dqmask files\n')

        for index,image in Pointing.iterrows():
            dqmask = image['filename'].decode('UTF-8')[0:20]+'d'+image['filename'].decode('UTF-8')[21:]
            night = str(image['survey_night'])
            if image['survey_night'] is not 3:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+ night +
                      '/night_'+night+'/'+dqmask +
                      ' '+dir_name+'/ingest/dqmask/\n')
            else:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+night +
                      '/night_'+night+'/night_'+night+'/'+ dqmask +
                      ' '+dir_name+'/ingest/dqmask/\n')

        with open(script_name,'a') as link_files:
            link_files.write('# Link the wtmask files\n')

        for index,image in Pointing.iterrows():
            wtmap = image['filename'].decode('UTF-8')[0:20]+'w'+image['filename'].decode('UTF-8')[21:]
            night = str(image['survey_night'])
            if image['survey_night'] is not 3:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+ night +
                      '/night_'+night+'/'+ wtmap +
                      ' '+dir_name+'/ingest/wtmap/\n')
            else:
                with open(script_name,'a') as link_files:
                    link_files.write('ln -s '+NEO_src+'/night_'+night +
                      '/night_'+night+'/night_'+night+'/'+ wtmap +
                      ' '+dir_name+'/ingest/wtmap/\n')

def process_visits(Pointing_Groups,pg_location,pg_lims,pg_idx_type='limits',script_name='process_visits.sh',
                   num_cores=20,source_stack=False,setup_loc=''):

    with open(script_name,'w') as f:
        if source_stack:
            f.write('source '+setup_loc+'\n')
            f.write('setup lsst_distrib\n')
        f.write('cd '+pg_location+'\n')
        if pg_idx_type=='limits':
            pg_ids = np.linspace(pg_lims[0],pg_lims[1],pg_lims[1]-pg_lims[0]+1,dtype=int)
        elif pg_idx_type=='index':
            pg_ids = np.copy(pg_lims)
        else:
            print('Invalid pointing group index type.')
            pg_ids = []
            return()
        for i in pg_ids:
            if Pointing_Groups[i]['stellar_density'][0]>10000:
                continue
            min_visit = str(np.min(Pointing_Groups[i]['visit_id']))
            max_visit = str(np.max(Pointing_Groups[i]['visit_id']))
            f.write('cd Pointing_Group_'+str(i).zfill(3)+'\n')
            f.write('ingestImagesDecam.py processed_data --filetype instcal --mode=link ingest/instcal/* >& ingest.log\n')
            f.write('processCcd.py processed_data/ --rerun rerun_processed_data --id visit='+min_visit+'..'+max_visit+' --longlog -C configProcessCcd_neo.py -j'+str(num_cores)+' >& processCcd.log\n')
            f.write('cd ..\n')





#################################################################################################
#################################         Dino's               ##################################
#################################       Configuration          ##################################
#################################################################################################

# the default commands are written out here just for convenience when defining functions for
# hyak VS epyc jobs later on. They are not required for the code to run.
slurmcmd = string.Template(("imageDifference.py {inpath} --output {outpath} -C {cfgpath} "
                           "--id visit={visitid} ${ccd} --templateId visit={templateid} ${ccd} "
                           "${parallel} ${longlog} ${timeout} > {stdoutpath} 2>{stderrpath}"))

batchcmd = string.Template(("${niced}imageDifference.py {inpath} --output {outpath} -C {cfgpath} "
                           "--id visit={visitid} ${ccd} --templateId visit={templateid} ${ccd} "
                           "${parallel} ${longlog} ${timeout} > {stdoutpath} 2>{stderrpath}"))


class JobConf:
    """A class that holds execution parameters required to create scripts. Not
    all parameters are mandatory so leaving the default empty or preset values
    can be fine, depending on the jobs that are being written.

    Jobs transform the in-going data from the "in" repository into out-going
    data saved in the "out" repository. On an example repository path:

    /epyc/users/smotherh/pointing_groups/Pointing_Group_001/processed_data/rerun/rerun_processed_data/
    |<-------  in repos topdir -------->|<-------------  in repo rerun dir ------------------------->|

    where in_repos_topdir, the in-going top level directory containing all
    individual repositories, points to the top directory containing repositories
    of individual PointingGroups as a subdirectory.
    The in_repo_rerun_dir, the in-going repository rerun directory, points to the
    rerun directory of a particular PointingGroup that contains all the ingested
    exposures. Because this path will change for every PointingGroup the way to
    specify this path is to set the in_repo_rerun_dirstr attribute to a string
    containing the groupid specifier. This specifier will then get written with
    a particular group id when the job is created. F.E. the correct string to
    use for in_repo_rerun_dirstr in this case is:

    Pointing_Group_{groupid:03d}/processed_data/rerun/rerun_processed_data/

    Similar approach is used for defining the place where to store the out-going
    data. Specifying the out_repos_topdir, the out-going repository top level
    directory, and the out_saveloc_dirstr format will set the save location of
    the processing results. For example using '/usr/name/' and '{groupid:03d}'
    will store the outgoing data into

    /usr/name/001/

    In that directory, except for the results, you will also find 2 files into
    which the standard out and standard errors have been dumped. The file names
    formats are, similarly, set by stdoutstr and seterrstr. By default these are

    imdiff_{visitid}.out
    imdiff_{visitid}.err

    If per-group reports are desired the default command has to be edited to
    append to a file, not overwrite it.

    The full formats of the paths, the path identifiers described above are
    joined to form a complete absolute filesystem path to the selected locations,
    can be seen by looking at the inrepopath, outrepopath, stdoutpath and
    stderrpath attributes.

    The save_path attribute sets the local filesystem location where the script
    will be created.

    The remaining attributes controll the values that are substituted in the
    command themselves and are not all required for all commands.

    cfgfile_path - set the path to the imdiff config.py file that holds the
                   image difference configuration
    parallel     - bool, add a -j flag to the command
    longlog      - bool, add a --longlog to the command
    limited      - bool, add a --timeout flag to the command
    perccd       - bool, add a ccd identified to the --id flag
    niced        - prepend 'nice -n' to the command 

    The values for the selected flag in the command are then set by timeout,
    niceness, ncpu (for parallel), and ccd attributes.
    """
    def __init__(self, **kwargs):

        self.in_repos_topdir = ""
        self.in_repo_rerun_dirstr = ""

        self.out_repos_topdir = ""
        self.out_saveloc_dirstr = "{groupid:03d}"

        self.stdoutstr = "imdiff_{visitid}.out"
        self.stderrstr = "imdiff_{visitid}.err"

        self.script_namestr = "job_group_{groupid:03d}"
        self.save_path = "."

        self.cfgfile_path = ""

        self.parallel = False
        self.longlog = True
        self.limited = True
        self.perccd = False
        self.niced = False

        self.timeout = 18000
        self.niceness = "17"
        self.ncpu = 20
        self.ccd = 1

        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def inrepopath(self):
        return path.join(self.in_repos_topdir, self.in_repo_rerun_dirstr)

    @property
    def outrepopath(self):
        return path.join(self.out_repos_topdir, self.out_saveloc_dirstr)

    @property
    def stdoutpath(self):
        return path.join(self.outrepopath, self.stdoutstr)

    @property
    def stderrpath(self):
        return path.join(self.outrepopath, self.stderrstr)

    def __repr__(self):
        retrstr = object.__repr__(self) + "\n"
        for key, val in self.__dict__.items():
            retrstr += "{0:30}{1}\n".format(key, val)
        return retrstr


#################################################################################################
#################################        CMD Builder           ##################################
#################################################################################################

def tmplt_replacer(template, key, what, condition, else_=""):
    """Generic replacement of keys in 'template', a string.Template object.
    Replaces key with 'what' if condition is True, otherwise replaces key
    with else_, an empty string by default.
    """
    if condition:
        tmp = template.safe_substitute({key : what})
    else:
        tmp = template.safe_substitute({key : else_})
    return string.Template(tmp)

def cmd_builder(cmd, conf):
    """Defines a generic template for command and then replaces the
    required keywords. Returns a string with the replacements, not a template.
    """
    cmd = tmplt_replacer(cmd, "niced", "nice -n {niceness} ", conf.niced)
    cmd = tmplt_replacer(cmd, "parallel", "-j {ncpu}", conf.parallel)
    cmd = tmplt_replacer(cmd, "longlog", "--longlog", conf.longlog)
    cmd = tmplt_replacer(cmd, "timeout", "--timeout {timeout}", conf.timeout) 
    cmd = tmplt_replacer(cmd, "ccd", "ccd={ccd}", conf.perccd)

    # recasting back to normal string
    return cmd.safe_substitute({"":""})


def script_writer(cmd, pairs, groupid, conf, sample=False, header="", footer=""):
    """Generic script writer that will loop through all the pairs and write a
    command to process them according to the provided configurations to a file
    with a name in a location also as provided by the configuration.
    Writes both batch and slurm jobs depending on the args and conf sent.
    """
    inrepopath = conf.inrepopath.format(groupid=groupid)
    outrepopath = conf.outrepopath.format(groupid=groupid)

    cmds = []
    for pair in pairs:
        visitid = pair[0]
        tmpltid = pair[1]

        stdoutpath = conf.stdoutpath.format(groupid=groupid, visitid=visitid)
        stderrpath = conf.stderrpath.format(groupid=groupid, visitid=visitid)

        tmp = cmd.format(
            groupid=groupid,
            visitid=visitid,
            templateid=tmpltid,
            inpath=inrepopath,
            outpath=outrepopath,
            cfgpath=conf.cfgfile_path,
            stdoutpath=stdoutpath,
            stderrpath=stderrpath,
            ccd=conf.ccd,
            ncpu=conf.ncpu,
            timeout=conf.timeout,
            niceness=conf.niceness)
        cmds.append(tmp+"\n")

    if sample:
        print(header)
        print(cmds[0])
        print(footer)
    else:
        fname = conf.script_namestr.format(groupid=groupid)
        fpath = path.join(conf.save_path, fname)
        with open(fpath, "w") as f:
            f.write(header)
            f.writelines(cmds)
            f.write(footer)




#################################################################################################
#################################      Epyc util. func.        ##################################
#################################################################################################

def batch_script_from_pairs(pairs, groupid, conf, sample=False, cmd=batchcmd):
    """Given a list of pairs and the id of the group from which the pairs were
    created, will produce a file with imageDifference commands required to produce
    difference images between the given pairs as instructed by the settings.

    If sample is True will output the first command as a string to stdout.
    """
    master_cmd = cmd_builder(cmd, conf)
    script_writer(master_cmd, pairs, groupid, conf, sample=sample)


def batch_script_from_groupid(groups, groupid, conf, sample=False, cmd=batchcmd):
    """Convenience function that takes an group index and produces a file with
    commands that will produce image differences as instructed by the settings.
    """
    pairs = matcher.largest_dt(groups[groupid])
    batch_script_from_pairs(pairs, groupid, conf, sample)


#################################################################################################
#################################      Hyak util. func.        ##################################
#################################################################################################
def slurm_script_from_pairs(pairs, groupid, conf, sample=False, cmd=slurmcmd):
    """Given a list of pairs and the id of the group from which the pairs were
    created, will produce a slurm script with imageDifference commands required
    to produce difference images between the given pairs as instructed by the
    configuration.

    If sample is True will output the first command as a string to stdout.
    """
    header = open("slurm.template").read()
    header = header.format(jobtype="imdiff", groupid=groupid, 
                           workdir="/gscratch/scrubbed/dinob/imdiff")
    master_cmd = cmd_builder(cmd, conf)
    script_writer(master_cmd, pairs, groupid, conf, sample=sample,
                  header=header)


def slurm_script_from_groupid(groups, groupid, conf, sample=False, cmd=slurmcmd):
    """Convenience function that takes an group index and produces a file with
    commands that will produce image differences as instructed by the settings.
    """
    pairs = matcher.largest_dt(groups[groupid])
    slurm_script_from_pairs(pairs, groupid, conf, sample, cmd)


#################################################################################################
#################################      Hyak warp func.         ##################################
#################################################################################################
def slurm_warp_script_from_groupid(groupid, conf=JobConf(), sample=False,
                                   cmd="./warp.py {groupid}"):
    """Given a groupid will produce a slurm script with commands required to
    produce warp.py script as instructed by the configuration.

    If sample is True will output the first command as a string to stdout.
    """
    header = open("slurm.template").read()
    header = header.format(jobtype="warp", groupid=groupid,
                           workdir="/gscratch/scrubbed/dinob/warps")
    # mock the minimum required input to script-writer, most of it will be
    # ignored anyhow
    master_cmd = cmd
    pairs = [(groupid, groupid)]
    script_writer(master_cmd, pairs, groupid, conf, sample=sample,
                  header=header)
