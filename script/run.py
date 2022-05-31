#!/usr/bin/python3
import os
import sys
import stat
import argparse
import configparser
import jinja2
from pathlib import Path


# render a templated init file from a supplied config file 
parser = argparse.ArgumentParser(description='Run vpfc simulation scipt')
parser.add_argument("-i",
    help="Input config file", metavar="input", required=True)

args = vars(parser.parse_args())

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.optionxform=str
config.read(args["i"])

templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)
template = templateEnv.get_template("script/template.txt")

init_txt = template.render(problem="vpfc", basic=config["basic"], parameters=config["parameters"], domain=config["domain"],
                            solver=config["solver"], output=config["output"]["names"].split(","), adapt=config["adapt"])

# create the destination folder for the simulation
folder_path = config["basic"]["folder"]
Path(folder_path).mkdir(parents=True, exist_ok=True)

# write the init file
with open(os.path.join(folder_path,"init.2d"), "w") as f:
    f.write(init_txt)
    f.close()

# write the run.sh either in local or hpc mode
with open(os.path.join(folder_path,"run.sh"), "w") as f:
    if config["basic"]["mode"] == "hpc":
        f.write("#!/bin/bash \n")
        f.write("#SBATCH --ntasks=" + config["basic"]["np"] + " \n")
        f.write("#SBATCH --mail-user=" + config["basic"]["email"] + " \n")
        f.write("#SBATCH --time 96:00:00 \n")
        f.write("#SBATCH --mem 4096M \n")
        f.write("#SBATCH --output="+config["basic"]["hpc_base_dir"]+'/'+config["basic"]["folder"]+"/log.out \n")
        f.write("#SBATCH --error="+config["basic"]["hpc_base_dir"]+'/'+config["basic"]["folder"]+"/log.err \n")
        f.write("#SBATCH --partition haswell64 \n")
        f.write("#SBATCH --account wir \n")
        f.write("ulimit -s unlimited \n")
        f.write("source /scratch/ws/0/seya960b-thesis/.bashrc_local \n")
        f.write("module load SuiteSparse/5.7.1-foss-2020a-METIS-5.1.0 \n")
        f.write("module load CMake/3.16.4-GCCcore-9.3.0 \n")
        f.write("./build-cmake/src/vpfc2 " + os.path.join(folder_path,"init.2d") + "\n")
        # f.write("srun build-cmake/src/vpfc2 " + os.path.join(folder_path,"init.2d") + "\n")

    else:
        f.write("#!/bin/bash \n")
        f.write("\n")
        f.write("mpirun " + "-np 1 " + "--bind-to core " + " build-cmake/src/vpfc2 " + os.path.join(folder_path,"init.2d"))
        f.write("\n")
        f.write("exit 0")
        f.close()
    f.close()

    os.chmod(os.path.join(folder_path,"run.sh"), os.stat(os.path.join(folder_path,"run.sh")).st_mode | stat.S_IEXEC)

# create an ouput folder for Paraview files and particle position logs
Path(os.path.join(folder_path,"output")).mkdir(parents=True, exist_ok=True)
