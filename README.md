# How to create a simulation
1. Modify config.ini
2. `./script/run.py -i script/config.ini`-> creates an init file, output folder and `run.sh` in the folder as in `basic: folder` from the ini file.
3. `./[folder]/run.sh`

# Config.ini
- `mode`: `local` or `hpc`
- `folder`: destination folder with all the simulation related data. Created if doesn't exist
- all others should be self-explanatory, and do either on the simulation mode or the simulation itself
