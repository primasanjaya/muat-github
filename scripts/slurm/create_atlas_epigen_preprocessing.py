import csv
import pandas as pd
import os


pcawg_dir = '/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/allclasses/simplified_data/'
sourcedir = '/csc/epitkane/projects/PCAWG20191001/data/modified_data/raw/indelmeisubsvperclass/'
tokenized_data = '/csc/epitkane/projects/PCAWG20191001/data/modified_data/train/allclasses/tokenized_data/'

all_class = os.listdir(sourcedir)

for i in range(0,len(all_class)):

    all_samples = os.listdir(sourcedir + all_class[i])

    pcawg_histology = all_class[i]

    for j in range(0,len(all_samples)):

            string ="#!/bin/bash\n#SBATCH --partition=cpu \n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=1\n#SBATCH --nodelist=a[0-2,5-7]\n#SBATCH --mem=30G\n#SBATCH --time=08:00:00\n#SBATCH --output=/csc/epitkane/projects/litegpt/.out/R-%x.%j.out\n#SBATCH --error=/csc/epitkane/projects/litegpt/.out/R-%x.%j.err\n\n" 
            command = f"srun python3 /csc/epitkane/projects/muat/preprocessing/notebook/epigenetics/create_epigen_data_slurm.py --class-name '{pcawg_histology}' --sample-file '{all_samples[j]}'"

            string_logs = pcawg_histology + all_samples[j]

            completeName = "/csc/epitkane/projects/muat/epigen_" + string_logs +".sh"
            file1 = open(completeName, "w")
            toFile = string + command
            file1.write(toFile)
            file1.close()
