import csv
import pandas as pd
import os
import pdb

pd_allsamples = pd.read_csv('/users/primasan/projects/muat/extfile/all_tumoursamples_tcga.tsv',sep='\t', index_col=0) 


count=0
for i in range(len(pd_allsamples)):
    pcawg_histology = pd_allsamples.iloc[i]['tumour_type']
    samp = pd_allsamples.iloc[i]['sample']

    check_data = pd.read_csv('/scratch/project_2001668/data/tcga/tokenized/' + pcawg_histology + '/SNV_' + samp,index_col=0)

    if len(check_data.columns) == 149:
        pass
    else:
        count = count + 1
        if count < 250:
            string ="#!/bin/bash\n#SBATCH --account=project_2001668\n#SBATCH --partition=small\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=1\n#SBATCH --mem=64G\n#SBATCH --time=72:00:00\n#SBATCH --output=/scratch/project_2001668/primasan/.out/R-%x.%j.out\n#SBATCH --error=/scratch/project_2001668/primasan/.out/R-%x.%j.err\n\n" 
            command = f"srun python3 /users/primasan/projects/muat/preprocessing/notebook/tcga/tcga_create_epigen_data_slurm.py --class-name '{pcawg_histology}' --sample-file '{samp}' --muat-dir '/users/primasan/projects/muat/' --epigen-file '/scratch/project_2001668/data/pcawg/allclasses/epigenetic_token.tsv' --simplified-dir '/scratch/project_2001668/data/tcga/simplified/' --tokenized-dir '/scratch/project_2001668/data/tcga/tokenized/'"
            string_logs = pcawg_histology + samp

            completeName = "/users/primasan/projects/muat/tcgaepigen_" + string_logs +".sh"
            file1 = open(completeName, "w")
            toFile = string + command
            file1.write(toFile)
            file1.close()
        else:
            print(a)