# muat

Mutation-Attention Model

How to use:

Preprocessing:
A) Prepare the environment:
1) create conda env python 2.7, conda create -n <env_name> python=2.7
2) conda install -c anaconda pandas
3) conda install -c anaconda numpy
3) conda install -c bioconda bedops
4) conda install -c bioconda swalign

B) Download dataset:
5) run scripts/download_icgc.sh to download the data
6) create temporary dir (assign all temporary preprocessing directory in this dir)
6a) How to download genome tracks?
6b) How to download genome reference?

7) Install tabix > apt-get install tabix
8) install bedops > apt-get install bedops

8) copy scripts/dmmpre.sh, scripts/preprocessdmm.sh, scripts/annotate_preprocessed.sh to <data_dir>/release_28/
9) edit dmmpre.sh assigning to dependencies (check the dmmpre.sh for information)
10) run preprocessdmm.sh > if successful, this will give an output of 'somatic_mutations.<icgc_project>.dmm.k256.tsv.gz' in every icgc project folder
11) annotate_preprocessed.sh > ex: ./annotate_preprocessed.sh ./BLCA-US/somatic_mutations.BLCA-US.dmm.k256.tsv.gz BLCA-US
    if successful, this file will give an output of somatic_mutations.<icgc_project>.dmm.k256.annotated.tsv.gz in every icgc project folder

C) Prepare environment for MuAt
1) conda create -n <env_name> python=3.7
2) conda install pytorch 1.8 
3) conda install -c conda-forge tqdm
4) conda install -c anaconda scikit-learn
5) conda install -c conda-forge tensorboardx
6) conda install -c anaconda seaborn

D) Create dataset for MuAt
1) run preprocessing/create_PCAWG_dataset.py (adjust the dependencies accordingly)

DEMO
I) Training from the scratch
python3 main.py --dataloader 'pcawg' --block-size 5000 --n-class 24 --n-layer 1 --n-head 1 --n-emb 256 --motif --mut-type 'SNV+MNV' --fold 1 --input-data-dir 'path/to/input/data/dir/' --save-ckpt-dir 'path/to/saveckptdir/' --train

II) Predicting from I
python3 main.py --dataloader 'pcawg' --block-size 5000 --context-length 3 --n-class 24 --n-layer 1 --n-head 1 --n-emb 256 --motif --mut-type 'SNV+MNV' --fold 1 --input-data-dir 'path/to/input/data/dir/' --load-ckpt-dir 'path/to/saveckptdir/' --predict 

III) Predicting from pretrained model
python3 main.py --dataloader 'pcawg' --block-size 5000 --context-length 3 --n-class 24 --n-layer 2 --n-head 1 --n-emb 512 --motif-pos --mut-type 'SNV+MNV+indel' --fold 1 --input-data-dir 'path/to/input/data/dir/' --load-ckpt-dir './bestckpt/fullpcawgfold1_11100_wpos_TripletPosition_bs5000_nl2_nh1_ne512_cl3/' --predict 

IV) Predicting vcf files from pretrained model 
python main.py --dataloader 'pcawg' --block-size 5000 --n-class 24 --n-layer 2 --n-head 1 --n-emb 512 --motif-pos --mut-type 'SNV+MNV+indel' --fold 1 --input-data-dir '/mnt/g/experiment/muat/data/raw/vcf/' --input-filename '00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.vcf' --tmp-dir 'path/to/tempdir/' --reference '/path/to/genome_reference' --load-ckpt-dir 'path/to/ckptdir' --single-pred-vcf


IV) Predicting vcf files from pretrained model
python main.py --dataloader 'pcawg' --input-data-dir '/mnt/g/experiment/muat/data/raw/vcf/' --input-filename '00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.vcf' --tmp-dir '/mnt/g/experiment/muat/data/raw/temp/' --reference '/mnt/g/experiment/muat/hs37d5_1000GP.fa' --load-ckpt-dir '/mnt/g/experiment/muat/bestckpt/wgs/' --load-ckpt-filename 'motif+position_features.pthx' --output-pred-dir '/mnt/g/experiment/muat/data/raw/outputdir/' --output-prefix 'test' --single-pred-vcf --get-features

Notes: --get-features, --output-prefix are optional


V) Predict all vcf files in the directory from pretrained model

python main.py --dataloader 'pcawg' --input-data-dir '/mnt/g/experiment/muat/data/raw/vcf/' --tmp-dir '/mnt/g/experiment/muat/data/raw/temp/' --reference '/mnt/g/experiment/muat/hs37d5_1000GP.fa' --load-ckpt-dir '/mnt/g/experiment/muat/bestckpt/wgs/' --load-ckpt-filename 'motif+position_features.pthx' --output-pred-dir '/mnt/g/experiment/muat/data/raw/outputdir/' --output-prefix 'test' --multi-pred-vcf --get-features

Notes: --get-features, --output-prefix are optional