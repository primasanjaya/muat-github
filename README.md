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
7) download genome reference --> http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz

8) Install tabix > apt-get install tabix
9) copy scripts/dmmpre.sh, scripts/preprocessdmm.sh, scripts/annotate_preprocessed.sh to <data_dir>/release_28/
10) edit dmmpre.sh assigning to dependencies (check the dmmpre.sh for information)
11) run preprocessdmm.sh > if successful, this will give an output of 'somatic_mutations.<icgc_project>.dmm.k256.tsv.gz' in every icgc project folder
12) annotate_preprocessed.sh > ex: ./annotate_preprocessed.sh ./BLCA-US/somatic_mutations.BLCA-US.dmm.k256.tsv.gz BLCA-US
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
```
python3 main.py --dataloader 'pcawg' --block-size 5000 --n-class 24 --n-layer 1 --n-head 1 --n-emb 256 --motif --mut-type 'SNV+MNV' --fold 1 --input-data-dir '/path/to/input/data/dir/' --save-ckpt-dir '/path/to/saveckptdir/' --train
```
II) Predicting from I
```
python3 main.py --dataloader 'pcawg' --block-size 5000 --context-length 3 --n-class 24 --n-layer 1 --n-head 1 --n-emb 256 --motif --mut-type 'SNV+MNV' --fold 1 --input-data-dir '/path/to/input/data/dir/' --load-ckpt-dir '/path/to/saveckptdir/' --predict 
```
III) Predicting from pretrained model
```
python3 main.py --dataloader 'pcawg' --block-size 5000 --context-length 3 --n-class 24 --n-layer 2 --n-head 1 --n-emb 512 --motif-pos --mut-type 'SNV+MNV+indel' --fold 1 --input-data-dir '/path/to/input/data/dir/' --load-ckpt-dir './bestckpt/fullpcawgfold1_11100_wpos_TripletPosition_bs5000_nl2_nh1_ne512_cl3/' --predict 
```
IV) Predicting vcf files from pretrained model
```
python3 main.py --dataloader 'pcawg' --input-data-dir '/path/to/muat/data/raw/vcf/' --input-filename '00b9d0e6-69dc-4345-bffd-ce32880c8eef.consensus.20160830.somatic.snv_mnv.vcf' --tmp-dir '/path/to/muat/data/raw/temp/' --reference '/path/to/genome_reference/files' --load-ckpt-dir '/path/to/muat/bestckpt/wgs/' --load-ckpt-filename 'motif+position_features.pthx' --output-pred-dir '/path/to/muat/data/raw/outputdir/' --output-prefix 'test' --single-pred-vcf --get-features
```
Notes: --get-features, --output-prefix are optional

V) Predict all vcf files in the directory from pretrained model
```
python3 main.py --dataloader 'pcawg' --input-data-dir '/path/to/muat/data/raw/vcf/' --tmp-dir '/path/to/tempdir/' --reference '/path/to/genome_reference/files' --load-ckpt-dir '/path/to/muat/bestckpt/wgs/' --load-ckpt-filename 'motif+position_features.pthx' --output-pred-dir '/path/to/muat/data/raw/outputdir/' --output-prefix 'test' --multi-pred-vcf --get-features
```
Notes: --get-features, --output-prefix are optional







