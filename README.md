# MuAt

Mutation-Attention Model

## Quick Start (Tested on Linux)

  * Clone muat repository
  * Go to muat repository
  * Create conda environment
```
conda env create -f muat-conda.yml
```
  * activate muat environment
```
conda activate muat
```  
  * Download genome reference to ./ref:
```
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/hs37d5.fa.gz -O ./ref/ref.gz
gunzip ./ref/ref.gz

``` 
* Make sure that preprocessing/dmm/annotate_mutations_with_bed.sh is permitted to be executed.
```
chmod 755 preprocessing/dmm/annotate_mutations_with_bed.sh
```

* Predicting .vcf file with MuAt pretrained model :
```
python3 main.py --dataloader 'pcawg' --input-file '/path/to/muat/data/raw/vcf/file.vcf' --reference '/path/to/muat/ref/ref' --load-ckpt-file '/path/to/muat/bestckpt/wgs/motif+position_features.pthx' --output-pred-dir '/path/to/muat/data/raw/output/' --single-pred-vcf --get-features
```

* Predicting .vcf file with MuAt ensamble model :
```
python3 main.py --dataloader 'pcawg' --input-file '/path/to/muat/data/raw/vcf/file.vcf' --reference '/path/to/muat/ref/ref' --load-ckpt-dir '/path/to/muat/bestckpt/wgs/ensamble/' --output-pred-dir '/path/to/muat/data/raw/output/' --ensamble --get-features
```


## Training PCAWG
### Download dataset:
* run scripts/download_icgc.sh
	
> Downloading data approximately may take five hours to complete.

### Preprocessing

  * Create conda environment
```
conda env create -f muat-pre-conda.yml
```
  * Activate muat-pre env
```
conda activate muat-pre
```
  * Copy scripts/dmmpre.sh, scripts/preprocessdmm.sh, scripts/annotate_preprocessed.sh to <data_dir>/release_28/
  * Edit dmmpre.sh assigning to dependencies (check the dmmpre.sh for information)
  * run preprocessdmm.sh > if successful, this will give an output of 'somatic_mutations.<icgc_project>.dmm.k256.tsv.gz' in every icgc project folder
  * annotate_preprocessed.sh > ex: ./annotate_preprocessed.sh ./BLCA-US/somatic_mutations.BLCA-US.dmm.k256.tsv.gz BLCA-US. If successful, this file will give an output of somatic_mutations.<icgc_project>.dmm.k256.annotated.tsv.gz in every icgc project folder

### Create Dataset for MuAt
* run preprocessing/create_PCAWG_dataset.py (adjust the dependencies accordingly)

### Training from scratch
```
python3 main.py --dataloader 'pcawg' --block-size 5000 --n-class 24 --n-layer 1 --n-head 1 --n-emb 256 --motif --mut-type 'SNV+MNV' --fold 1 --input-data-dir '/path/to/input/data/dir/' --save-ckpt-dir '/path/to/saveckptdir/' --train
```

### Predicting from scratch
```
python3 main.py --dataloader 'pcawg' --block-size 5000 --context-length 3 --n-class 24 --n-layer 1 --n-head 1 --n-emb 256 --motif --mut-type 'SNV+MNV' --fold 1 --input-data-dir '/path/to/input/data/dir/' --load-ckpt-dir '/path/to/saveckptdir/' --predict 
```

























