# MuAt

Mutation-Attention Model

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