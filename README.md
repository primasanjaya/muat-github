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

### Predicting .vcf file (GRCh37) with MuAt pretrained model :
* Look at the example file in ./extfile/example_for_alldata_prediction_gz.tsv file. This lists all files which will be predicted by the model
* Run this code
```
python3 main.py --dataloader 'pcawg' --predict-filepath '/path/to/muat/extfile/example_for_alldata_prediction_gz.tsv' --reference-h19 '/path/to/muat/ref/ref' --load-ckpt-file '/path/to/muat/bestckpt/wgs/ensemble/finalpcawgFeaturefold1_11110_wpos_TripletPositionF_bs5000_nl2_nh2_ne256_cl3/new_weight.pthx' --output-pred-dir '/path/to/muat/data/raw/output/' --predict-all
```

* If it succeed, you can see all preprocessed files in /path/to/muat/data/raw/temp/
* and all prediction outputs in --output-pred-dir

### Predicting .vcf file (GRCh38) : *add --convert-hg38-hg19 and --reference-h38 <filepath to ref hg38>
* Download genome reference GRCh38
```
wget http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa -O ./ref/ref_hg38
```
* add --convert-hg38-hg19 and --reference-h38 parser
```
python3 main.py --dataloader 'pcawg' --predict-filepath '/path/to/muat/extfile/example_for_alldata_prediction_gz.tsv' --reference-h19 '/path/to/muat/ref/ref' --load-ckpt-file '/path/to/muat/bestckpt/wgs/ensemble/finalpcawgFeaturefold1_11110_wpos_TripletPositionF_bs5000_nl2_nh2_ne256_cl3/new_weight.pthx' --output-pred-dir '/path/to/muat/data/raw/output/' --predict-all --convert-hg38-hg19 --reference-h38 '/path/to/muat/ref/ref_hg38'
```

### DOWNLOAD PRETRAINED MODELS (CHECKPOINTS)

#### Whole Genome Sequence (trained on PCAWG) : for the best results, input type should be the same as pretrained models type
| Model | Description | Link |
| :---:|     :---:      |          :---: |
| SNV | Model trained on SNV input  | Download  |
| SNV+MNV | Model trained on SNV+MNV input  | Download  |
| SNV+MNV+indels | Model trained on SNV+MNV+indels input  | Download  |
| SNV+MNV+indels+SV/MEI | Model trained on SNV+MNV+indels+SV/MEI input  | Download  |
| SNV+pos | Model trained on SNV+pos input | [Download](https://bitbucket.org/primasanjaya/muat-weight-bitbucket/src/master/snv_pos/)  |
| SNV+indel+pos | Model trained on SNV+indel+pos input | [Download](https://github.com/primasanjaya/snvindelpos)  |
| SNV+MNV+indel+SV/MEI+pos | Model trained on SNV+MNV+indel+SV/MEI+pos input | [Download](https://github.com/primasanjaya/snv-mnv-indel-svmei-pos)| 

#### Whole Exome Sequence (trained on TCGA)
| Model | Description | Link |
| :---:|     :---:      | :---: |
| SNV+MNV | Model trained on SNV+MNV input  |  [Download](https://github.com/primasanjaya/wes_snv_mnv)| 

## Training MuAt PCAWG
Read README_training.md