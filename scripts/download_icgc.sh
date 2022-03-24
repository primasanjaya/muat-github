# Download ICGC public release in a semi-automated manner
#
# Usage: ./download_icgc.sh
#
# Be sure to check the original link to contain all summary files
#   https://dcc.icgc.org/releases/release_23/Summary
# and list all files you want from the cohorts in contents().
#
# There may be different contents in the project folders. Check a couple, e.g.:
#   https://dcc.icgc.org/releases/release_23/Projects/CLLE-ES
#   https://dcc.icgc.org/releases/release_23/Projects/BRCA-US
#   https://dcc.icgc.org/releases/release_23/Projects/PBCA-DE
set -x
URL=https://dcc.icgc.org/api/v1/download?fn=
RELEASE=28

summary=(
    donor.all_projects.tsv.gz
    donor_biomarker.all_projects.tsv.gz
    donor_exposure.all_projects.tsv.gz
    donor_family.all_projects.tsv.gz
    donor_surgery.all_projects.tsv.gz
    donor_therapy.all_projects.tsv.gz
    sample.all_projects.tsv.gz
    simple_somatic_mutation.aggregated.vcf.gz
    specimen.all_projects.tsv.gz
)

contents=( # comment out datasets you don't want
	copy_number_somatic_mutation.%.tsv.gz
	donor.%.tsv.gz
	donor_biomarker.%.tsv.gz
	donor_exposure.%.tsv.gz
	donor_family.%.tsv.gz
	donor_therapy.%.tsv.gz
	exp_array.%.tsv.gz
	exp_seq.%.tsv.gz
	meth_array.%.tsv.gz
	meth_seq.%.tsv.gz
	mirna_seq.%.tsv.gz
	protein_expression.%.tsv.gz
	sample.%.tsv.gz
	simple_somatic_mutation.open.%.tsv.gz
	specimen.%.tsv.gz
	structural_somatic_mutation.%.tsv.gz
)

download_file() {
    mkdir -p release_$RELEASE/$(dirname $1)
    [ ! -f release_$RELEASE/$1 ] &&
        wget -q --show-progress $URL/release_$RELEASE/$1 -O release_$RELEASE/$1
}

download_file Projects/README.txt
STUDIES=$(egrep -o "[A-Z]+-[A-Z]+" release_$RELEASE/Projects/README.txt)

for SUM in "${summary[@]}"; do
    download_file Summary/$SUM
done

for STUDY in $STUDIES; do
    for CONTENT in "${contents[@]}"; do
        download_file Projects/$STUDY/$(sed "s/%/$STUDY/" <<< $CONTENT)
    done
done