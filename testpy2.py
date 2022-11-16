import vcf
vcf_reader = vcf.Reader(open('/mnt/e/CRC_testsample_GRCh38_Mutect2_PASS_MSS.vcf.gz', 'rb'))
for record in vcf_reader:
        print(record)