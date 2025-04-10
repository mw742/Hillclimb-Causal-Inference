#!/bin/bash
#SBATCH --job-name=cal_PRS            # Job name
#SBATCH --output=cal_PRS_%j.log       # Log file
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --mem=128G
#SBATCH --partition=shared

echo "Starting dataset harmonization and PRS calculation..."

# -----------------------------
# Step 0: Load environment
# -----------------------------
module load python
module load plink
module load R

# Quick checks for needed tools
if ! command -v plink &> /dev/null; then
    echo "Error: PLINK not found! Ensure it is installed or loaded."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found! Ensure it is installed or loaded."
    exit 1
fi

# -----------------------------
# Step 1: Extract SNP lists
# -----------------------------
echo "Extracting SNP lists from ABCD and GWAS..."

# ABCD genotype .bim -> ABCD_snps.txt
awk '{print $2}' QCed_data_third.bim > ABCD_snps.txt
if [ $? -ne 0 ]; then
  echo "Error extracting SNPs from ABCD dataset."
  exit 1
fi
echo "Extracted SNPs from ABCD dataset to ABCD_snps.txt."

# GWAS summary file -> GWAS_snps.txt
# Assuming the first column in 'gwas_data_prepared_for_PRSice.csv' is 'SNP'
cut -d ',' -f 1 gwas_data_prepared_for_PRSice.csv | tail -n +2 > GWAS_snps.txt
if [ $? -ne 0 ]; then
  echo "Error extracting SNPs from GWAS dataset."
  exit 1
fi
echo "Extracted SNPs from GWAS dataset to GWAS_snps.txt."

# -----------------------------
# Step 2: Find common SNPs
# -----------------------------
echo "Finding common SNPs between datasets..."
comm -12 <(sort ABCD_snps.txt) <(sort GWAS_snps.txt) > common_snps.txt
if [ $? -ne 0 ]; then
  echo "Error finding common SNPs."
  exit 1
fi
echo "Common SNPs saved to common_snps.txt."

# -----------------------------
# Step 3: Filter GWAS summary
# -----------------------------
echo "Filtering GWAS summary to common SNPs..."
python3 <<EOF
import pandas as pd

try:
    # Load full GWAS data
    gwas = pd.read_csv("gwas_data_prepared_for_PRSice.csv", sep=',')
    
    # Load common SNPs
    common_snps = pd.read_csv("common_snps.txt", header=None, names=["SNP"])

    # Filter by common SNPs
    filtered_gwas = gwas[gwas["SNP"].isin(common_snps["SNP"])]

    # Save as tab-delimited
    filtered_gwas.to_csv("filtered_gwas.tsv", index=False, sep='\t')
    print("Filtered GWAS dataset saved as filtered_gwas.tsv.")
except Exception as e:
    print(f"Error filtering GWAS dataset: {e}")
    exit(1)
EOF
if [ $? -ne 0 ]; then
  echo "Python filtering of GWAS data failed."
  exit 1
fi

# -----------------------------
# Step 4: Filter ABCD genotype
# -----------------------------
echo "Filtering ABCD genotype to common SNPs..."
plink --bfile QCed_data_third --extract common_snps.txt --make-bed --out ABCD_filtered
if [ $? -ne 0 ]; then
  echo "Error filtering ABCD dataset with PLINK."
  exit 1
fi
echo "Filtered ABCD dataset saved as ABCD_filtered.*"

# -----------------------------
# Step 5: Remove ambiguous SNPs
# (A/T or C/G, which can cause
# strand-flip issues)
# -----------------------------
echo "Removing ambiguous SNPs (A/T or C/G)..."
awk '($5=="A" && $6=="T") || \
     ($5=="T" && $6=="A") || \
     ($5=="G" && $6=="C") || \
     ($5=="C" && $6=="G") {print $2}' ABCD_filtered.bim > ambiguous_snps.txt

plink --bfile ABCD_filtered --exclude ambiguous_snps.txt --make-bed --out ABCD_noambig
if [ $? -ne 0 ]; then
  echo "Error removing ambiguous SNPs from ABCD dataset."
  exit 1
fi
echo "ABCD dataset without ambiguous SNPs saved as ABCD_noambig.*"

# -----------------------------
# Step 6: Subset individuals 
# to those present in 
# phenotype + covariates + genotype
# -----------------------------
echo "Ensuring all files share the same individuals..."

# 6a. Extract FID/IID from phenotype (assuming CSV with FID,IID,...)
#    phenotype_file.csv columns are something like:
#    FID, IID, phenotype
#    then the below line can be used. Adjust if needed.
awk '{print $1, $2}' phenotype_file_cauc.txt > pheno_fid_iid_cauc.txt

# 6b. Extract FID/IID from covariates (they are space-delimited with FID and IID in the first two columns)
awk '{print $1, $2}' ABCD_cauc_only_pca.cov > covar_fid_iid_cauc.txt

# 6c. Extract FID/IID from genotype fam
awk '{print $1, $2}' ABCD_noambig.fam > geno_fid_iid.txt

# 6d. Find the intersection
comm -12 <(sort pheno_fid_iid_cauc.txt) <(sort covar_fid_iid_cauc.txt) | comm -12 - <(sort geno_fid_iid.txt) > common_fid_iid_cauc.txt

echo "Filtering phenotype_file.csv and covariates_with_fid.txt to keep only intersection..."

# 6e. Filter phenotype (CSV with FID,IID in columns 1,2):
awk 'NR == FNR { ids[$1" "$2] = 1; next }
     { if (($1" "$2) in ids) print $1, $2, $3 }' \
     common_fid_iid_cauc.txt phenotype_file_cauc.txt \
  > filtered_phenotype_file_cauc.txt



# 6f. Filter covariates (space-delimited with FID, IID in columns 1,2):
awk 'NR==FNR {a[$1,$2]; next} ($1,$2) in a' common_fid_iid_cauc.txt ABCD_cauc_only_pca.cov > filtered_covariates_with_fid_cauc.txt

# 6g. Filter genotype to keep only these individuals
plink --bfile ABCD_noambig --keep common_fid_iid_cauc.txt \
      --make-bed --out ABCD_cleaned_cauc
if [ $? -ne 0 ]; then
  echo "Error subsetting ABCD_noambig to common individuals."
  exit 1
fi

plink --bfile ABCD_cleaned_cauc --freq --out test_ABCD_cleaned_cauc
if [ $? -ne 0 ]; then
  echo "Error calculating allele frequencies on ABCD_cleaned_cauc."
  exit 1
fi

# -----------------------------
# Step 7: Run PRSice
# -----------------------------
echo "Calculating PRS with PRSice..."

# Path to PRSice 
PRSice_path="/gpfs/home/mwei/PRSice"

# Important: Check that 'filtered_gwas.tsv' columns match:
#   Col0 = SNP
#   Col1 = CHR
#   Col2 = BP
#   Col3 = A1
#   Col4 = A2
#   Col5 = BETA (or OR)
#   Col6 = P
# Adjust the --snp/--chr/--bp/--a1/--a2/--stat/--pvalue indices if needed
"${PRSice_path}/PRSice_linux" \
    --base filtered_gwas.tsv \
    --target ABCD_cleaned_cauc \
    --snp 0 \
    --chr 1 \
    --bp 2 \
    --a1 3 \
    --a2 4 \
    --stat 5 \
    --pvalue 6 \
    --beta \
    --index \
    --cov filtered_covariates_with_fid_cauc.txt \
    --pheno filtered_phenotype_file_cauc.txt \
    --binary-target F \
    --thread 16 \
    --perm 10721 \
    --print-snp \
    --out PRS_result_permall_cauc

echo "PRS calculation complete! Results are in PRS_results_permall_cauc.*"

