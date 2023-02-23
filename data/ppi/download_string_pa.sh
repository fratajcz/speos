wget --no-check-certificate https://stringdb-static.org/download/protein.physical.links.v11.5/9606.protein.physical.links.v11.5.txt.gz
gunzip 9606.protein.physical.links.v11.5.txt.gz
sed -e 's/9606\.//g' 9606.protein.physical.links.v11.5.txt > string_pa.txt
rm 9606.protein.physical.links.v11.5.txt.gz
rm 9606.protein.physical.links.v11.5.txt

