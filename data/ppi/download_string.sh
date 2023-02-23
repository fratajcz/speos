wget --no-check-certificate https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz
gunzip 9606.protein.links.v11.5.txt.gz
sed -e 's/9606\.//g' 9606.protein.links.v11.5.txt > string.txt
rm 9606.protein.links.v11.5.txt.gz
rm 9606.protein.links.v11.5.txt