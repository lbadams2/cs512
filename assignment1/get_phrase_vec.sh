dataset='YELP'

#./auto_phrase.sh
#./phrasal_segmentation.sh
mv models/${dataset}/segmentation.txt models/${dataset}/${dataset}_segmentation.txt

# replace spaces with underscores between tags
perl -pe 's{>[^<>]*</}{ $& =~ tr/ /_/r }eg' models/${dataset}/${dataset}_segmentation.txt > models/${dataset}/${dataset}_segmentation_.txt

# replace tags with _
sed 's/<phrase>/_/g' models/${dataset}/${dataset}_segmentation_.txt > models/${dataset}/${dataset}_segmentation_tmp.txt
rm models/${dataset}/${dataset}_segmentation_.txt
sed 's#</phrase>#_#g' models/${dataset}/${dataset}_segmentation_tmp.txt > models/${dataset}/${dataset}_segmentation_.txt
rm models/${dataset}/${dataset}_segmentation_tmp.txt

# add space between _ and punctuation
sed -E 's/(_)([\.,?"\(\):;!])/\1 \2/g' models/${dataset}/${dataset}_segmentation_.txt > models/${dataset}/${dataset}_segmentation_tmp.txt
rm models/${dataset}/${dataset}_segmentation_.txt
# add space between punctuation and _
sed -E 's/([\.,?"\(\):;!])(_)/\1 \2/g' models/${dataset}/${dataset}_segmentation_tmp.txt > models/${dataset}/${dataset}_segmentation_.txt
rm models/${dataset}/${dataset}_segmentation_tmp.txt

# send file to ec2
scp -i /Users/liamadams/.ssh/ec2.pem models/${dataset}/${dataset}_segmentation_.txt ec2-user@3.83.65.90:/home/ec2-user/assignment1/word2vec/data
ssh -i /Users/liamadams/.ssh/ec2.pem ec2-user@3.83.65.90 MODEL=$dataset 'bash -s' << 'ENDSSH'
cd assignment1/word2vec
./bin/word2vec -train data/${MODEL}_segmentation_.txt -output ${MODEL}_phrase.emb -cbow 0 -size 100 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 0
ENDSSH

# download word2vec output from ec2
scp -i /Users/liamadams/.ssh/ec2.pem ec2-user@3.83.65.90:/home/ec2-user/assignment1/word2vec/${dataset}_phrase.emb .

# extract phrase vectors
# sed -E "s/_'/_/g"
# sed -E "s/'_/_/g"
#awk '/_.*?_/{ print $0 }' {$model}_phrase.emb > {$model}_auto_phrase.emb
#awk '/^_.*_\s+/{ print $0 }' {$model}_phrase.emb > {$model}_auto_phrase.emb
awk '/^_.*?_/{ print $0 }' {$model}_phrase.emb > {$model}_auto_phrase.emb