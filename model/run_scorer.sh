#!/usr/bin/env bash
PROJ_DIR="$(dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd ))"
export PYTHONPATH=${PROJ_DIR}
scorer_tests_path=${PROJ_DIR}'/data/'
gold_file_task1=${PROJ_DIR}/gold/2019/'20151219_3_dem.tsv'
gold_file_task2=${PROJ_DIR}/gold/2019/'20160129_7_gop.tsv'
gold_file_task3=${PROJ_DIR}/gold/2019/'20160311_12_gop.tsv'
gold_file_task4=${PROJ_DIR}/gold/2019/'20180131_state_union.tsv'
gold_file_task5=${PROJ_DIR}/gold/2019/'20181015_60_min.tsv'
gold_file_task6=${PROJ_DIR}/gold/2019/'20190205_trump_state.tsv'
gold_file_task7=${PROJ_DIR}/gold/2019/'20190215_trump_emergency.tsv'

gold18_file_task1=${PROJ_DIR}/gold/2018/'task1-en-file1.txt'
gold18_file_task2=${PROJ_DIR}/gold/2018/'task1-en-file2.txt'
gold18_file_task3=${PROJ_DIR}/gold/2018/'task1-en-file3.txt'
gold18_file_task4=${PROJ_DIR}/gold/2018/'task1-en-file4.txt'
gold18_file_task5=${PROJ_DIR}/gold/2018/'task1-en-file5.txt'
gold18_file_task6=${PROJ_DIR}/gold/2018/'task1-en-file6.txt'
gold18_file_task7=${PROJ_DIR}/gold/2018/'task1-en-file7.txt'

pred18_file_task1=${PROJ_DIR}/model/data/score/'2018clf_score_test01.txt'
pred18_file_task2=${PROJ_DIR}/model/data/score/'2018clf_score_test02.txt'
pred18_file_task3=${PROJ_DIR}/model/data/score/'2018clf_score_test03.txt'
pred18_file_task4=${PROJ_DIR}/model/data/score/'2018clf_score_test04.txt'
pred18_file_task5=${PROJ_DIR}/model/data/score/'2018clf_score_test05.txt'
pred18_file_task6=${PROJ_DIR}/model/data/score/'2018clf_score_test06.txt'
pred18_file_task7=${PROJ_DIR}/model/data/score/'2018clf_score_test07.txt'

pred19_file_task1=${PROJ_DIR}/model/data/score/'2019clf_score_test01.tsv'
pred19_file_task2=${PROJ_DIR}/model/data/score/'2019clf_score_test02.tsv'
pred19_file_task3=${PROJ_DIR}/model/data/score/'2019clf_score_test03.tsv'
pred19_file_task4=${PROJ_DIR}/model/data/score/'2019clf_score_test04.tsv'
pred19_file_task5=${PROJ_DIR}/model/data/score/'2019clf_score_test05.tsv'
pred19_file_task6=${PROJ_DIR}/model/data/score/'2019clf_score_test06.tsv'
pred19_file_task7=${PROJ_DIR}/model/data/score/'2019clf_score_test07.tsv'


pred18_file_task1=${PROJ_DIR}/model/data/score/'Lrank2018_score_test01.txt'
pred18_file_task2=${PROJ_DIR}/model/data/score/'Lrank2018_score_test02.txt'
pred18_file_task3=${PROJ_DIR}/model/data/score/'Lrank2018_score_test03.txt'
pred18_file_task4=${PROJ_DIR}/model/data/score/'Lrank2018_score_test04.txt'
pred18_file_task5=${PROJ_DIR}/model/data/score/'Lrank2018_score_test05.txt'
pred18_file_task6=${PROJ_DIR}/model/data/score/'Lrank2018_score_test06.txt'
pred18_file_task7=${PROJ_DIR}/model/data/score/'Lrank2018_score_test07.txt'

pred19_file_task1=${PROJ_DIR}/model/data/score/'Rrank2019_score_test01.tsv'
pred19_file_task2=${PROJ_DIR}/model/data/score/'Rrank2019_score_test02.tsv'
pred19_file_task3=${PROJ_DIR}/model/data/score/'Rrank2019_score_test03.tsv'
pred19_file_task4=${PROJ_DIR}/model/data/score/'Rrank2019_score_test04.tsv'
pred19_file_task5=${PROJ_DIR}/model/data/score/'Rrank2019_score_test05.tsv'
pred19_file_task6=${PROJ_DIR}/model/data/score/'Rrank2019_score_test06.tsv'
pred19_file_task7=${PROJ_DIR}/model/data/score/'Rrank2019_score_test07.tsv'

#predcb_file_task1=${PROJ_DIR}/data/cb_pred/2019/'cb_pred20151219_3_dem.tsv'
#predcb_file_task2=${PROJ_DIR}/data/cb_pred/2019/'cb_pred20160129_7_gop.tsv'
#predcb_file_task3=${PROJ_DIR}/data/cb_pred/2019/'cb_pred20160311_12_gop.tsv'
#predcb_file_task4=${PROJ_DIR}/data/cb_pred/2019/'cb_pred20180131_state_union.tsv'
#predcb_file_task5=${PROJ_DIR}/data/cb_pred/2019/'cb_pred20181015_60_min.tsv'
#predcb_file_task6=${PROJ_DIR}/data/cb_pred/2019/'cb_pred20190205_trump_state.tsv'
#predcb_file_task7=${PROJ_DIR}/data/cb_pred/2019/'cb_pred20190215_trump_emergency.tsv'




echo 'Scoring a random baseline for CB 2019'
python task1.py --gold_file_path=${gold18_file_task1} --pred_file_path=${pred18_file_task1}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold18_file_task2} --pred_file_path=${pred18_file_task2}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold18_file_task3} --pred_file_path=${pred18_file_task3}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold18_file_task4} --pred_file_path=${pred18_file_task4}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold18_file_task5} --pred_file_path=${pred18_file_task5}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold18_file_task6} --pred_file_path=${pred18_file_task6}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold18_file_task7} --pred_file_path=${pred18_file_task7}
echo '**********'

echo 'Scoring a random baseline for CB 2019'
python task1.py --gold_file_path=${gold_file_task1} --pred_file_path=${pred19_file_task1}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold_file_task2} --pred_file_path=${pred19_file_task2}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold_file_task3} --pred_file_path=${pred19_file_task3}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold_file_task4} --pred_file_path=${pred19_file_task4}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold_file_task5} --pred_file_path=${pred19_file_task5}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold_file_task6} --pred_file_path=${pred19_file_task6}
echo '**********'
echo 'Scoring a random baseline for task 1'
python task1.py --gold_file_path=${gold_file_task7} --pred_file_path=${pred19_file_task7}
echo '**********'



cmd /k
