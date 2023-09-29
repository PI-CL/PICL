echo "example nswe"

python main.py 
-lf nswe_l1_1_l4_100_l5_100_n_400_epochs_100_10_10_10_0.25 \
--batch_size 32 \
-ch 32 \
-fc 0 \
-l1 1 \
-l2 0 \
-l3 0 \
-l4 100 \
-l5 100 \
--traj_end_tr 400 \
--epochs 100 \
--re_epochs 0 \
--tune_epochs 0 \
--loop 10 \
--unlabel_size 0.25 \
--recall_size 4 \
--gap_sample 5 \
--obs_num 7 \
--state_num 32

echo "done"