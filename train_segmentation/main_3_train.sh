data_root_dir=/root/data1/spine/public/data
echo "step 6: training the 2D ResUNet model for refinement stage....................................................................."
for fold_ind in 1;do
	python -u train_fine.py --fold_ind=${fold_ind} --data_dir=${data_root_dir}/fine_weak_true --device=cuda:0
done