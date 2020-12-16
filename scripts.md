# simple run
python -m spinup.run trpo --env Ant-v2  
# plot previous run
python -m spinup.run plot data/trpo/ant
# test previous policy
python -m spinup.run test_policy data/trpo/ant



python -m spinup.run vpg --env Ant-v2 --exp_name ant_vpg --act tf.nn.relu --hid [32,]
# exp name 
python -m spinup.run ppo --env Walker2d-v2 --exp_name ppo_walker --data_dir data/walker --epochs 200 --steps 10000 