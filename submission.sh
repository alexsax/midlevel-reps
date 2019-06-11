#!/usr/bin/env bash

cd /root/perception_module
python -m scripts.evaluate_habitat run_cfg with cfg_overwrite \
  cfg.learner.backout.use_backout=True \
  cfg.learner.backout.patience=80 \
  cfg.learner.backout.unstuck_dist=0.3 \
  cfg.learner.validator.use_validator=True \
  cfg.learner.taskonomy_encoder='/mnt/models/curvature_encoder.dat' \
  cfg.eval_kwargs.exp_path='/mnt/eval_runs/curvature_encoding_moresteps_collate3'
#  cfg.eval_kwargs.exp_path='/mnt/eval_runs/depth_encoding_moresteps_restart3'
#  cfg.eval_kwargs.exp_path='/mnt/eval_runs/curvature_encoding_moresteps_collate_unpool'
#  cfg.eval_kwargs.exp_path='/mnt/eval_runs/depth_encoding_horror'
#  cfg.eval_kwargs.exp_path='/mnt/eval_runs/keypoints3d_encoding_moresteps_collate_visitpenalty'
#  cfg.eval_kwargs.exp_path='/mnt/eval_runs/keypoints3d_encoding_moresteps_collate'

