import glob
import json
import os
import shutil
import subprocess
import torch



def load_experiment_configs(log_dir, uuid=None):
    ''' 
        Loads all experiments in a given directory 
            Optionally, may be restricted to those with a given uuid
    '''
    dirs = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    results = []
    for d in dirs:
        cfg_path = os.path.join(log_dir, d, 'config.json')
        if not os.path.exists(cfg_path):
            continue
        with open(os.path.join(log_dir, d, 'config.json'), 'r') as f:
            results.append(json.load(f))
            if uuid is not None and results[-1]['uuid'] != uuid:
                results.pop()
    return results

def load_experiment_config_paths(log_dir, uuid=None):
    dirs = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    results = []
    for d in dirs:
        cfg_path = os.path.join(log_dir, d, 'config.json')
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)
            results.append(cfg_path)
            if uuid is not None and cfg['uuid'] != uuid:
                results.pop()
    return results





def checkpoint_name(checkpoint_dir, epoch='latest'):
    return os.path.join(checkpoint_dir, 'ckpt-{}.dat'.format(epoch))

def last_archived_run(base_dir, uuid):
    ''' Returns the name of the last archived run. Of the form:
        'UUID_run_K'
    '''
    archive_dir = os.path.join(base_dir, 'archive')
    existing_runs = glob.glob(os.path.join(archive_dir, uuid + "_run_*"))
    print(os.path.join(archive_dir, uuid + "_run_*"))
    if len(existing_runs) == 0:
        return None
    run_numbers = [int(run.split("_")[-1]) for run in existing_runs]
    current_run_number = max(run_numbers) if len(existing_runs) > 0 else 0
    current_run_archive_dir = os.path.join(archive_dir, "{}_run_{}".format(uuid, current_run_number))
    return current_run_archive_dir

def archive_current_run(base_dir, uuid):
    ''' Archives the current run. That is, it moves everything
        base_dir/*uuid* -> base_dir/archive/uuid_run_K/
        where K is determined automatically.
    '''
    matching_files = glob.glob(os.path.join(base_dir, "*" + uuid + "*"))
    if len(matching_files) == 0:
        return
    
    archive_dir = os.path.join(base_dir, 'archive')
    os.makedirs(archive_dir, exist_ok=True)
    existing_runs = glob.glob(os.path.join(archive_dir, uuid + "_run_*"))
    run_numbers = [int(run.split("_")[-1]) for run in existing_runs]
    current_run_number = max(run_numbers) + 1 if len(existing_runs) > 0 else 0
    current_run_archive_dir = os.path.join(archive_dir, "{}_run_{}".format(uuid, current_run_number))
    os.makedirs(current_run_archive_dir)
    for f in matching_files:
        shutil.move(f, current_run_archive_dir)
    return


def save_checkpoint(obj, directory, step_num):
    os.makedirs(directory, exist_ok=True)
    torch.save(obj, checkpoint_name(directory))
    subprocess.call('cp {} {} &'.format(
                        checkpoint_name(directory), 
                        checkpoint_name(directory, step_num)),
                    shell=True)