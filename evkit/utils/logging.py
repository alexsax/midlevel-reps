import os
import shutil
import pickle

def replay_logs(existing_log_paths, mlog):
    existing_results_path = combined_paths(existing_log_paths, 'result_log.pkl')
    save_training_logs(existing_results_path, mlog)

def move_metadata_file(old_log_dir, new_log_dir, uuid):
    fp_metadata_old = get_subdir(old_log_dir, 'metadata')
    fp_metadata_old = [fp for fp in fp_metadata_old if uuid in fp]

    if len(fp_metadata_old) == 0:
        logger.info(f'No metadata for new experiment found at {old_log_dir} for {uuid}')
    else:
        fp_metadata_new = new_log_dir
        logger.info(f'Moving logs from {fp_metadata_old[0]} to {fp_metadata_new}')
        shutil.move(fp_metadata_old, fp_metadata_new)
                
def checkpoint_name(checkpoint_dir, epoch='latest'):
    return os.path.join(checkpoint_dir, 'ckpt-{}.dat'.format(epoch))

def get_parent_dirname(path):
    return os.path.basename(os.path.dirname(path))

def get_subdir(training_directory, subdir_name):
    """
    look through all files/directories in training_directory
    return all files/subdirectories whose basename have subdir_name
    if 0, return none
    if 1, return it
    if more, return list of them

    e.g. training_directory: '/path/to/exp'
         subdir_name: 'checkpoints' (directory)
         subdir_name: 'rewards' (files)
    """
    training_directory = training_directory.strip()
    subdirectories = os.listdir(training_directory)
    special_subdirs = []

    for subdir in subdirectories:
        if subdir_name in subdir:
            special_subdir = os.path.join(training_directory, subdir)
            special_subdirs.append(special_subdir)

    if len(special_subdirs) == 0:
        return None
    elif len(special_subdirs) == 1:
        return special_subdirs[0]
    return special_subdirs

def read_pkl(pkl_name):
    with open(pkl_name, 'rb') as f:
        data = pickle.load(f)
    return data


def unused_dir_name(output_dir):
    """
    Returns a unique (not taken) output_directory name with similar structure to existing one
    Specifically,
    if dir is not taken, return itself
    if dir is taken, return a new name where
        if dir = base + number, then newdir = base + {number+1}
        ow: newdir = base1
    e.g. if output_dir = '/eval/'
         if empty: return '/eval/'
         if '/eval/' exists: return '/eval1/'
         if '/eval/' and '/eval1/' exists, return '/eval2/'

    """
    existing_output_paths = []
    if os.path.exists(output_dir):
        if os.path.basename(output_dir) == '':
            output_dir = os.path.dirname(output_dir)  # get rid of end slash
        dirname = os.path.dirname(output_dir)
        base_name_prefix = re.sub('\d+$', '', os.path.basename(output_dir))

        existing_output_paths = get_subdir(dirname, base_name_prefix)
        assert existing_output_paths is not None, f'Bug, cannot find output_dir {output_dir}'
        if not isinstance(existing_output_paths, list):
            existing_output_paths = [existing_output_paths]
        numbers = [get_number(os.path.basename(path)[-5:]) for path in existing_output_paths]
        eval_num = max(max(numbers), 0) + 1

        output_dir = os.path.join(dirname, f'{base_name_prefix}{eval_num}', '')
        print('New output dir', output_dir)

    return output_dir, existing_output_paths







def combined_paths(paths, name):
    """
    Runs get_subdir on every path in paths then flattens
    Finds all files/directories in all paths whose basename includes name
    Returns all these in a one-dimensional list
    """
    ret_paths = []
    for exp_path in paths:
        evals = get_subdir(exp_path, name)
        if evals is None:
            continue
        if isinstance(evals, list):
            ret_paths.extend(evals)
        else:
            ret_paths.append(evals)
    return ret_paths

def read_logs(pkl_name):
    return read_pkl(pkl_name)['results'][0]

def save_training_logs(results_paths, mlog):
    """
    results_path is a list of experiment's result pkl file paths
    e.g. results_path = ['exp1/results_log.pkl', 'exp2/results_log.pkl']
    """
    step_num_set = set()
    for results_path in results_paths:
        print(f'logging {results_path}')
        try:
            results = read_logs(results_path)
        except Exception as e:
            print(f'Could not read {results_path}. could be empty', e)
            continue
        for result in results:
            i = result['step_num']
            if i in step_num_set:
                continue  # make sure not to add the same element doubly
            else:
                step_num_set.add(i)
            del result['step_num']
            for k,v in result.items():
                log(mlog, k, v, phase='train')
            reset_log(mlog, None, i, phase='train')

            
def save_testing_logs(eval_paths, mlog):
    """
    eval_paths is a list of eval runs path
    e.g. eval_paths = ['exp1/eval', 'exp1/eval1', 'exp2/eval']
    """
    data_all_epochs = []  # contains (epoch_num, (rewards_lst, lengths_lst))
    seen_epochs = set()
    for eval_path in eval_paths:
        subdirectories = os.listdir(eval_path)
        for subdir in subdirectories:
            if 'rewards' in subdir:
                print(f'logging {eval_path}/{subdir}')
                # Set up data, decide if we are using it
                epoch_num = get_number(subdir)
                if epoch_num in seen_epochs:
                    continue
                else:
                    seen_epochs.add(epoch_num)
                rewards_pkl = os.path.join(eval_path, subdir)
                try:
                    rewards_lst = read_logs(rewards_pkl)
                    rewards = [r['reward'] for r in rewards_lst]
                    lengths = [r['length'] for r in rewards_lst]
                except Exception as e:
                    print(f'Could not read {rewards_pkl}', e)
                    continue

                # Save all rewards over all epochs
                data_all_epochs.append((epoch_num, (rewards, lengths)))

    # I think I need to log them in order to show up properly... may not be necessary, misses one or two
    data_all_epochs = sorted(data_all_epochs, key=lambda x: x[0])
    for epoch_num, (reward, length) in data_all_epochs:
        reward = np.array(reward)
        avg_reward = np.mean(reward)
        length = np.array(length)
        avg_length = np.mean(length)
        print(f'logging epoch {epoch_num} with r={avg_reward} of length {avg_length}')
        log(mlog, 'rewards_all_epochs', avg_reward, phase='val')
        log(mlog, 'rewards_histogram', reward, phase='val')

        log(mlog, 'lengths_all_epochs', avg_length, phase='val')
        log(mlog, 'lengths_histogram', length, phase='val')
        reset_log(mlog, None, epoch_num, phase='val')


def save_train_testing(exp_paths, mlog):
    # Training logs
    train_result_paths = combined_paths(exp_paths, 'result_log.pkl')
    save_training_logs(train_result_paths, mlog)

    eval_paths = combined_paths(exp_paths, 'eval')
    save_testing_logs(eval_paths, mlog)

