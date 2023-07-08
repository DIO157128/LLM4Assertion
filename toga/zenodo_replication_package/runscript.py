import os
import multiprocessing
import subprocess


def run_command(command, output_file):
    with open(output_file, 'w') as f:
        subprocess.call(command, stdout=f, shell=True)
if __name__ == '__main__':

    commands = [
        'python toga.py data/evosuite_buggy_regression_all/1/inputs.csv data/evosuite_buggy_tests/1/meta.csv',
        'python toga.py data/evosuite_buggy_regression_all/2/inputs.csv data/evosuite_buggy_tests/2/meta.csv',
        'python toga.py data/evosuite_buggy_regression_all/3/inputs.csv data/evosuite_buggy_tests/3/meta.csv',
        'python toga.py data/evosuite_buggy_regression_all/4/inputs.csv data/evosuite_buggy_tests/4/meta.csv',
        'python toga.py data/evosuite_buggy_regression_all/5/inputs.csv data/evosuite_buggy_tests/5/meta.csv'
    ]


    max_concurrent_processes = 5

    pool = multiprocessing.Pool(processes=max_concurrent_processes)




    for idx,command in enumerate(commands):
        output_file = './{}.txt'.format(idx)
        pool.apply_async(run_command, args=(command, output_file))

    pool.close()

    pool.join()