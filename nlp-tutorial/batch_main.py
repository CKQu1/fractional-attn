import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='nlp-tutorial/main.py training arguments')   
    # training settings 
    #parser.add_argument('--train_with_ddp', default=True, type=bool, help='to use DDP or not')
    parser.add_argument('--arg_strss', type=str)
    parser.add_argument('--script', default='main.py', type=str)
    parser.add_argument('--script_command', default='python3', type=str)

    args = parser.parse_args()    

    arg_strss = args.arg_strss.split(';')
    for arg_strs in arg_strss:
        #arg_strs = '--' + 'arg_strs.replace(',',' ')        
        arg_strs = '--' + ' --'.join(arg_strs.split(','))

        os.system(f'{args.script_command} {args.script} {arg_strs}')            