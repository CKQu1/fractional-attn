import argparse
import os

#from main import train_vit
"""
ss=model_name=fnsvit,manifold=rd,alpha=1.2,a=0,bandwidth=1,is_op=True,dataset=mnist,model_root=/taiji1/chqu7424/fractional-attn/vit-pytorch/.droot/1L-ps=1-v3/config_qqv/mnist/layers=1-heads=1-hidden=48-qqv,seed=1,qk_share=True,n_layers=1,hidden_size=48,patch_size=1,weight_decay=0,lr_scheduler_type=binary,max_lr=0.001,min_lr=0.0001,epochs=45,n_attn_heads=1,train_bs=32,is_rescale_dist=True
ss
'--' + (' --'.join(ss.split(',')))
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='long-range-arena/main.py training arguments')   
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

    