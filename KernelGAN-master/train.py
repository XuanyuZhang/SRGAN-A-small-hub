import os
import tqdm

from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner


def train(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)

    # count total number of parameters
    #total_num = sum(p.numel() for p in gan.parameters())
    #trainable_num = sum(p.numel() for p in gan.parameters() if p.requires_grad)
    #print('Total_parameters: ', total_num)
    #print('trainable_parameter_num: ', trainable_num)
    gan.finish()


# 服务器测试命令
# python train.py --input-dir ../ESRGAN-master/ESRGAN-master/bottle/ --X4 --SR --real
# python train.py --input-dir /data/zhangxuanyu/datasets/bottle/Images_bicubic_x4/ --X4 --SR --real
# python train.py -i /home/zhangxuanyu/GAN/KernelGAN-master/test_images/ --X4 --SR --real
# python train.py --X4 --SR --real

def main():
    """The main function - performs kernel estimation (+ ZSSR) for all images in the 'test_images' folder"""
    import argparse
    # Parse the command line arguments
    prog = argparse.ArgumentParser()
    prog.add_argument('--input-dir', '-i', type=str, default='test_images', help='path to image input directory.')
    prog.add_argument('--output-dir', '-o', type=str, default='results', help='path to image output directory.')
    prog.add_argument('--X4', action='store_true', help='The wanted SR scale factor')
    prog.add_argument('--SR', action='store_true', help='when activated - ZSSR is not performed')
    prog.add_argument('--real', action='store_true', help='ZSSRs configuration is for real images')
    prog.add_argument('--noise_scale', type=float, default=1., help='ZSSR uses this to partially de-noise images')
    args = prog.parse_args()
    # Run the KernelGAN sequentially on all images in the input directory
    for filename in os.listdir(os.path.abspath(args.input_dir)):
        conf = Config().parse(create_params(filename, args))
        train(conf)

    prog.exit(0)


def create_params(filename, args):
    params = ['--input_image_path', os.path.join(args.input_dir, filename),
              '--output_dir_path', os.path.abspath(args.output_dir),
              '--noise_scale', str(args.noise_scale)]
    if args.X4:
        params.append('--X4')
    if args.SR:
        params.append('--do_ZSSR')
    if args.real:
        params.append('--real_image')
    return params


if __name__ == '__main__':
    main()
