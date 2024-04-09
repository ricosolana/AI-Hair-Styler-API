import argparse
import os
import cv2


def save_samples(args):
    im_path1 = os.path.join(args.input_dir, args.im_path1)
    im_path2 = os.path.join(args.input_dir, args.im_path2)
    im_path3 = os.path.join(args.input_dir, args.im_path3)

    im_name_1 = os.path.splitext(os.path.basename(im_path1))[0]
    im_name_2 = os.path.splitext(os.path.basename(im_path2))[0]
    im_name_3 = os.path.splitext(os.path.basename(im_path3))[0]

    #
    output_image_path = os.path.join(args.output_dir,
                                     '{}_{}_{}_{}.png'.format(im_name_1, im_name_2, im_name_3, args.sign))

    # fake image to show we did some "processing"
    image = cv2.imread(os.path.abspath(im_path1))
    inverted_image = ~image

    cv2.imwrite(output_image_path, inverted_image)

    #with open(output_image_path, 'w') as out_img:
        #out_img.write('this is an image. :)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Barbershop')

    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input/face',
                        help='The directory of the images to be inverted')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='The directory to save the latent codes and inversion images')
    parser.add_argument('--im_path1', type=str, default='16.png', help='Identity image')
    parser.add_argument('--im_path2', type=str, default='15.png', help='Structure image')
    parser.add_argument('--im_path3', type=str, default='117.png', help='Appearance image')
    parser.add_argument('--sign', type=str, default='realistic', help='realistic or fidelity results')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')

    args = parser.parse_args()

    save_samples(args)
