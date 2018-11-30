from __future__ import print_function
import sys
sys.path.insert(0, 'src')
import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy
from moviepy.video.io.VideoFileClip import VideoFileClip
import moviepy.video.io.ffmpeg_writer as ffmpeg_writer
import cv2 as cv
import matplotlib.pyplot as plt

BATCH_SIZE = 4
DEVICE = '/gpu:0'


def ffwd_video(path_in, path_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    video_clip = VideoFileClip(path_in, audio=False)
    video_writer = ffmpeg_writer.FFMPEG_VideoWriter(path_out, video_clip.size, video_clip.fps, codec="libx264",
                                                    preset="medium", bitrate="2000k",
                                                    audiofile=path_in, threads=None,
                                                    ffmpeg_params=None)

    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size, video_clip.size[1], video_clip.size[0], 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        X = np.zeros(batch_shape, dtype=np.float32)
        frame_count = 0  # The frame count that written to X
        curFrame = np.zeros(batch_shape[1:])
        prevFrame = np.zeros(batch_shape[1:])
        prevStyledFrame = np.zeros(batch_shape[1:])

        def rgb2gray(rgb):

            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

            return gray

        def style_and_write(count, prevStyledFrame, prevFrame, curFrame):
            for i in range(count, batch_size):
                X[i] = X[count - 1]  # Use last frame to fill X
            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for i in range(0, count):
                newStyledFrame = np.clip(_preds[i], 0, 255).astype(np.uint8)
                if not prevStyledFrame.any():
                    # print("inital")
                    video_writer.write_frame(newStyledFrame)

                else:
                    nxt = cv.cvtColor(curFrame,cv.COLOR_BGR2GRAY)
                    prv = cv.cvtColor(prevFrame,cv.COLOR_BGR2GRAY)
                    flow = cv.calcOpticalFlowFarneback(prv,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])

                    normalizedMag = np.array(cv.normalize(mag,None,0,255,cv.NORM_MINMAX))

                    threshold = 25
                    sigChange = normalizedMag > threshold

                    #plt.imshow(sigChange.astype(np.double))
                    #plt.show()

                    newStyledFrame[np.logical_not(sigChange)] = prevStyledFrame[np.logical_not(sigChange)]

                    defaultStyledFrame = np.clip(_preds[i], 0, 255).astype(np.uint8)

                    # Penalizing pixels with large optical flow by choosing the
                    # default style frame instead
                    penalties = normalizedMag / np.max(normalizedMag)
                    penalties = np.repeat(penalties[:,:,np.newaxis], 3, axis=2)
                    newStyledFrame = (newStyledFrame*(1-penalties) + defaultStyledFrame*(penalties)).astype(np.uint8)

                    #newStyledFrame = (newStyledFrame*0.5 + defaultStyledFrame*0.5).astype(np.uint8) # Averaging images together

                    # Treating all the dots as salt and pepper and Gaussian noise.
                    # Best filter to use seems to be a median filter
                    newStyledFrame[:, :, 0] = ndimage.median_filter(newStyledFrame[:, :, 0], 5)
                    newStyledFrame[:, :, 1] = ndimage.median_filter(newStyledFrame[:, :, 1], 5)
                    newStyledFrame[:, :, 2] = ndimage.median_filter(newStyledFrame[:, :, 2], 5)

                    #plt.imshow(newStyledFrame)
                    #plt.show()

                    # print("other")
                    #diff = curFrame - prevFrame.astype(np.uint8)
                    #diff = np.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2 + diff[:, :, 2]**2)
                    # print(diff.mean())
                    # input()
                    #diff = diff > diff.mean()

                    #newStyledFrame[np.logical_not(diff)] = prevStyledFrame[np.logical_not(diff)]

                    # print(newStyledFrame)
                    #newStyledFrame[:, :, 0] = gaussian_filter(newStyledFrame[:, :, 0], 1)
                    #newStyledFrame[:, :, 1] = gaussian_filter(newStyledFrame[:, :, 1], 1)
                    #newStyledFrame[:, :, 2] = gaussian_filter(newStyledFrame[:, :, 2], 1)
                    # input()
                    video_writer.write_frame(newStyledFrame)

            #return np.clip(_preds[i], 0, 255).astype(np.uint8)
            return newStyledFrame


        for frame in video_clip.iter_frames():
            if not prevFrame.any() and not curFrame.any():
                X[frame_count] = frame
                curFrame = frame
            else:
                prevFrame = curFrame.copy()
                curFrame = frame.copy()
                X[frame_count] = curFrame.copy()

            frame_count += 1
            if frame_count == batch_size:

                prevStyledFrame = style_and_write(frame_count, prevStyledFrame, prevFrame, curFrame)
                frame_count = 0

        if frame_count != 0:
            style_and_write(frame_count, prevStyledFrame, prevFrame, curFrame)

        video_writer.close()

# get img_shape
def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    g = tf.Graph()
    batch_size = min(len(paths_out), batch_size)
    curr_num = 0
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t), \
            tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        if os.path.isdir(checkpoint_dir):
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, checkpoint_dir)

        num_iters = int(len(paths_out)/batch_size)
        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos+batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos+batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' +  \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos+batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder:X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])

        remaining_in = data_in[num_iters*batch_size:]
        remaining_out = paths_out[num_iters*batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, checkpoint_dir,
            device_t=device_t, batch_size=1)

def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)

def ffwd_different_dimensions(in_path, out_path, checkpoint_dir,
            device_t=DEVICE, batch_size=4):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape],
            checkpoint_dir, device_t, batch_size)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path',help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device',help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size',help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions',
                        help='allow different image dimensions')

    return parser

def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = \
                    os.path.join(opts.out_path,os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir,
                    device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path,x) for x in files]
        full_out = [os.path.join(opts.out_path,x) for x in files]
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir,
                    device_t=opts.device, batch_size=opts.batch_size)
        else :
            ffwd(full_in, full_out, opts.checkpoint_dir, device_t=opts.device,
                    batch_size=opts.batch_size)

if __name__ == '__main__':
    main()
