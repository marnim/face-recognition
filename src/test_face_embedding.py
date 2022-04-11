from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from os import listdir
from os.path import isfile, join
import cv2

def get_paths(lfw_dir, dirs, file_ext):
	nrof_skipped_pairs = 0
	path_list = []
	for d in dirs:
		#print(d)
		onlyfiles = [f for f in listdir(d) if isfile(join(d, f))]
		for f in onlyfiles:
			path_list.append(os.path.join(d, f))
	return path_list

def read_image_dirs(direc):
	p = [direc+'/'+name for name in os.listdir(direc) if os.path.isdir(os.path.join(direc, name))]
	print(p)
	return p

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Read the file containing the pairs used for testing
            pairs = read_image_dirs(os.path.expanduser(args.lfw_pairs))

            # Get the paths for the corresponding images
            paths = get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

            with open('test_images'+'.txt', 'w') as the_file:
                for p in paths:
                    the_file.write(p+'\n')

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on LFW images')
            batch_size = args.lfw_batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            np.savetxt('test_embedding'+'.txt', emb_array)
    with open('embedding'+'.txt', 'r') as the_file:
        emb = [row.strip().split(' ') for row in the_file]
        n = len(emb)

    with open('test_embedding'+'.txt', 'r') as the_file :
        testemb = [row.strip().split(' ') for row in the_file]
        n = len(testemb)

    min_dist = sys.float_info.max
    match_dir = ''
    with open('images'+'.txt', 'r') as the_file:
        cnt = 0
        for line in the_file:
            e = emb[cnt]
            cnt += 1
            t = testemb[0]
            a = np.array(e, dtype = 'float32')
            b = np.array(t, dtype = 'float32')
            diff = np.subtract(a,b)
            dist = np.sum(np.square(diff), axis = 0)
            if dist < min_dist:
                min_dist = dist
                match_dir = line
    print(match_dir)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('lfw_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='./datasets/image_dir_resize/lfw_mtcnnpy_160')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
