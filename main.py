import os.path
import tensorflow as tf
import helper
import warnings
import project_tests as tests
import numpy as np
import scipy.misc

from distutils.version import LooseVersion
from glob import glob


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tags = [vgg_tag]
    tf.saved_model.loader.load(sess, tags, vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # I use L2 regulizer
    reg_param = 0.01
    
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, 1,
                                name = "conv11",
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_param),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    decod_1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding='same',
                                         name = "decoderlayer1",
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_param),
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    classified_layer4_out = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 1, 
                                             name = "conv11_pool4",
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_param),
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    add_1 = tf.add(decod_1, classified_layer4_out,
                   name = "decoderlayer2")
    decod_2 = tf.layers.conv2d_transpose(add_1, num_classes, 4, 2, padding='same',
                                         name = "decoderlayer3",
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_param),
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    classified_layer3_out = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 1,
                                             name = "conv11_pool3",
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_param),
                                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    add_2 = tf.add(decod_2, classified_layer3_out, 
                   name = "decoderlayer4")
    last_layer_output = tf.layers.conv2d_transpose(add_2, num_classes, 16, 8, padding='same', 
                                                   name = "decoderlayerout",
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_param),
                                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return last_layer_output
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = correct_label))
    # For L2 regularization, I calculate the regularization part
    l2_loss = tf.losses.get_regularization_loss()
    
    opt = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = opt.minimize(cross_entropy_loss + l2_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    # I use dropout method only for training
    dropout = 0.5
    epoch_learning_rate = 0.0005
    
    sess.run(tf.global_variables_initializer())

    last_batch_i = len(glob(os.path.join('./data/data_road/', 'image_2', '*.png'))) // batch_size

    for epoch in range(epochs): #Training for each epoch
        
        i = 0
        
        print('-'*10)
        print("EPOCH {}...".format(epoch+1))
        print('-'*10)
        
        if epoch == 8:
            epoch_learning_rate = 0.000005
            batch_size = 6
            #dropout = 1.0
            print("learning_rate customize to {}".format(epoch_learning_rate))
            print("batch_size customize to {}".format(batch_size))
            #print("dropout customize to {}".format(dropout))

        for batch_images, batch_labels in get_batches_fn(batch_size):
        
            
            # For the last batch, I do just an evaluation - kind of validation, with dropout = 1.0
            if i == last_batch_i: 
                feed = { input_image: batch_images,
                         correct_label: batch_labels,
                         keep_prob: 1.,
                         learning_rate: epoch_learning_rate}
                eval_loss = sess.run([cross_entropy_loss], feed_dict = feed)
            else: # For the other batches, I perform training
                feed = { input_image: batch_images,
                         correct_label: batch_labels,
                         keep_prob: dropout,
                         learning_rate: epoch_learning_rate}
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
                print('Batch {} - Cross entropy loss: {}'.format(i, loss))
            i += 1

        print(' ')
        print("EPOCH {} over ... evaluation loss: {}".format(epoch+1, eval_loss))
        print(' ')

tests.test_train_nn(train_nn)


def run():

    # --------------------------------
    # Parameters
    num_classes = 2
    image_shape = (160, 576)
    batch_size = 4
    epochs = 18
    
    # --------------------------------

    # learning like a placeholder in order to customize it according to epoch
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    data_dir = './data'
    runs_dir = './runs'
    

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        
        # I augment images just by flipping horizontally each image
        if len(glob(os.path.join(data_dir, 'data_road/training', 'image_2', '*flr.png'))) == 0: # If augmented images does not exist
            print('Image Augmentation in progress ...')
            # Input images
            image_paths = glob(os.path.join(data_dir, 'data_road/training', 'image_2', '*.png'))
            for image_file in image_paths:
                image = np.fliplr(scipy.misc.imread(image_file))
                new_path = image_file.split(".png")[0] + 'flr.png'
                scipy.misc.imsave(new_path, image)
            # label images
            label_paths = glob(os.path.join(data_dir, 'data_road/training', 'gt_image_2', '*_road_*.png'))
            for image_file in label_paths:
                image = np.fliplr(scipy.misc.imread(image_file))
                new_path = image_file.split(".png")[0] + 'flr.png'
                scipy.misc.imsave(new_path, image)

        # TODO: Build NN using load_vgg, layers, and optimize function
        image_input, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        graph = tf.get_default_graph()
        # I freeze the previous trained model by excluding the weights from trainable variable collection
        graph.clear_collection('trainable_variables')
        graph.add_to_collections('trainable_variables', tf.global_variables()[-6])
        graph.add_to_collections('trainable_variables', tf.global_variables()[-5])
        graph.add_to_collections('trainable_variables', tf.global_variables()[-4])
        graph.add_to_collections('trainable_variables', tf.global_variables()[-3])
        graph.add_to_collections('trainable_variables', tf.global_variables()[-2])
        graph.add_to_collections('trainable_variables', tf.global_variables()[-1])
        
        # I exclude the previous regularization from collection to ignore them
        graph.clear_collection( 'regularization_losses')

        # Scaling as it says in "read me" tips - I have checked that is better with
        vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name='layer3_out_scaled')
        vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name='layer4_out_scaled')

        nn_last_layer = layers(vgg_layer3_out_scaled, vgg_layer4_out_scaled, vgg_layer7_out, num_classes)
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.s   ave_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob,
        image_input)
        
        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
