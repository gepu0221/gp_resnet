from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import sys
import TensorflowUtils as utils
import read_expression_data as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange
sys.path.append('.')

try:
    from cfgs.config import FLAGS
except Exception:
    from cfgs.config import FLAGS


def get_kernel_bias(n, weights, name):
    kernels = weights[n][1]
    print('the shape of kernel',kernels.shape)
    n += 2
    bias = weights[n][1]
    n += 2
    kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
    bias = utils.get_variable(bias.reshape(-1), name=name + "_b")

    return kernels, bias, n
def get_kernel_bias_res(n, weights, name, i):
    kernels = weights[n][1]
    n += 2
    bias = weights[n][1]
    n += 2
    out_chan = kernels.shape[i]
    kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
    bias = utils.get_variable(bias.reshape(-1), name=name + "_b")

    return kernels, bias, out_chan, n



def res_net(weights, image):
    layers = ('conv1',
             'res2a', 'res2b', 'res2c',
             'res3a', 'res3b1', 'res3b2', 'res3b3',
             'res4a', 'res4b1', 'res4b2', 'res4b3', 'res4b4',
             'res4b5', 'res4b6', 'res4b7', 'res4b8', 
             'res4b9', 'res4b10', 'res4b11', 'res4b12',
             'res4b13', 'res4b14', 'res4b15', 'res4b16',
             'res4b17', 'res4b18', 'res4b19', 'res4b20', 
             'res4b21', 'res4b22', 
            'res5a', 'res5b', 'res5c'
              )
    res_branch1_parm={
    #branch1:padding(if 0 then 'VALID' else 'SMAE'),stride
    'res2a':[0,1,0,1,1,1,0,1],
    'res3a':[0,2,0,2,1,1,0,1],
    'res4a':[0,2,0,2,1,1,0,1],
    'res5a':[0,2,0,2,1,1,0,1]}

    net = {}
    current = image
    n = 0
    for i, name in enumerate(layers):
        kind = name[:3]
        print('-----------------layer: %s-----------------------' % name)
        print('Input: ', current.shape)
        #convolutional
        if kind == 'con':
            kernels ,bias ,n = get_kernel_bias(n, weights, name)
            print('kernel size: ', kernels.shape, 'bias size', bias.shape)
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            current = utils.conv2d_strided(current, kernels, bias)
            current = utils.max_pool_2x2(current, 3)
        #resnet
        elif kind == 'res':
            sub_kind = name[4]
            #not blockneck
            if sub_kind == 'a':
                res_param = res_branch1_parm[name]
                branch1_name = '%s_%s' % (name, 'branch1')
                branch1_w, branch1_b, out_chan_t, n = get_kernel_bias_res(n, weights, branch1_name, 2)
                print('branch1:kernel size: ', branch1_w.shape, 'bias size', branch1_b.shape)
            else:
                res_param = None
                branch1_w = None
            branch2a_name = '%s_%s' % (name, 'branch2a')
            branch2a_w, branch2a_b, out_chan_t2, n = get_kernel_bias_res(n, weights, branch2a_name, 0)
            print('branch2a:kernel size: ', branch2a_w.shape, 'bias size', branch2a_b.shape)
            branch2b_name = '%s_%s' % (name, 'branch2b')
            branch2b_w, branch2b_b, _, n = get_kernel_bias_res(n, weights, branch2b_name, 0)
            print('branch2b:kernel size: ', branch2b_w.shape, 'bias size', branch2b_b.shape)
            branch2c_name = '%s_%s' % (name, 'branch2c')
            branch2c_w, branch2c_b, out_chan2, n = get_kernel_bias_res(n, weights, branch2c_name, 3)
            print('branch2c:kernel size: ', branch2c_w.shape, 'bias size', branch2c_b.shape)
            if sub_kind == 'a':
                out_chan1 = out_chan_t
            else:
                out_chan1 = out_chan_t2
            current = utils.bottleneck_unit(current, res_param, 
                                            branch1_w, branch1_b, branch2a_w, branch2a_b, branch2b_w, branch2b_b, branch2c_w, branch2c_b, 
                                            out_chan1, out_chan2, False, False, name)
        print('layer output ', current.shape)
        net[name] = current
    current = utils.avg_pool(current, 7, 1)
    print('resnet final sz ', current.shape )
    #return net
    return current


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, FLAGS.MODEL_NAME)

    #mean = model_data['normalization'][0][0][0]
    #mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['params'][0])

    #processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = res_net(weights, image)
        #conv_final_layer = image_net["res5c"]
        conv_final_layer = image_net
        
        fc_w = utils.weight_variable([1, 1, 2048, FLAGS.NUM_OF_CLASSESS], name="fc_w")
        fc_b = utils.bias_variable([FLAGS.NUM_OF_CLASSESS], name="fc_b")
        fc = utils.conv2d_basic(conv_final_layer, fc_w, fc_b)
       
        fc_dropout = tf.nn.dropout(fc, keep_prob)
        logits = tf.squeeze(fc_dropout, [1,2])
        #logits = tf.squeeze(utils.conv2d_basic(conv_final_layer, fc_w, fc_b), [1,2])
        print('logits shape', logits.shape)

        annotation_pred = tf.argmax(logits, dimension=1, name="prediction")

    return tf.expand_dims(annotation_pred, dim=1), logits
    
    #return conv_final_layer

def train(loss_val, var_list, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    batch_sz = tf.placeholder(tf.int32, name='batch_size')
    image = tf.placeholder(tf.float32, shape=[None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, 1], name="annotation")
    
    #logits = inference(image, keep_probability)
    pred_annotation, logits = inference(image, keep_probability)
    #logits:the last layer of conv net
    #labels:the ground truth

    loss_weight = tf.pow(1-tf.reduce_sum(logits * tf.one_hot(tf.squeeze(annotation, squeeze_dims=[1]), FLAGS.NUM_OF_CLASSESS), 1), FLAGS.gamma)

    
    loss = tf.reduce_mean(loss_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[1]),
                                                                          name="entropy"))
    loss_train_nodp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[1]),
                                                                          name="entropy_train_nodp"))
    loss_valid = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[1]),
                                                                          name="entropy_valid"))


    #read learning rate from file
    path_lr = FLAGS.learning_rate_path
    with open(path_lr, 'r') as f:
        lr_ = float(f.readline().split('\n')[0])
    print('lr_:%', learning_rate)
    tf.summary.scalar("entropy", loss)
    tf.summary.scalar("entropy_train_nodp", loss_train_nodp)
    tf.summary.scalar("entropy_valid", loss_valid)
    tf.summary.scalar('learning_rate', learning_rate)
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var, learning_rate)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records, test_records= scene_parsing.my_read_dataset(FLAGS.data_dir)
    print(train_records[0])
    print('number of train_records',len(train_records))
    print('number of valid_records',len(valid_records))
    print('number of test_records', len(test_records))
  
    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': FLAGS.IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep = 0)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    #if not train,restore the model trained before
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    loss_min = 10000
    valid_loss_min = 10000

    t_start = time.time()
    th_ = 36000

    if FLAGS.mode == "train":
        current_itr = FLAGS.train_itr
        for itr in xrange(current_itr, FLAGS.MAX_ITERATION):
            t_e = time.time() - t_start
            if t_e > th_:
                th_ += 36000
                lr_ /= 10
                
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict_train = {image: train_images, annotation: train_annotations, keep_probability: 0.5, learning_rate: lr_ }
            #sess.run(train_op, feed_dict=feed_dict_train)
            train_loss, train_op_, summary_str = sess.run([loss, train_op, summary_op], feed_dict=feed_dict_train)
            summary_writer.add_summary(summary_str, itr)
            if itr % 10 == 0:
                feed_dict_train_nodp = {image: train_images, annotation: train_annotations, keep_probability: 1.0, learning_rate: lr_}
                train_loss_nodp, summary_str_nodp = sess.run([loss_train_nodp, summary_op], feed_dict=feed_dict_train_nodp)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                print("Step: %d, Train_loss_nodp:%g" % (itr, train_loss_nodp))

                summary_writer.add_summary(summary_str_nodp, itr)

            if itr % 10 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss, summary_str_valid = sess.run([loss_valid, summary_op], feed_dict={image: valid_images, annotation: valid_annotations, learning_rate: lr_,
                                                       keep_probability: 1})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                summary_writer.add_summary(summary_str_valid, itr)
            if itr % 500 == 0:
                '''
                print('train_loss:%g, loss_min:%g' % (train_loss, loss_min))
                print('valid_loss_min:%g, valid_loss:%g' % (valid_loss_min, valid_loss))

                if train_loss < loss_min or valid_loss_min < valid_loss:
                    #print('train_loss:%g, loss_min:%g' % (train_loss, loss_min))
                    #print('valid_loss_min:%g, valid_loss:%g' % (valid_loss_min, valid_loss))
                    loss_min = train_loss
                    valid_loss_min = valid_loss
                    print(itr)'''
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

           
        

if __name__ == "__main__":
    tf.app.run()
