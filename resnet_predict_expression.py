from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import TensorflowUtils as utils
import read_expression_data as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs20180613_expression/", "path to logs directory")
#tf.flags.DEFINE_string("logs_dir", "train_model/pretrained_dropout/learning_rate0_0001", "path to dataset")
#tf.flags.DEFINE_string("data_dir", "crop_save", "path to dataset")
tf.flags.DEFINE_string("data_dir", "datasets", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "validation", "Mode train/ test/ visualize")
#tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")

MODEL_NAME = "imagenet-resnet-101-dag.mat"

MAX_ITERATION = int(1e5 + 1)
NUM_OF_CLASSESS = 7
IMAGE_SIZE = 224

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
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_NAME)

    #mean = model_data['normalization'][0][0][0]
    #mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['params'][0])

    #processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = res_net(weights, image)
        #conv_final_layer = image_net["res5c"]
        conv_final_layer = image_net
        
        fc_w = utils.weight_variable([1, 1, 2048, NUM_OF_CLASSESS], name="fc_w")
        fc_b = utils.bias_variable([NUM_OF_CLASSESS], name="fc_b")
        logits = tf.squeeze(utils.conv2d_basic(conv_final_layer, fc_w, fc_b), [1,2])
        print('logits shape', logits.shape)

        annotation_pred = tf.argmax(logits, dimension=1, name="prediction")

    return tf.expand_dims(annotation_pred, dim=1), logits
    
    #return conv_final_layer

def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, 1], name="annotation")
    one_hot_anno = tf.placeholder(tf.int32, shape=[None, NUM_OF_CLASSESS], name="one_hot_anno")

    #logits = inference(image, keep_probability)
    pred_annotation, logits = inference(image, keep_probability)
    #tf.summary.image("input_image", image, max_outputs=2)
    #tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    #tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    #logits:the last layer of conv net
    #labels:the ground truth
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[1]),
                                                                          name="entropy"))
    softmax_pro = tf.nn.softmax(logits, name='soft_pro')
    '''
    sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=softmax_pro, 
                                                           labels=one_hot_anno,
                                                           name='sigmoid_entropy')'''
    tf.summary.scalar("entropy", loss)
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records, test_records= scene_parsing.my_read_dataset(FLAGS.data_dir)
    print('number of train_records',len(train_records))
    print('number of valid_records',len(valid_records))
    print('number of test_records', len(test_records))
  
    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'validation':
        validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    elif FLAGS.mode == 'test':
        test_dataset_reader = dataset.BatchDatset(test_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    #if not train,restore the model trained before
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "validation":
        count=0
        p_count=0
        if_con=True
        turn=0
        loss_=0
        while if_con:
            turn=turn+1
            valid_images, valid_annotations, valid_filenames, if_con, start, end=validation_dataset_reader.next_batch_valid(FLAGS.batch_size)
            #one_hot_annotations = utils.one_hot_convert(valid_annotations)
            valid_loss,pred_anno=sess.run([loss,pred_annotation],feed_dict={image:valid_images,
            #valid_loss,pred_anno,valid_sigmoid_loss=sess.run([loss,pred_annotation,sigmoid_loss],feed_dict={image:valid_images,
                                                                                                            annotation:valid_annotations,
                                                                                                            #one_hot_anno:one_hot_annotations, 
                                                                                                            keep_probability:1.0})
            loss_ += valid_loss
            len_ = len(pred_anno)
            #pred_anno = np.expand_dims(pred_anno, axis=1)
            for i in range(len_):
                count +=1
                #print('valid_annotation: %d, pred_anno: %d'%(valid_annotations[i][0], pred_anno[i][0]))
                #print('valid_annotation: ', valid_annotations[i][0], 'pred_anno: ',pred_anno[i][0])               
                if int(valid_annotations[i][0]) == int(pred_anno[i][0]):
                    p_count += 1
                    #print('p_count : %d' % p_count)
            print('Turn%d: valid_loss: %g' % (turn, valid_loss))
            #print('Turn%d: valid_sigmoid_loss: %g' % (turn, valid_sigmoid_loss))
        print('precision: %g' % (p_count/count*100))
        print('mean loss: %g' % (loss_/turn))
           
    elif FLAGS.mode == "test":
       f = open('re.txt', 'w')
       f_ = open('re_8500_0_4.txt', 'w')
       count=0
       p_count=0
       if_con=True
       turn=0
       loss_=0
       while if_con:
           turn=turn+1
           valid_images, valid_annotations, valid_filenames, if_con, start, end=test_dataset_reader.next_batch_valid(FLAGS.batch_size)
           #one_hot_annotations = utils.one_hot_convert(valid_annotations)
           valid_loss,pred_anno,softmax_pro_=sess.run([loss,pred_annotation,softmax_pro],feed_dict={image:valid_images,
           #valid_loss,pred_anno,valid_sigmoid_loss=sess.run([loss,pred_annotation,sigmoid_loss],feed_dict={image:valid_images,
                                                                                                           annotation:valid_annotations,
                                                                                                           #one_hot_anno:one_hot_annotations, 
                                                                                                           keep_probability:0.85})
           loss_ += valid_loss
           len_ = len(pred_anno)
           #pred_anno = np.expand_dims(pred_anno, axis=1)
           for i in range(len_):
               count +=1
               f.write('%d %g\n' % (count, softmax_pro_[i][1]))
               f_.write('%g\n' %  softmax_pro_[i][1])   
               if valid_annotations[i][0] == pred_anno[i][0]:
                   p_count += 1
           print('Turn%d: valid_loss: %g' % (turn, valid_loss))
           #print('Turn%d: valid_sigmoid_loss: %g' % (turn, valid_sigmoid_loss))
       print('precision: %g' % (p_count/count*100))
       print('mean loss: %g' % (loss_/turn))   
       f.close()
       f_.close()


           
        

if __name__ == "__main__":
    tf.app.run()
