#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 15:49:52 2017

@author: wd
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:56:43 2017

@author: wd
"""
import numpy as np
import tensorflow as tf
import PG_supervised_CNN_action_class
import io
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import gym

# #############################################################################
# PARAMETERS
# #############################################################################

tensorboard_log_path = 'data/board/test2'
net_name = 'eye2'
num_epochs = 500
batch_size = 100
train_size = 100000

# #############################################################################


env = gym.make('ND-v0')
output_size = env.action_space.n  # number of actions
image_width = 28
input_size = image_width * image_width


def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


def get_result_figure(input_data, actions):

    # line corners
    margin = 0
    line_thickness = 20
    x_int = [[image_width-1-margin, image_width-1-margin], [margin, image_width-1-margin],
             [margin, margin], [margin, image_width-1-margin]]
    y_int = [[margin, image_width-1-margin], [image_width-1-margin, image_width-1-margin],
             [margin, image_width-1-margin], [margin, margin]]

    # reshape input array
    image = input_data.reshape(image_width, image_width)

    # draw image
    fig = plt.figure(num=0)
    fig.clf()
    plt.imshow(image)
    ax = plt.gca()

    # draw indication line
    cur_action = np.argmax(actions)
    if 0 == cur_action:
        for i in range(4):
            ax.add_line(mlines.Line2D(x_int[i], y_int[i], color='r', linewidth=line_thickness))
    else:
        ax.add_line(mlines.Line2D(x_int[cur_action-1], y_int[cur_action-1], color='r', linewidth=line_thickness))

    # title
    plt.title("input image and output direction")
    # plt.show()

    return fig


def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def make_labels(train_batch, hit_range=1):

    x_stack = np.empty(0).reshape(0, 784)
    a_stack = np.empty(0).reshape(0, 5)
    c_stack = np.empty(0).reshape(0, 2)

    # Get stored information from the buffer

    for state, state_pt, gt in train_batch:
        cur_class = [0, 0]
        cur_action = [0, 0, 0, 0, 0]
        diff = gt - state_pt
        d_x, d_y = diff[0], diff[1]

        # class label

        distance = d_x * d_x + d_y * d_y
        not_hit = 0 if hit_range * hit_range > distance else 1
        cur_class[not_hit] = 1

        # action label


        if 1 == not_hit:
            if abs(d_x) > abs(d_y):
                if d_x > 0:
                    action_label = 1  # right
                else:
                    action_label = 3  # left
            else:
                if d_y > 0:
                    action_label = 2  # down
                else:
                    action_label = 4  # up
        else:
            action_label = 0
        cur_action[action_label] = 1

        # stacking label data

        x_stack = np.vstack([x_stack, np.reshape(state, 784)])
        a_stack = np.vstack([a_stack, cur_action])
        c_stack = np.vstack([c_stack, cur_class])

    return x_stack, a_stack, c_stack


def main():

    training_batch = []
    test_batch = []

    # =================================================================
    # image visualize
    # =================================================================

    # prepare the plot
    fig = get_result_figure(np.zeros((1, input_size)), [1, 0, 0, 0, 0])
    vis_placeholder = tf.placeholder(tf.uint8, fig2rgb_array(fig).shape)
    vis_summary1 = tf.summary.image('input_output', vis_placeholder)


    # =================================================================
    # Training data sampling
    # =================================================================

    for j in range(50):
        state2, state_pt2, gt2, _ = env.reset()
        test_batch.append((state2, state_pt2, gt2))

    for i in range(train_size):
        state, state_pt, gt, _ = env.reset()
        training_batch.append((state, state_pt, gt))

    with tf.Session() as sess:

        SL_net = PG_supervised_CNN_action_class.CNN_SL(sess, input_size, output_size, net_name)
        writer = tf.summary.FileWriter(tensorboard_log_path, sess.graph)
        tf.global_variables_initializer().run()

        # =================================================================
        # Training loop
        # =================================================================
        count_iter = 0
        iter_per_epoch = np.floor(train_size / batch_size + 1)

        for epoch in range(num_epochs):
            np.random.shuffle(training_batch)

            start_pos = 0

            while start_pos < train_size:
                end_pos = min(start_pos + batch_size, train_size - 1)
                cur_batch = training_batch[start_pos:end_pos]
                x_batch, a_batch, c_batch = make_labels(cur_batch)
                accuracy, _, loss_summary = SL_net.update(x_batch, a_batch, c_batch)

                # visualize

                writer.add_summary(loss_summary, count_iter)
                print('[%03d/%03d](%05d/%05d) Accuracy: %.3f' %
                      (epoch, num_epochs, count_iter, iter_per_epoch, accuracy))

                # next batch
                count_iter += 1
                start_pos += batch_size

                # draw image
                if 0 == count_iter % 10:
                    actions = SL_net.predict(x_batch[0])
                    fig = get_result_figure(x_batch[0], actions)
                    image = fig2rgb_array(fig)
                    name = 'image/'+str(epoch)+str(count_iter)+'.png'
                    # open file and write file
                    file = open(name, 'w')
                    file.write(image)
                    file.close()
                    # add image summary on tensorboard
                    writer.add_summary(vis_summary1.eval(feed_dict={vis_placeholder: image}))


        # save point
        np.save('data/%s_weight1.npy' % net_name, SL_net.get_wc1())
        np.save('data/%s_weight2.npy' % net_name, SL_net.get_wc2())
        np.save('data/%s_weight3.npy' % net_name, SL_net.get_wc3())
        np.save('data/%s_cnnweight.npy' % net_name, SL_net.get_cnn_weights())

        writer.close()


if __name__ == "__main__":
    main()
