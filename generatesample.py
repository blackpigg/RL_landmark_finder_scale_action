import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import gym

# #############################################################################
# PARAMETERS
# #############################################################################

tensorboard_log_path = 'data/board/img'
net_name = 'eye2'
num_epochs = 1
batch_size = 1
train_size = 50000
# train_size = 100
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
    elif cur_action < 5:
        ax.add_line(mlines.Line2D(x_int[cur_action-1], y_int[cur_action-1], color='r', linewidth=line_thickness))
    elif cur_action == 5:  # Expand
        for i in range(4):
            ax.add_line(mlines.Line2D(x_int[i], y_int[i], color='g', linewidth=line_thickness))
    else:
        for i in range(4):
            ax.add_line(mlines.Line2D(x_int[i], y_int[i], color='b', linewidth=line_thickness))

    # plt.show()

    return fig


def fig2rgb_array(fig, expand=True):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
    return np.fromstring(buf, dtype=np.uint8).reshape(shape)


def make_labels(training_batch, hit_range=1):


    # Get stored information from the buffer
    state, cur_x, cur_y, gt_x, gt_y, scale, whole_img = training_batch[0]
    cur_class = [0, 0]
    cur_action = [0, 0, 0, 0, 0, 0, 0]
    d_x, d_y = gt_x-(cur_x+0.5*scale), gt_y-(cur_y+0.5*scale)
    gt_box = [(gt_x-0.5*28), (gt_y-0.5*28), (gt_x-0.5*28)+28,  (gt_y-0.5*28)+28]
    pt_box = [cur_x,  cur_y, cur_x+scale, cur_y+scale]
    cover = env.get_cover(pt_box, gt_box)

    # class label

    distance = d_x * d_x + d_y * d_y
    not_hit = 0 if hit_range * hit_range > distance else 1
    cur_class[not_hit] = 1
    # print(cover)
    # print(scale)
        # action label



    if cover > 0.5:
        if 1 == cover:
            if image_width == scale:
                action_label = 0
            else:
                action_label = 6
        else:
            if np.random.rand(1) > 0.5:
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
        action_label = 5


    # stacking label data
    cur_action[action_label] = 1

    return cur_action


def main():

    training_batch = []

    # =================================================================
    # image visualize
    # =================================================================

    # prepare the plot
    fig = get_result_figure(np.zeros((1, input_size)), [1, 0, 0, 0, 0, 0, 0])




    # =================================================================
    # Training loop
    # =================================================================
    count_iter = 0
    buffer = deque()
    sample_buffer = deque()

    for episode in range(train_size):
        done = 0
        step_count = 0
        state, cur_x, cur_y, gt_x, gt_y, scale, whole_image = env.reset()
        # plt.imshow(whole_image)
        # plt.plot(gt_x,gt_y, 'r+')
        # plt.savefig('img/'+str(episode)+'_'+'0'+'.png')
        while not done:

            buffer.append((state, cur_x, cur_y, gt_x, gt_y, scale, whole_image))
            sample_buffer.append((state, cur_x, cur_y, gt_x, gt_y, scale))
            action = make_labels(buffer)
            new_state, _, done, new_x, new_y, new_scale = env.step(action)
            state = new_state
            cur_x = new_x
            cur_y = new_y
            scale = new_scale
            step_count += 1

            if step_count > 100:
                fig = get_result_figure(state, action)
                name = 'img/'+str(episode)+'_'+str(step_count)+'.png'
                fig.savefig(name)
                gt_box = [(gt_x - 0.5 * 28), (gt_y - 0.5 * 28), (gt_x - 0.5 * 28) + 28, (gt_y - 0.5 * 28) + 28]
                pt_box = [cur_x, cur_y, cur_x + scale, cur_y + scale]
                cover = env.get_cover(pt_box, gt_box)
                print(gt_box)
                print(pt_box)
            buffer.clear()

        # if 1 == done:
        #     break
        #
        # if step_count>100:
        #     break
        print(episode)
        if 0 == episode % 1000:
            sample = np.asarray(sample_buffer)
            name = 'data/' + str(episode) + '.npy'
            np.save(name, sample)


    # save point



if __name__ == "__main__":
    main()
