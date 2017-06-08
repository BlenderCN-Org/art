import tensorflow as tf


def get_var(varname):
    return [v for v in tf.global_variables() if v.name == varname][0]


target = tf.placeholder(tf.float32, [1, 2], name='targets')
inputs_ = tf.placeholder(tf.float32, [1, 2], name='inputs')

with tf.variable_scope('main'):
    main_out = tf.contrib.layers.fully_connected(inputs_, 2)

with tf.variable_scope('copy'):
    copy_out = tf.contrib.layers.fully_connected(inputs_, 2)

loss = tf.reduce_mean(tf.square(target - main_out))
optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)



vars2copy = []
for vvar in tf.global_variables():
    if vvar.name.startswith('copy/'):
        vars2copy.append(vvar.name[5:])

copying = []
for vvar in vars2copy:
    fromvar = get_var('main/{}'.format(vvar))
    tovar = get_var('copy/{}'.format(vvar))
    #print (fromvar, tovar)
    copying.append(tovar.assign(fromvar))

#print (copying)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feed = {
        inputs_: [[1, 2]],
        target: [[2, 4]]
    }
    _, first_out = sess.run([optimizer, main_out], feed_dict=feed)
    # Apparently TF first get main_out and then optimize
    # so the value don't correspond
    first_out = sess.run(main_out, feed_dict=feed)
    second_out = sess.run(copy_out, feed_dict=feed)

    print ('copy/fully_connected/weights:0')
    print (sess.run(get_var('copy/fully_connected/weights:0')))
    print ('main/fully_connected/weights:0')
    print (sess.run(get_var('main/fully_connected/weights:0')))

    sess.run(copying) # Perform parameters copy

    print ('copy/fully_connected/weights:0')
    print (sess.run(get_var('copy/fully_connected/weights:0')))
    print ('mani/fully_connected/weights:0')
    print (sess.run(get_var('main/fully_connected/weights:0')))

    third_out = sess.run(copy_out, feed_dict=feed)

    print ("First Out (Main Out)")
    print (first_out)

    print ("Second Out (Copy Out)")
    print (second_out)

    print ("Third Out (Copy Out)")
    print (third_out)
