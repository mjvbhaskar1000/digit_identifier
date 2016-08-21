import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#n_nodes_hl1 = 500
#n_nodes_hl2 = 100
#n_nodes_hl3 = 500
n_nodes = [500,500,500,500,500,500]

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	'''
	h1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])), 
			'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	h2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
			'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	h3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
			'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	op_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
			'biases':tf.Variable(tf.random_normal([n_classes]))}
	
	l1 = tf.add(tf.matmul(data, h1_layer['weights']),  h1_layer['biases'])
	l1 = tf.nn.relu(l1)
	
	l2 = tf.add(tf.matmul(l1, h2_layer['weights']), h2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, h3_layer['weights']), h3_layer['biases'])
	l3 = tf.nn.relu(l3)

	op = tf.matmul(l3, op_layer['weights']) + op_layer['biases']
	'''
	h_layer = []
	op_layer = []
	for _ in range(len(n_nodes)):
		if _ == 0:
			h_layer.append({'weights':tf.Variable(tf.random_normal([784, n_nodes[0]])), 
		                        'biases':tf.Variable(tf.random_normal([n_nodes[0]]))})
			op_layer.append(tf.nn.relu(tf.add(tf.matmul(data, h_layer[0]['weights']), 
					h_layer[0]['biases'])))
		else:
			h_layer.append({'weights':tf.Variable(tf.random_normal([n_nodes[_ - 1],n_nodes[_]])),
					'biases':tf.Variable(tf.random_normal([n_nodes[ _ ]]))})
			op_layer.append(tf.nn.relu(tf.add(tf.matmul(op_layer[_ - 1], h_layer[_]['weights']), 
					h_layer[_]['biases']))) 
	opt_layer = {'weights':tf.Variable(tf.random_normal([n_nodes[len(n_nodes) - 1],n_classes])),
			'biases':tf.Variable(tf.random_normal([n_classes]))}

	op = tf.add(tf.matmul(op_layer[len(n_nodes) - 1], opt_layer['weights']) , opt_layer['biases'])

	return op


def trainNN (x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost) #learning_rate = 0.001
	epochs = 30
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
	
		for epoch in range(epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
				epoch_loss += c
			print ('Epoch', epoch, '/', epochs, 'loss', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))



trainNN(x)
	
