import tensorflow as tf
from utils import data_process

class model():
	def __init__(self):
		self.images=tf.placeholder(tf.float32,[None,64,64,1])
		self.labels=tf.placeholder(tf.int64,[None])

	def build(self):
		dataset=tf.data.Dataset.from_tensor_slices((self.images,self.labels))
		dataset=dataset.shuffle(buffer_size=3755)
		dataset=dataset.batch(3755)
		dataset=dataset.repeat(1)
		iterator=dataset.make_initializable_iterator()
		return iterator

	def conv_bn(self,inputs,filters,kernel_size):
		conv=tf.layers.conv2d(inputs,filters=filters,kernel_size=kernel_size,strides=1,padding='same')
		bn=tf.layers.batch_normalization(conv)
		return tf.nn.relu(bn)

	def res_block1(self,input):
		conv1=tf.layers.conv2d(input,filters=input.shape[3],kernel_size=1,strides=1,padding='same')
		conv2=self.conv_bn(conv1,filters=input.shape[3],kernel_size=3)
		conv3=tf.layers.conv2d(conv2,filters=input.shape[3],kernel_size=1,strides=1,padding='same')
		conv=tf.add(input,conv3)
		return conv

	def res_block2(self,input):
		filters=2*input.shape[3].value
		conv1=tf.layers.conv2d(input,filters=filters,kernel_size=1,strides=1,padding='same')
		conv2=self.conv_bn(conv1,filters=filters,kernel_size=3)
		conv3=tf.layers.conv2d(conv2,filters=filters,kernel_size=1,strides=1,padding='same')
		conv_input=tf.layers.conv2d(input,filters=filters,kernel_size=1,strides=1,padding='same')
		conv=tf.add(conv_input,conv3)
		return conv

	def network(self,images):
		with tf.name_scope('conv_layer1'):
			conv0=tf.layers.conv2d(images,filters=16,kernel_size=5,strides=1,padding='same')
		with tf.name_scope('pool_layer1'):
			conv_in=tf.layers.max_pooling2d(conv0,pool_size=2,strides=2)
		pool=[]
		conv=[]
		pool_layer=1
		res_layer=0
		for layer in range(12):
			if layer in (2,6):
				pool_layer+=1
				res_layer+=1
				with tf.name_scope('pool_layer'+str(pool_layer)):
					conv_in=tf.layers.max_pooling2d(conv_in,pool_size=2,strides=2)
				pool.append(conv_in)
				with tf.name_scope('res_layer'+str(res_layer)):	
					conv_in=self.res_block2(conv_in)
				conv.append(conv_in)
			elif layer==11:
				pool_layer+=1
				with tf.name_scope('pool_layer'+str(pool_layer)):
					conv_in=tf.layers.average_pooling2d(conv_in,pool_size=2,strides=2)
				pool.append(conv_in)
			else:
				res_layer+=1
				with tf.name_scope('res_layer'+str(res_layer)):
					conv_in=self.res_block1(conv_in)
				conv.append(conv_in)
		with tf.name_scope('fully_connected'):
			fc1=tf.layers.flatten(pool[-1])
			fc2=tf.layers.dense(fc1,units=4096,activation=tf.nn.relu)
			fc3=tf.layers.dense(fc2,units=3755)
		return fc3

	def optimizer(self,images,targets):
		with tf.name_scope('optimizer'):
			logits=self.network(images)
			cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits))
			train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
			correct=tf.equal(tf.argmax(tf.nn.softmax(logits),1),targets)
			accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
		return train_op,cross_entropy,accuracy

	def train(self):
		iterator=self.build()
		next_images,next_labels=iterator.get_next()
		train_op,cross_entropy,accuracy=self.optimizer(next_images,next_labels)
		dataset=data_process('data').load_file()
		num=0
		#for i in range(1000):
		sess=tf.Session()
                sess.run(tf.global_variables_initializer())
                saver=tf.train.Saver(tf.all_variables())
		for i in range(700):
                        image,target=dataset.next()
                        try:
			    sess.run(iterator.initializer,feed_dict={self.images:image,self.labels:target})
                        except:
                            continue
			num+=1
			_,loss,acc=sess.run([train_op,cross_entropy,accuracy])
			if num%10==0:
				saver.save(sess,'variables/handwriting.module',global_step=num)
				print('number %d,loss is %f' % (num,loss))
				print('number %d,accuracy is %f' % (num,acc))
                test_accuracy=0
                for j in range(300):
                        image_test,target_test=dataset.next()
                        try:
                            sess.run(iterator.initializer,feed_dict={self.images:image_test,self.labels:target_test})
                        except:
                            continue
                        acc=sess.run(accuracy)
                        test_accuracy+=acc
                        print(acc)
                print('test accuracy is %f'%(test_accuracy/300.0))

