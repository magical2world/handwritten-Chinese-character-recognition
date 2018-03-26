import tensorflow as tf
from network import model
def main():
	# images=tf.placeholder(tf.float32,[None,64,64,3])
	# labels=tf.placeholder(tf.int32,[None])

	net=model()
	net.train()

if __name__=='__main__':
	main()