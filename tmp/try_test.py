import tensorflow as tf

'''
class SquareTest(tf.test.TestCase):
    def testSquare(self):
        with self.test_session():
          x = tf.square([2, 3])
          self.assertAllEqual(x.eval(), [4, 9])


if __name__ == '__main__':
    tf.test.main()
'''
with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')
  
sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()


sess.run(init)