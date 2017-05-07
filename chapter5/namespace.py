'''
学习tensorflow的命名空间
variable_scope函数：一个上下文管理器
get_variable函数：获取或者创建一个变量
'''

import tensorflow as tf

v1 = tf.get_variable("v", [1])
print (v1.name)

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name)

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)
    v4 = tf.get_variable("v1", [1])
    print(v4.name)

with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1])
    print(v3 == v5)
    v6 = tf.get_variable("foo/v1", [1])
    print(v6 == v4)
