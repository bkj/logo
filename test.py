# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# input_       = tf.ones((1, 2, 2, 1))
# output_shape = [1, 4, 4, 1]

# w      = tf.get_variable('w', [3, 3, 1, 1], initializer=tf.ones_initializer)
# deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, 2, 2, 1])

# # --

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())

# print(w.eval().shape)

# z = deconv.eval()

# print(z.shape)
# print(z.squeeze())

# sess.close()

# ------------------------------------------------------------

from main import *

q = torch.rand(1, 3, 64, 64)
x = torch.rand(1, 3, 256, 256)

query_encoder   = QueryEncoder(in_channels=3)
gallery_encoder = GalleryEncoder(in_channels=3)
gallery_decoder = GalleryDecoder(q_channels=512)

query_enc    = query_encoder(q)
gallery_encs = (x1, x2, x3, x4, x5) = gallery_encoder(x)

print('query_enc.shape', query_enc.shape)
print('gallery_encs.shape', [g.shape for g in gallery_encs])

gallery_decoder(gallery_encs, query_enc).shape
