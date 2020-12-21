import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
	return pos * angle_rates

#used to give transformer positional info
def positional_encoding(position, d_model):
	angle_rads = get_angles(np.arange(position)[:,np.newaxis],
		np.arange(d_model)[np.newaxis, :],
		d_model)
	
	# applt sin to even indices
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	
	#apply cos to odd indices in the array
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
	
	pos_encoding = angle_rads[np.newaxis, ...]
	
	return tf.cast(pos_encoding, dtype = tf.float32)

def create_2d_mask(seqs):
	# We mask only those vectors of the sequence in which we have all zeroes 
	# (this is more scalable for some situations).
	mask = tf.cast(tf.reduce_all(tf.math.equal(seqs, 0), axis=-1), tf.float32)
	return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
	# so the model cannot look ahead to future data
	mask = 1 - tf.linalg.band_part(tf.ones((size,size)), -1, 0) #lower triangular
	#try changing -1 and 0 to make upper triangular
	return mask # (seq_len, seq_len)

def create_masks(inp, tar):
	''' create all masks needed for the transformer '''
	
	#encoder padding mask
	enc_padding_mask = create_2d_mask(inp)
	
	#padding mask for encoder outputs in 2nd attention block in decoder
	dec_padding_mask = create_2d_mask(inp)
	
	#used in the 1st attention block in the decoder
	#to mask future tokens
	look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
	combined_mask = tf.maximum(dec_padding_mask[..., 1:], look_ahead_mask)
	
	return enc_padding_mask, combined_mask, dec_padding_mask

def create_loss_mask(inp):
	''' mask for the loss function '''
	mask = tf.cast(tf.reduce_all(tf.math.equal(inp, 0), axis=-1), tf.float32)
	return mask[:, 1:]  # (batch_size, seq_len - 1)

class EntityEmbedding(tf.keras.layers.Layer):
	''' embedding of multiple features'''
	
	def __init__(self, vocab_sizes, emb_dims):
		super(EntityEmbedding, self).__init__()
		assert len(vocab_sizes) == len(emb_dims) # check consistent
		
		self.emb_layers = [tf.keras.layers.Embedding(vocab, emb) for vocab, emb in zip(vocab_sizes, emb_dims)]
		self.concat = tf.keras.layers.Concatenate(axis = -1)
		
	def call(self, x):
		
		x = self.concat([embedding(x[..., i]) for i, embedding in enumerate(self.emb_layers)])
		
		return x

def scaled_dot_product_attention(q, k, v, mask):
	'''
	q: query = (..., seq_len_q, depth)
	k: key = (..., seq_len_k, depth)
	v: value = (..., seq_len_v, depth_v)
	mask: float tensor with shape broadcastable to
		(..., seq_len_q, seq_len_k)
		
	must have seq_len_k == seq_len_v
		
	Returns:
		output, attention_weights
	'''
	
	# Q @ K^T
	matmul_qk = tf.matmul(q, k, transpose_b = True) # (..., seq_len_q, seq_len_k)
	
	# (Q @ K^T) / sqrt(d_k)
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
	
	#mask
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)
	
	#softmax(.)
	#(..., seq_len_q, seq_len_k)
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
	
	#(Weights @ V)
	output = tf.matmul(attention_weights, v) #(..., seq_len_q, depth_v_)
	
	return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
	
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model
		
		assert d_model % self.num_heads == 0
		
		self.depth = d_model // self.num_heads
		
		self.wq = tf.keras.layers.Dense(d_model)
		self.wk = tf.keras.layers.Dense(d_model)
		self.wv = tf.keras.layers.Dense(d_model)
		
		self.dense = tf.keras.layers.Dense(d_model)
		
	def split_heads(self, x, batch_size):
		'''
		Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		
		x : (batch_size, seq_len, d_model)
		batch_size : int
		
		Returns:
		x : (batch_size, num_heads, seq_len, depth)
		'''
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm = [0, 2, 1, 3])
	
	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]
		
		#linear
		# (batch_size, seq_len, d_model)
		q = self.wq(q)
		k = self.wk(k)
		v = self.wv(v)
		
		#split into heads
		q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
				
		#Scaled Dot-Product Attention
		# scaled_attention (batch_size, num_heads, seq_len_q, depth)
		# attention_weights (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(
			q, k, v, mask)
		
		# (batch, seq_len_q, num_heads, depth)
		scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
		
		#concat
		# (batch_size, seq_len_q, d_model)
		concat_attention = tf.reshape(scaled_attention, 
									  (batch_size, -1, self.d_model))
									  
		output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)
									  
		return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
	'''
	A pointwise feed forward network is two fully connected
	layers with a ReLU activation in between.
	'''
	return tf.keras.Sequential([
		tf.keras.layers.Dense(dff, activation = 'relu'), # (batch_size, seq_len, dff)
		tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
	])

class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate = 0.1):
		'''
		d_model = model dimension
		num_heads = number of heads
		dff = dimension of feed forward network
		rate = dropout rate
		'''    
		super(EncoderLayer, self).__init__()
		
		self.mha = MultiHeadAttention(d_model, num_heads)
		self.ffn = point_wise_feed_forward_network(d_model, dff)
		
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
		
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		
	def call(self, x, training, mask):
		
		attn_output, _ = self.mha(x, x, x, mask) # (bath_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training = training)
		out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)
		
		ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
		ffn_output = self.dropout2(ffn_output, training = training)
		out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)
		
		return out2

class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate = 0.1):
		super(DecoderLayer, self).__init__()
		
		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)
		
		self.ffn = point_wise_feed_forward_network(d_model, dff)
		
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
		
		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.dropout3 = tf.keras.layers.Dropout(rate)
		
	def call(self, x, enc_output, training,
			look_ahead_mask, padding_mask):
		# enc_output shape == (batch_size, input_seq_len, d_model)
		
		attn1, _ = self.mha1(x, x, x, look_ahead_mask) # (batch_size, target_seq_len, d_model)
		attn1 = self.dropout1(attn1, training = training)
		out1 = self.layernorm1(attn1 + x)
		
		attn2, _ = self.mha2(enc_output, enc_output, out1, padding_mask) # (batch_size, target_seq_len, d_model)
		attn2 = self.dropout2(attn2, training = training)
		out2 = self.layernorm1(attn2 + out1) # (batch_size, target_seq_len, d_model)
		
		ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
		ffn_output = self.dropout3(ffn_output, training = training)
		out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, d_model)
		
		return out3

class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, emb_dims, num_heads, dff, input_vocab_sizes,
				 maximum_position_encoding, rate = 0.1):
		super(Encoder, self).__init__()
		
		self.d_model = np.sum(emb_dims)
		self.num_layers = num_layers
		
		self.embedding = EntityEmbedding(input_vocab_sizes, emb_dims)
		
		self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
		
		self.enc_layers = [EncoderLayer(self.d_model, num_heads[i], dff, rate) for i in range(num_layers)]
		
		self.dropout = tf.keras.layers.Dropout(rate)
		
	def call(self, x, training, mask):
		
		#x shape is batch_size x input_seq_len
		
		seq_len = tf.shape(x)[1]
				
		#adding embedding and position encoding
		x = self.embedding(x) # (batch_size, input_seq_len, d_model)
		
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # times by sqrt(d_model)
		
		x += self.pos_encoding[:, :seq_len, :]
		
		x = self.dropout(x, training = training)
		
		for i in range(self.num_layers):
			x = self.enc_layers[i](x, training, mask)
			
		return x # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, emb_dims, num_heads, dff, target_vocab_sizes,
				maximum_position_encoding, rate = 0.1):
		super(Decoder, self).__init__()
		
		self.d_model = np.sum(emb_dims)
		self.num_layers = num_layers
		
		self.embedding = EntityEmbedding(target_vocab_sizes, emb_dims)
		
		self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
		
		self.dec_layers = [DecoderLayer(self.d_model, num_heads[i], dff, rate) for i in range(num_layers)]
		
		self.dropout = tf.keras.layers.Dropout(rate)
	
	def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
		
		seq_len = tf.shape(x)[1]
		attention_weights = {}
		
		x = self.embedding(x) # (batch_size, target_seq_len, d_model)
		
		x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
			
		x = x + self.pos_encoding[:, :seq_len, :]
		
		x = self.dropout(x, training=training)
		
		for i in range(self.num_layers):
			x = self.dec_layers[i](x, enc_output, training,
								   look_ahead_mask, padding_mask)
			
		return x

class Transformer(tf.keras.Model):
	def __init__(self, num_layers, num_heads, dff,
				 input_emb_dims, target_emb_dims,
				 input_vocab_sizes, target_vocab_sizes,
				 pe_input, pe_target, rate = 0.1):
		
		super(Transformer, self).__init__()
		
		assert np.sum(input_emb_dims) == np.sum(target_emb_dims)
		d_model = np.sum(input_emb_dims)
		
		self.encoder = Encoder(num_layers, input_emb_dims, num_heads, dff,
							   input_vocab_sizes, pe_input, rate)
		
		self.decoder = Decoder(num_layers, target_emb_dims, num_heads, dff,
							  target_vocab_sizes, pe_target, rate)
		
		self.final_layer = tf.keras.layers.Dense(target_vocab_sizes[0])
		
	def call(self, inp, tar, training):
		
		enc_mask, look_ahead_mask, dec_mask = create_masks(inp, tar)
		
		enc_output = self.encoder(inp, training, enc_mask) # (batch_size, inp_seq_len, d_model)
		
		# (batch_size, tar_seq_len, d_model)
		dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, dec_mask)
		
		# (batch_size, tar_seq_len, target_vocab_size
		final_output = self.final_layer(dec_output)
		
		return final_output