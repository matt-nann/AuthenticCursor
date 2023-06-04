import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.nn.parameter import Parameter

class MiniBatchDiscrimination(nn.Module):
	def __init__(self, A, B, C, batch_size):
		super(MiniBatchDiscrimination, self).__init__()
		self.feat_num = A
		self.out_size = B
		self.row_size = C
		self.N = batch_size
		self.T = Parameter(torch.Tensor(A,B,C))
		self.reset_parameters()

	def forward(self, x):
		# Output matrices after matrix multiplication
		M = x.mm(self.T.view(self.feat_num,self.out_size*self.row_size)).view(-1,self.out_size,self.row_size)
		out = Variable(torch.zeros(self.N,self.out_size))
		for k in range(self.N): # Not happy about this 'for' loop, but this is the best we could do using PyTorch IMO
			c = torch.exp(-torch.sum(torch.abs(M[k,:]-M),2)) # exp(-L1 Norm of Rows difference)
			if k != 0 and k != self.N -1: 
				out[k,:] = torch.sum(c[0:k,:],0) + torch.sum(c[k:-1,:],0)
			else:
				if k == 0:
					out[k,:] = torch.sum(c[1:,:],0)
				else:
					out[k,:] = torch.sum(c[0:self.N-1],0)
		return out

	def reset_parameters(self):
		stddev = 1/self.feat_num
		self.T.data.uniform_(stddev)


# import torch
# import torch.nn as nn
# import torch.nn.init as init

# class MinibatchDiscrimination(nn.Module):
#     def __init__(self, in_features, out_features, kernel_dims):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.kernel_dims = kernel_dims

#         self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
#         init.normal_(self.T, 0, 1)

#     def forward(self, x):
#         # x is NxA
#         # T is AxBxC
#         matrices = x.mm(self.T.view(self.in_features, -1))
#         matrices = matrices.view(-1, self.out_features, self.kernel_dims)

#         M = matrices.unsqueeze(0)  # 1xNxBxC
#         M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
#         norm = torch.abs(M - M_T).sum(3)  # NxNxB
#         expnorm = torch.exp(-norm)
#         o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance

#         x = torch.cat([x, o_b], 1)
#         return x


class MinibatchDiscrimination(Layer):
    def __init__(self, num_filters=None, hidden_filters=None, use_mean=True, name=None):
        super().__init__(name=name)
        self.num_filters = num_filters
        self.hidden_filters = hidden_filters
        self.use_mean = use_mean

    def build(self, *input_shape: TensorShape):
        if not self._built:
            if self.num_filters is None:
                self.num_filters = minimum(self.input_filters // 2, 64)
            if self.hidden_filters is None:
                self.hidden_filters = self.num_filters // 2
            self.register_parameter('weight', Parameter(random_normal((self.input_filters, self.num_filters, self.hidden_filters)).to(get_device())))

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.weight.view(self.input_filters, -1))
        matrices = matrices.view(-1, self.num_filters, self.hidden_filters)
        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)  # NxB, subtract self distance
        if self.use_mean:
            o_b /= x.size(0) - 1
        x = torch.cat([x, o_b], 1)
        return x


class MinibatchDiscriminationLayer(Layer):
    def __init__(self, averaging='spatial', name=None):
        super(MinibatchDiscriminationLayer, self).__init__(name=name)
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode' % self.averaging

    def adjusted_std(self, t, dim=0, keepdim=True):
        return torch.sqrt(torch.mean((t - torch.mean(t, dim=dim, keepdim=keepdim)) ** 2, dim=dim, keepdim=keepdim) + 1e-8)

    def forward(self, x):
        shape = list(x.size())

        target_shape = deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = mean(vals, axis=[2, 3], keepdim=True)  # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = mean(x, [0, 2, 3], keepdim=True)  # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] / self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)


class MinibatchDiscrimination(tf.keras.layers.Layer):

    def __init__(self, num_kernel, dim_kernel,kernel_initializer='glorot_uniform', **kwargs):
        self.num_kernel = num_kernel
        self.dim_kernel = dim_kernel
        self.kernel_initializer = kernel_initializer
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.num_kernel*self.dim_kernel),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(MinibatchDiscrimination, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        activation = tf.matmul(x, self.kernel)
        activation = tf.reshape(activation, shape=(-1, self.num_kernel, self.dim_kernel))
        #Mi
        tmp1 = tf.expand_dims(activation, 3)
        #Mj
        tmp2 = tf.transpose(activation, perm=[1, 2, 0])
        tmp2 = tf.expand_dims(tmp2, 0)
        
        diff = tmp1 - tmp2
        
        l1 = tf.reduce_sum(tf.math.abs(diff), axis=2)
        features = tf.reduce_sum(tf.math.exp(-l1), axis=2)
        return tf.concat([x, features], axis=1)        
        
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.num_kernel)
      
    def get_config(self):
        config = super().get_config()
        config['dim_kernel'] =  self.dim_kernel
        config['num_kernel'] = self.num_kernel
        config["kernel_initializer"] = self.kernel_initializer
        return config