import tensorflow as tf

class MLP(tf.keras.Model):
    """
    genc(·) is a multi-layer perceptron (MLP), whose weights are shared over all nodes; specifically,
    the MLP contains a single fully connected layer followed by layer normalization [3] and then ReLU
    non-linearity.
    """
    def __init__(self, input_dim, output_dim, hidden_dim = 64, no_relu = False):
        super(MLP, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, kernel_initializer=tf.keras.initializers.HeNormal())
        self.fc2 = tf.keras.layers.Dense(output_dim, kernel_initializer=tf.keras.initializers.HeNormal())

        self.norm = tf.keras.layers.LayerNormalization()

        self.activation = tf.keras.activations.relu

        self.no_relu = no_relu

    
    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)

        x = self.norm(x)

        if not self.no_relu:
            x = self.activation(x)

        return x

class SubGraphLayer(tf.keras.Model):
    def __init__(self, len):
        super(SubGraphLayer, self).__init__()
        self.genc = MLP(len, len)

    def call(self, inputs, training = True, mask = None):
        # inputs = (feature_num, feature_size)
        x = self.genc(inputs)

        if mask is not None:
            curr_mask = tf.expand_dims(mask, -1)
            curr_mask = tf.tile(curr_mask, (1, 1, x.shape[-1]))
            x = tf.where(curr_mask, x, -1e9)

        # print(x)

        # print("x shape:")
        # print(x.shape)
        # # x_t = tf.transpose(x, (0, 2, 1))
        # print("x_t shape:")
        # print(x_t.shape)
        x_pool = tf.keras.layers.MaxPool1D(x.shape[1])(x)
        # print("x_pool shape:")
        # print(x_pool.shape)
        # x_t = tf.transpose(x_pool, (0, 2, 1))
        # print(x_t.shape)
        x_tile = tf.tile(x_pool, (1, inputs.shape[1], 1))

        # print(inputs.shape[1])
        # print(x_tile.shape)
        # print(x.shape)

        # x_cat = tf.keras.layers.Concatenate(axis=-1)([x, x_tile])

        x_cat = tf.concat([x, x_tile], axis = -1)

        # print("x_cat shape")
        # print(x_cat.shape)

        return x_cat

class SubGraph(tf.keras.Model):
    def __init__(self, len):
        super(SubGraph, self).__init__()

        self.layer_1 = SubGraphLayer(len)
        self.layer_2 = SubGraphLayer(len * 2)
        self.layer_3 = SubGraphLayer(len * 4)

    def call(self, inputs, training = True, mask = None):
        x = self.layer_1(inputs, training, mask)
        x = self.layer_2(x, training, mask)
        x = self.layer_3(x, training, mask)

        # x_t = tf.transpose(x, (0, 2, 1))
        x_pool = tf.keras.layers.MaxPool1D(x.shape[1])(x)
        # x_t = tf.transpose(x_pool, (0, 2, 1))

        # print("x pool cat shape:")
        # print(x_pool.shape)

        x = tf.squeeze(x_pool, [1])
        
        return x

class GlobalGraph(tf.keras.Model):
    def __init__(self):
        super(GlobalGraph, self).__init__()
        self.attention = tf.keras.layers.Attention(dropout = 0.1, use_scale = True)
        self.fc1 = tf.keras.layers.Dense(units = 64, kernel_initializer=tf.keras.initializers.HeNormal())
        self.fc2 = tf.keras.layers.Dense(units = 64, kernel_initializer=tf.keras.initializers.HeNormal())
        self.fc3 = tf.keras.layers.Dense(units = 64, kernel_initializer=tf.keras.initializers.HeNormal())

    def call(self, inputs, training):
        featues = inputs[0]
        graph_mask = inputs[1]

        query = self.fc1(featues)
        value = self.fc2(featues)
        key = self.fc3(featues)

        x, attention_score = self.attention([query, value, key], mask = [graph_mask, graph_mask], return_attention_scores = True,  training = training)

        # print("x shape:" + str(x.shape) + ", query shape:" + str(query.shape))

        # x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = tf.math.reduce_mean(x, axis = 1)

        return x, attention_score


class VectorNet(tf.keras.Model):
    def __init__(self, arg):
        super(VectorNet, self).__init__()
        
        self.layers_num = 3
        self.global_graph = GlobalGraph()

        self.pred_len = arg.pred_len

        self.polyline_num = 10


    def build(self, input_shape):

        feature_len = input_shape[0][-1]
        self.sub_graph = SubGraph(feature_len)

        batch_size = input_shape[0][0]
        self.decoder = MLP(batch_size, self.pred_len * 5, True)

        sub_graph_output_dim = feature_len * 2**self.layers_num + 2
        self.node_decoder = MLP(batch_size, self.polyline_num * sub_graph_output_dim, True)


    def call(self, inputs, training = True):
        # inputs = [batch_size, sub_graph_size, elem_size, feature_len]

        features = inputs[0]
        masks = inputs[1]
        graph_masks = inputs[2]

        pids = inputs[3]
        node_masks = inputs[4]

        batch_size = features.shape[0]
        sub_graph_size = features.shape[1]
        feature_len = features.shape[-1]
        sub_graph_output_dim = feature_len * 2**self.layers_num + 2

        # features = tf.keras.layers.BatchNormalization()(features, training = training)

        global_graphs = tf.zeros([0, sub_graph_size, sub_graph_output_dim])
        for batch_idx in range(batch_size):
            global_graph = self.sub_graph(features[batch_idx], training, masks[batch_idx])

            global_graph = tf.concat([global_graph, pids[batch_idx]], axis = -1)

            global_graph = tf.expand_dims(global_graph, 0)

            global_graphs = tf.concat([global_graphs, global_graph], axis = 0)

        # output: global_graphs = [batch_size, sub_graph, feature_len * 2**self.layers_num]

        # tf.print(global_graphs)
        x = tf.math.l2_normalize(global_graphs, axis = -1)
        # x = global_graphs

        graph_masks = tf.logical_and(graph_masks, node_masks)

        node_true = tf.boolean_mask(global_graphs, node_masks)

        attention_feature, attention_score = self.global_graph([x, graph_masks], training)

        # output: x = [batch_size, sub_graph, 64]

        x = self.decoder(attention_feature)
        x = tf.reshape(x, [batch_size, self.pred_len, 5])

        node_cmp = self.node_decoder(attention_feature)
        node_cmp = tf.reshape(node_cmp, [batch_size * self.polyline_num,  sub_graph_output_dim])


        return x, node_cmp, node_true
        