import tensorflow as tf

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == '__main__':
    tf_graph = load_pb('./models/model_simple.pb')
    print(tf_graph.get_collection())
    # print(dir(tf_graph))